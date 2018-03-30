//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <thread>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

static constexpr size_t kDoubleBufferSize = 2;

class DoubleBufferReader : public framework::DecoratedReader {
 public:
  struct Item {
    Item() : ctx_(nullptr) {}

    std::vector<framework::LoDTensor> payloads_;
    platform::DeviceContext* ctx_;
  };

  explicit DoubleBufferReader(
      ReaderBase* reader, platform::Place target_place = platform::CPUPlace())
      : DecoratedReader(reader), place_(target_place) {
    for (size_t i = 0; i < kDoubleBufferSize; ++i) {
      if (platform::is_gpu_place(place_)) {
#ifdef PADDLE_WITH_CUDA
        ctxs_.emplace_back(new platform::CUDADeviceContext(
            boost::get<platform::CUDAPlace>(place_)));
#endif
      }
    }

    start_thread();
  }

  void start_thread() {
    buffer_ = framework::MakeChannel<Item>(kDoubleBufferSize);
    empty_buffer_ = framework::MakeChannel<Item>(kDoubleBufferSize);
    for (int j = 0; j < kDoubleBufferSize; ++j) {
      Item* item = new Item();
      empty_buffer_->Send(item);
    }
    local = nullptr;
    prefetcher_ = std::thread([this] { PrefetchThreadFunc(); });
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  void ReInit() override;

  ~DoubleBufferReader() {
    buffer_->Close();
    empty_buffer_->Close();
    prefetcher_.join();
    delete buffer_;
  }

  bool HasNext() const override;

 private:
  void PrefetchThreadFunc();

  std::thread prefetcher_;
  framework::Channel<Item>* buffer_;
  framework::Channel<Item>* empty_buffer_;
  platform::Place place_;
  std::vector<std::unique_ptr<platform::DeviceContext>> ctxs_;
  Item* local;
  mutable Item local_buffer_;
};

class CreateDoubleBufferReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();

    auto place_str = Attr<std::string>("place");
    platform::Place place;
    if (place_str == "CPU") {
      place = platform::CPUPlace();
    } else {
      std::istringstream sin(place_str);
      sin.seekg(std::string("CUDA:").size(), std::ios::beg);
      size_t num;
      sin >> num;
      place = platform::CUDAPlace(static_cast<int>(num));
    }

    out->Reset(new DoubleBufferReader(underlying_reader.Get(), place));
  }
};

class CreateDoubleBufferReaderOpMaker : public DecoratedReaderMakerBase {
 public:
  CreateDoubleBufferReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : DecoratedReaderMakerBase(op_proto, op_checker) {
    AddComment(R"DOC(
      CreateDoubleBufferReader Operator

      A double buffer reader takes another reader as its 'underlying reader'.
      It launches another thread to execute the 'underlying reader' asynchronously, 
      which prevents reading process from blocking subsequent training.
    )DOC");
    std::unordered_set<std::string> enum_range;
    constexpr size_t kMaxCUDADevs = 128;
    for (size_t i = 0; i < kMaxCUDADevs; ++i) {
      enum_range.insert(string::Sprintf("CUDA:%d", i));
    }
    enum_range.insert("CPU");
    AddAttr<std::string>("place", "The double buffer place, default is CPU")
        .SetDefault("CPU")
        .InEnum({enum_range});
  }
};

void DoubleBufferReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  if (!HasNext()) {
    PADDLE_THROW("There is no next data!");
  }

  if (local == nullptr || local->payloads_.empty()) {
    buffer_->Receive(local);
  }
  *out = local->payloads_;
  local->payloads_.clear();
  if (local->ctx_) {
    local->ctx_->Wait();
  }
  empty_buffer_->Send(local);
  local = nullptr;
}

void DoubleBufferReader::ReInit() {
  reader_->ReInit();
  buffer_->Close();
  prefetcher_.join();
  delete buffer_;
  start_thread();
}

void DoubleBufferReader::PrefetchThreadFunc() {
  VLOG(5) << "A new prefetch thread starts.";
  size_t gpu_ctx_offset = 0;
  while (reader_->HasNext()) {
    Item batch;
    Item* gpu_batch = nullptr;
    reader_->ReadNext(&(batch.payloads_));
    if (platform::is_gpu_place(place_)) {
      try {
        empty_buffer_->Receive(gpu_batch);
      } catch (paddle::platform::EnforceNotMet e) {
        VLOG(5) << "WARNING: The double buffer channel has been closed. The "
                   "prefetch thread will terminate.";
        break;
      }
      //      std::vector<framework::LoDTensor> gpu_batch;
      gpu_batch->ctx_ = this->ctxs_[gpu_ctx_offset++].get();
      //      auto& gpu_ctx = this->ctxs_[gpu_ctx_offset++];
      gpu_ctx_offset %= this->ctxs_.size();
      gpu_batch->payloads_.resize(batch.payloads_.size());
      for (size_t i = 0; i < gpu_batch->payloads_.size(); ++i) {
        framework::TensorCopy(batch.payloads_[i], place_, *(gpu_batch->ctx_),
                              &(gpu_batch->payloads_[i]));
        gpu_batch->payloads_[i].set_lod(batch.payloads_[i].lod());
      }
      //      batch.ctx_ = gpu_ctx.get();
      //      std::swap(gpu_batch, batch.payloads_);
    }

    try {
      buffer_->Send(gpu_batch);
    } catch (paddle::platform::EnforceNotMet e) {
      VLOG(5) << "WARNING: The double buffer channel has been closed. The "
                 "prefetch thread will terminate.";
      break;
    }
  }
  buffer_->Close();
  VLOG(5) << "Prefetch thread terminates.";
}

bool DoubleBufferReader::HasNext() const {
  if (local == nullptr || local->payloads_.empty()) {
    bool ok = buffer_->Receive(local);
    return ok;
  } else {
    return true;
  }
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_double_buffer_reader,
                                   ops::CreateDoubleBufferReaderOp,
                                   ops::CreateDoubleBufferReaderOpMaker);
