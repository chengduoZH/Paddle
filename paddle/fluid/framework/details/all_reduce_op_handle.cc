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

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"

namespace paddle {
namespace framework {
namespace details {
AllReduceOpHandle::AllReduceOpHandle(const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places,
                                     const platform::NCCLContextMap &ctxs,
                                     const int device_count)
    : local_scopes_(local_scopes),
      places_(places),
      nccl_ctxs_(ctxs),
      device_count_(device_count) {
  for (auto &p : places_) {
    this->dev_ctxes_[p] = nccl_ctxs_.DevCtx(p);
  }
}

void AllReduceOpHandle::RunImpl() {
  auto &var_name = static_cast<VarHandle *>(this->inputs_[0])->name_;

  all_reduce_calls_ = std::thread([this]() {
    // sum
    Variable var;
    ParameterCollection::Instance().Get(var_name)->Receive(&var);
    Tensor reducer;
    Tensor temp;
    if (var.IsType<framework::SelectedRows>()) {
      // reduce sparse parameter
    } else if (var.IsType<framework::LoDTensor>()) {
      auto param = var.Get<LoDTensor>();
      reducer.mutable_data(param.dims(), platform::CUDAPinnedPlace());
      framework::TensorCopy(reducer, platform::CUDAPinnedPlace(), dev_ctx,
                            param);
      temp.mutable_data(param.dims(), platform::CUDAPinnedPlace());
    } else {
      PADDLE_THROW("Parameter should be LoDTensor or SelectedRows");
    }

    // TODO(zcd): float should be T
    float *reducer_ptr = reducer.data<float>();
    for (int j = 0; j < device_count_ - 1; ++j) {
      Variable var_2;
      ParameterCollection::Instance().Get(var_name)->Receive(&var_2);
      PADDLE_ENFORCE(var_2.Type() == var.Type());

      if (var.IsType<framework::SelectedRows>()) {
        // TODO(zcd): reduce sparse parameter

      } else if (var.IsType<framework::LoDTensor>()) {
        auto param = var_2.Get<LoDTensor>();
        PADDLE_ENFORCE_EQ(reducer.numel(), param.numel());
        framework::TensorCopy(temp, platform::CUDAPinnedPlace(), dev_ctx,
                              param);

        // TODO(zcd): Wait
        float *temp_ptr = temp.data<float>();
        for (int k = 0; k < reducer.numel(); ++k) {
          reducer_ptr[k] += temp_ptr[k];
        }
      }
    }

    // broadcast
    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      auto &p = places_[i];
      auto *s = local_scopes_[i];
      int dev_id = boost::get<platform::CUDAPlace>(p).device;
      auto var = s->FindVar(var_name);
      if (var.IsType<framework::SelectedRows>()) {
        // TODO(zcd): reduce sparse parameter

      } else if (var.IsType<framework::LoDTensor>()) {
        auto &lod_tensor = var->Get<LoDTensor>();
        void *buffer = const_cast<void *>(lod_tensor.data<void>());
        if (numel == 0) {
          numel = static_cast<size_t>(lod_tensor.numel());
        }

        auto &nccl_ctx = nccl_ctxs_.at(dev_id);
        auto stream = nccl_ctx.stream();

        framework::TensorCopy(lod_tensor, p, dev_ctx, reducer);
        // Wait(???)
      }
    }
  });
}

std::string AllReduceOpHandle::Name() const { return "nccl_all_reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
