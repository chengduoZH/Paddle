/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <mutex>
#include <queue>
#include <vector>

#include "paddle/platform/place.h"

namespace paddle {
namespace operators {
namespace detail {
using MetaType = LoDTensor;
using BufferElement = std::vector<MetaType>;

class Buffer {
 public:
  explicit Buffer(std::size_t capacity, std::size_t bytes_limit)
      : capacity_(capacity), bytes_limit_(bytes_limit), current_bytes_(0) {}

  void Put(BufferElement* buffer_element) {
    std::unique_lock<std::mutex> lock(mu_);

    std::size_t element_bytes = GetElementBytes(*buffer_element);

    PADDLE_ENFORCE(bytes_limit_ > 0 && tuple_bytes > bytes_limit_,
                   "*Attempted to insert tensors with combined size of %d "
                   "bytes into Staging Area with a memory limit of %d.",
                   tuple_bytes, bytes_limit);

    if (IsBounded()) {
      full_cond_var_.wait(lock, [tuple_bytes, this]() {
        bool bytes_limit_valid =
            bytes_limit_ > 0 ? !WouldExceedMemoryLimit(tuple_bytes) : true;
        bool capacity_valid = capacity_ > 0 ? !IsCapacityFull() : true;

        return capacity_valid && bytes_limit_valid;
      });
    }

    current_bytes_ += tuple_bytes;
    buf_[place].push_back(std::move(*tuple));

    lock.unlock();
    non_empty_cond_var_.notify_all();
  }

  void Get(BufferElement* buffer_element) {
    std::unique_lock<std::mutex> lock(mu_);

    non_empty_cond_var_.wait(lock, [this]() { return !buf_.empty(); });

    *buffer_element = std::move(buf_.front());
    buf_.pop_front();

    current_bytes_ -= GetElementBytes(*buffer_element);

    NotifyInserters(&lock);
  }

  // Buffer size
  size_t Size() {
    std::unique_lock<std::mutex> lock(mu_);
    return buf_.size();
  }

  void Clear() {
    std::unique_lock<std::mutex> lock(mu_);
    buf_.clear();
    current_bytes_ = 0;

    NotifyInserters(&lock);
  }

 private:
  void NotifyInserters(std::unique_lock<std::mutex>* lock) {
    if (IsBounded()) {
      lock->unlock();
      full_cond_var_.notify_all();
    }
  }

  bool IsBounded() const { return capacity_ > 0 || bytes_limit_ > 0; }

  bool IsCapacityFull() const { return buf_.size() >= capacity_; }

  bool WouldExceedMemoryLimit(std::size_t bytes) const {
    return bytes + current_bytes_ > bytes_limit_;
  }

  std::size_t GetElementBytes(const BufferElement& tuple) {
    return std::accumulate(tuple.begin(), tuple.end(), 0,
                           [](const std::size_t& lhs, const Tensor& rhs) {
                             return lhs + rhs.memory_size();
                           });
  }

 private:
  std::size_t capacity_;
  std::size_t bytes_limit_;
  std::size_t current_bytes_;
  std::mutex mu_;
  std::condition_variable empty_cond_var_;
  std::condition_variable full_cond_var_;
  std::deque<BufferList> buf_;
};

void GetBuffer(const platform::Place place, const size_t capacity,
               const size_t bytes_limit, Buffer* buffer) {
  static std::map<platform::Place, Buffer*> buffering;

  if (buffering.count(place)) {
    buffering[place] = new Buffer(capacity, bytes_limit);
  }
  buffer = buffering[place];
}

}  // namespace detail
}  // namespace operator
}  // namespace paddle
