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

#include <vector>

#include "paddle/platform/place.h"

namespace paddle {
namespace operators {
namespace detail {
using MetaType = LoDTensor;
using BufferElement = std::vector<MetaType>;

class Buffer {
 private:
  std::size_t capacity_;
  std::size_t bytes_limit_;
  std::size_t current_bytes_;
  std::mutex mu_;
  std::condition_variable empty_cond_var_;
  std::condition_variable full_cond_var_;
  std::map<platform::Place, std::deque<BufferList>> buf_;

 public:
  void Put(BufferElement* tuple) {
    //...
  }
  void Get(BufferElement* tuple) {
    //...
  }
  size_t Size() {
    //...
  }
  void Clear() {
    //...
  }
};
}  // namespace detail
}  // namespace operator
}  // namespace paddle
