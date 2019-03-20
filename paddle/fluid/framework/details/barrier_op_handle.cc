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

#include "paddle/fluid/framework/details/barrier_op_handle.h"

#include <string>
#include <vector>

namespace paddle {
namespace framework {
namespace details {

BarrierOpHandle::BarrierOpHandle(ir::Node *node) : OpHandleBase(node) {}

BarrierOpHandle::~BarrierOpHandle() {}

void BarrierOpHandle::RecordWaitEventOnCtx(
    platform::DeviceContext *waited_ctx) {}

void BarrierOpHandle::RunImpl() {}

std::string BarrierOpHandle::Name() const { return "Barrier"; }

}  // namespace details
}  // namespace framework
}  // namespace paddle
