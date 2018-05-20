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

#pragma once

#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/framework/details/execution_context.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

struct ReduceOpHandle : public OpHandleBase {
  //  const std::vector<Scope *> &local_scopes_;
  //  const std::vector<platform::Place> &places_;
  const std::vector<ExecutionContext> exe_ctxs_;

#ifdef PADDLE_WITH_CUDA
  const platform::NCCLContextMap *nccl_ctxs_;
  ReduceOpHandle(const std::vector<ExecutionContext> &exe_ctxs,
                 const platform::NCCLContextMap *nccl_ctxs)
      : exe_ctxs_(exe_ctxs), nccl_ctxs_(nccl_ctxs) {
    if (nccl_ctxs_) {
      for (auto &p_ctx : nccl_ctxs_->contexts_) {
        dev_ctxes_[platform::CUDAPlace(p_ctx.first)] = p_ctx.second.ctx_.get();
      }
    }
  }
#else
  explicit ReduceOpHandle(const std::vector<ExecutionContext> &exe_ctxs)
      : exe_ctxs_(exe_ctxs) {}
#endif

  std::string Name() const override;

  bool IsMultiDeviceTransfer() override { return true; };

 protected:
  void RunImpl() override;

  template <typename T>
  std::vector<const T *> GetInputValues(
      const std::vector<VarHandle *> &in_var_handles,
      const std::vector<const Scope *> &var_scopes) const;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
