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

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include <string>

namespace paddle {
namespace framework {
namespace details {
ComputationOpHandle::ComputationOpHandle(const OpDesc &op_desc,
                                         const ExecutionContext &exe_ctx)
    : op_(framework::OpRegistry::CreateOp(op_desc)), exe_ctx_(exe_ctx) {}

void ComputationOpHandle::RunImpl() {
  WaitInputVarGenerated(exe_ctx_.place);

  this->RunAndRecordEvent([this] {
    op_->Run(*exe_ctx_.scope->FindVar(kLocalExecScopeName)->Get<Scope *>(),
             exe_ctx_.place);
  });
}

bool ComputationOpHandle::NeedWait(VarHandleBase *in_var) {
  bool need_wait = in_var && in_var->generated_op_ &&
                   in_var->generated_op_->DeviceContext(exe_ctx_.place) !=
                       dev_ctxes_[exe_ctx_.place];
  return need_wait;
}

std::string ComputationOpHandle::Name() const { return op_->Type(); }
}  // namespace details
}  // namespace framework
}  // namespace paddle
