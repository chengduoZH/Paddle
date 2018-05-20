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

#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"

#include <string>

namespace paddle {
namespace framework {
namespace details {
ScaleLossGradOpHandle::ScaleLossGradOpHandle(size_t num_dev,
                                             const ExecutionContext &exe_ctx,
                                             platform::DeviceContext *dev_ctx)
    : coeff_(static_cast<float>(1.0 / num_dev)), exe_ctx_(exe_ctx) {
  dev_ctxes_[exe_ctx_.place] = dev_ctx;
}

ScaleLossGradOpHandle::~ScaleLossGradOpHandle() {}

void ScaleLossGradOpHandle::RunImpl() {
  // Doesn't wait any event
  std::string var_name = static_cast<VarHandle *>(this->outputs_[0])->name_;
  auto &local_scope =
      *exe_ctx_.scope->FindVar(kLocalExecScopeName)->Get<Scope *>();

  float *tmp = local_scope.FindVar(var_name)
                   ->GetMutable<LoDTensor>()
                   ->mutable_data<float>(make_ddim({1}), exe_ctx_.place);

  if (platform::is_cpu_place(exe_ctx_.place)) {
    *tmp = coeff_;
  } else {
#ifdef PADDLE_WITH_CUDA
    this->RunAndRecordEvent([&] {
      auto stream = static_cast<platform::CUDADeviceContext *>(
                        this->dev_ctxes_[exe_ctx_.place])
                        ->stream();
      memory::Copy(boost::get<platform::CUDAPlace>(exe_ctx_.place), tmp,
                   platform::CPUPlace(), &coeff_, sizeof(float), stream);
      VLOG(1) << exe_ctx_.place << "RUN Scale loss grad op";
    });
#endif
  }
}

std::string ScaleLossGradOpHandle::Name() const { return "Scale LossGrad"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
