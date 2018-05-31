/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fuse_vars_op.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {
using framework::Tensor;

class FuseVarOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("X"), "Inputs(X) should not be null");
    PADDLE_ENFORCE(ctx->HasOutputs("Y"),
                   "Output(Out) of FuseVarOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutputs("FusedX"),
                   "Output(FusedX) of FuseVarOp should not be null.");

    if (ctx->IsRuntime()) {
      return;
    }

    auto in_vars_type = ctx->GetInputsVarType("X");
    PADDLE_ENFORCE_EQ(in_vars_type[0],
                      paddle::framework::proto::VarType::LOD_TENSOR);
    for (auto in_var_type : in_vars_type) {
      PADDLE_ENFORCE_EQ(in_var_type, in_vars_type[0]);
    }

    auto var_dims = ctx->GetInputsDim("X");
    PADDLE_ENFORCE_GT(var_dims.size(), 0, "Input tensors count should > 0.");
    int64_t total_numel = 0;
    for (auto dim : var_dims) {
      int64_t numel = paddle::framework::product(dim);
      PADDLE_ENFORCE_GE(numel, 0);
      total_numel += numel;
    }
    ctx->SetOutputDim("Out", framework::DDim({total_numel}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto y_vars = ctx.OutputVar("FusedX");
    auto type = y_vars->GetMutable<framework::LoDTensor>()->type();
    return framework::OpKernelType(static_cast<framework::proto::VarType::Type>(
                                       framework::ToDataType(type)),
                                   ctx.device_context());
  }
};

class FuseVarOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(vector<Tensor>) The input tensors of FuseVar operator.")
        .AsDuplicable();
    AddOutput("FusedX", "(Tensor) The output tensor of FuseVar operator.");
    AddOutput("Y", "(vector<Tensor>) The output tensor of FuseVar operator.")
        .AsDuplicable();
    AddComment(R"DOC(
Sum operator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fuse_vars, ops::FuseVarOp, ops::FuseVarOpMaker);
REGISTER_OP_CPU_KERNEL(
    fuse_vars, ops::FuseVarKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FuseVarKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FuseVarKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FuseVarKernel<paddle::platform::CPUDeviceContext, int64_t>);
