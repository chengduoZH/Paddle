// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

static framework::proto::VarType::Type kDefaultDtype =
    framework::proto::VarType::Type::VarType_Type_BOOL;

template <typename DeviceContext, typename T>
class AllocContinuousSpace2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &in_var_names = context.Inputs("Input");
    auto &out_var_names = context.Outputs("Output");
    auto &in_vars = context.MultiInputVar("Input");
    auto out_vars = context.MultiOutputVar("Output");

    PADDLE_ENFORCE_GT(in_var_names.size(), static_cast<size_t>(0));
    PADDLE_ENFORCE_EQ(in_var_names.size(), out_var_names.size());

    for (size_t i = 0; i < in_var_names.size(); ++i) {
      // Only support LoDTensor
      PADDLE_ENFORCE_NOT_NULL(in_vars[i], "%s should not be nullptr,",
                              in_var_names[i]);
      PADDLE_ENFORCE_NOT_NULL(out_vars[i], "%s should not be nullptr,",
                              out_var_names[i]);
      PADDLE_ENFORCE(in_vars[i]->IsType<framework::LoDTensor>());
      PADDLE_ENFORCE(out_vars[i]->IsType<framework::LoDTensor>());
    }

    auto in_tensors = context.MultiInput<framework::LoDTensor>("Input");

    if (context.Attr<bool>("check_name")) {
      for (size_t i = 0; i < in_var_names.size(); ++i) {
        PADDLE_ENFORCE_EQ(in_var_names[i], out_var_names[i]);
      }
    } else {
      // Init the output as input
      for (size_t i = 0; i < in_tensors.size(); ++i) {
        out_vars[i]->GetMutable<framework::LoDTensor>()->Resize(
            in_tensors[i]->dims());
      }
    }

    auto &dev_ctx = context.template device_context<DeviceContext>();
    // Init the continuous space
    auto out_tensors = context.MultiOutput<framework::LoDTensor>("Output");
    for (size_t i = 0; i < in_var_names.size(); ++i) {
      framework::Tensor tmp;
      framework::TensorCopy(*in_tensors[i], context.GetPlace(), dev_ctx, &tmp);
      out_tensors[i]->ShareDataWith(tmp);
    }
  }

  void GetMemSizeAndDtype(
      const std::vector<const framework::LoDTensor *> &lod_tensors,
      const std::vector<std::string> var_names, size_t *numel,
      framework::proto::VarType::Type *dtype) const {
    PADDLE_ENFORCE_EQ(lod_tensors.size(), var_names.size());
    *numel = 0;
    for (size_t i = 0; i < var_names.size(); ++i) {
      PADDLE_ENFORCE(lod_tensors[i]->IsInitialized(), "%s is not initialized.",
                     var_names[i]);

      auto p_dtype = lod_tensors[i]->type();
      if (*dtype == kDefaultDtype) {
        PADDLE_ENFORCE_NE(p_dtype, kDefaultDtype, "%s's type should not be %s.",
                          var_names[i], kDefaultDtype);
        *dtype = p_dtype;
      }
      PADDLE_ENFORCE_EQ(p_dtype, *dtype, "Input vars is not equal.");

      auto size = lod_tensors[i]->numel();
      PADDLE_ENFORCE_GT(size, 0);
      VLOG(10) << "alloc_space_for_vars: input(" << var_names[i] << ") ,dim:("
               << lod_tensors[i]->dims() << ")";
      *numel += size;
    }
  }
};

class AllocContinuousSpace2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}
};

class AllocContinuousSpace2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(vector<LoDTensor>) The input tensors of"
             " alloc_continuous_space2 operator.")
        .AsDuplicable();
    AddOutput("Output",
              "(vector<LoDTensor>) The output "
              "tensors of alloc_continuous_space2 operator. And the address "
              "of output tensors are continuous, they are sliced from the "
              "tensor of FusedOutput.")
        .AsDuplicable();
    AddOutput("FusedOutput",
              "(LoDTensor) The output tensor "
              "of alloc_continuous_space2 operator. And the tensors of"
              " Output is sliced from the tensor of FusedOutput.");
    AddAttr<bool>("copy_data", "Whether to copy the Input value to Output.")
        .SetDefault(false);
    AddAttr<bool>("set_constant",
                  "Whether to set the Output with a constant value.")
        .SetDefault(false);
    AddAttr<float>("constant",
                   "If set_constant is true, the constant value will be used "
                   "to set the Output.")
        .SetDefault(0.0);
    AddAttr<bool>("check_name",
                  "Whether to check the name of Input and Output to ensure "
                  "they are the same separately.")
        .SetDefault(false);
    AddComment(R"DOC(
AllocContinuousSpace2 Operator.

alloc_continuous_space2 is used to make the address of Output
continuous according to the Input. This Op will alloc a big tensor
according to the tensors of Input, the dtype is the same with those input tensors,
the size is the sum of those input tensors' numel, and the dim of the big
tensor is {sum(numel)}. And the big tensor is stored in FusedOutput.
The tensors of Output are sliced from the tensor of FusedOutput.
Note that, the dtype of Input should be the same, and the dim of Input
and Output should equal.
The tensors of Input and Output could be the same or different. And
alloc_continuous_space2 allows copying the value of Input to Output, or
setting the Output with a constant value.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(alloc_continuous_space2,
                  paddle::operators::AllocContinuousSpace2Op,
                  paddle::operators::AllocContinuousSpace2OpMaker);
namespace ops = paddle::operators;
REGISTER_OP_CPU_KERNEL(
    alloc_continuous_space2,
    ops::AllocContinuousSpace2Kernel<paddle::platform::CPUDeviceContext, int>,
    ops::AllocContinuousSpace2Kernel<paddle::platform::CPUDeviceContext, float>,
    ops::AllocContinuousSpace2Kernel<paddle::platform::CPUDeviceContext,
                                     double>);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    alloc_continuous_space2,
    ops::AllocContinuousSpace2Kernel<paddle::platform::CUDADeviceContext, int>,
    ops::AllocContinuousSpace2Kernel<paddle::platform::CUDADeviceContext,
                                     float>,
    ops::AllocContinuousSpace2Kernel<paddle::platform::CUDADeviceContext,
                                     double>);
#endif
