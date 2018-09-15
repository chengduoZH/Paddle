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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

class FillConstantAsLodTensorArrayInferShape
    : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(Out) of FillConstantAsLodTensorArrayOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("X"),
        "Input(X) of FillConstantAsLodTensorArrayOp should not be null.");

    auto var_types = ctx->GetInputsVarType("X");
    PADDLE_ENFORCE(var_types.size(), 1);
    //    PADDLE_ENFORCE(var_types.at[0] ==
    //                   framework::proto::VarType::LOD_TENSOR_ARRAY);
    //    ctx->SetOutputDim("Out", framework::make_ddim(shape));
  }
};

class FillConstantAsLodTensorArrayOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto value = Attr<float>("value");
    //    auto force_cpu = Attr<bool>("force_cpu");

    auto &out_var = *scope.FindVar(Output("Out"));
    auto &in_var = *scope.FindVar(Input("X"));

    PADDLE_ENFORCE(out_var.IsType<framework::LoDTensorArray>());
    PADDLE_ENFORCE(in_var.IsType<framework::LoDTensorArray>());

    framework::LoDTensorArray src_lod_tensor_array =
        out_var.Get<framework::LoDTensorArray>();

    framework::LoDTensorArray *dst_lod_tensor_array =
        out_var.GetMutable<framework::LoDTensorArray>();

    dst_lod_tensor_array->resize(src_lod_tensor_array.size());

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);

    for (size_t j = 0; j < src_lod_tensor_array.size(); ++j) {
      if (src_lod_tensor_array[j].numel() != 0) {
        auto src_lod_tensor = src_lod_tensor_array[j];
        auto &dst_lod_tensor = (*dst_lod_tensor_array)[j];

        dst_lod_tensor.Resize(src_lod_tensor.dims());
        dst_lod_tensor.set_lod(src_lod_tensor.lod());

        auto data_type = src_lod_tensor_array[j].type();
        dst_lod_tensor.mutable_data(dev_place, data_type);

        math::set_constant(dev_ctx, &dst_lod_tensor, value);
      }
    }
  }
};

class FillConstantAsLodTensorArrayOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensorArray)");
    AddAttr<float>("value", "(float, default 0) The value to be filled")
        .SetDefault(0.0f);
    AddAttr<bool>("force_cpu",
                  "(bool, default false) Force fill output variable to cpu "
                  "memory. Otherwise, fill output variable to the running "
                  "device")
        .SetDefault(false);
    AddOutput("Out",
              "(LoDTensorArray) Tensor of specified shape will be filled "
              "with the specified value");
    AddComment(R"DOC(
FillConstantAsLodTensorArrayBatchSizeLike Operator.

Fill up a variable with specified constant value.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fill_constant_as_lodtensorarray,
                  ops::FillConstantAsLodTensorArrayOp,
                  ops::FillConstantAsLodTensorArrayInferShape,
                  ops::FillConstantAsLodTensorArrayOpMaker,
                  paddle::framework::EmptyGradOpMaker);
