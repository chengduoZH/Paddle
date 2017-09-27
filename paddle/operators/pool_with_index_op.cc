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

#include "paddle/operators/pool_with_index_op.h"

namespace paddle {
namespace operators {

int OutputSizeMaxPool(int input_size, int filter_size, int padding,
                      int stride) {
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

class MaxPoolWithIndexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "X(Input) of Pooling should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Out(Output) of Pooling should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Mask"),
                   "Out(Output) of Pooling should not be null.");

    auto in_x_dims = ctx->GetInputDim("X");

    std::vector<int> ksize = Attr<std::vector<int>>("ksize");
    std::vector<int> strides = Attr<std::vector<int>>("strides");
    std::vector<int> paddings = Attr<std::vector<int>>("paddings");

    PADDLE_ENFORCE(in_x_dims.size() == 4 || in_x_dims.size() == 5,
                   "Pooling intput should be 4-D or 5-D");

    if (Attr<bool>("globalPooling")) {
      ksize.resize(static_cast<size_t>(in_x_dims.size()) - 2);
      for (size_t i = 0; i < ksize.size(); ++i)
        ksize[i] = static_cast<int>(in_x_dims[i + 2]);
    }

    PADDLE_ENFORCE(in_x_dims.size() - ksize.size() == 2U,
                   "Pooling intput size and pooling size should be consistent");
    PADDLE_ENFORCE(ksize.size() == 2 || ksize.size() == 3,
                   "Pooling size size should be 2 elements. or 3 elements.");
    PADDLE_ENFORCE_EQ(ksize.size(), strides.size(),
                      "strides size and pooling size should be the same.");
    PADDLE_ENFORCE_EQ(ksize.size(), paddings.size(),
                      "paddings size and pooling size should be the same.");

    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      output_shape.push_back(OutputSizeMaxPool(in_x_dims[i + 2], ksize[i],
                                               paddings[i], strides[i]));
    }
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
    ctx->SetOutputDim("Mask", framework::make_ddim(output_shape));
  }
};

class MaxPoolWithIndexOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("X")),
                   "X(Input) of MaxPoolWithIndexOpGrad should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput(framework::GradVarName("X")),
        "X@GRAD(Input@GRAD) of MaxPoolWithIndexOpGrad should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

class MaxPool2dWithIndexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MaxPool2dWithIndexOpMaker(framework::OpProto *proto,
                            framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "The input tensor of pooling operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of image.");
    AddOutput("Out",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCHW.");
    AddOutput("Mask",
              "The Mask tensor of pooling operator."
              "The format of output tensor is also NCHW.");

    AddAttr<std::vector<int>>(
        "ksize", "pooling size(height, width) of pooling operator.");
    AddAttr<bool>(
        "globalPooling",
        "whether to use the globalPooling."
        "int constant equal to false or true"
        "default false"
        "If globalPooling = true, ksize is ignored and need not be specified.")
        .SetDefault(false);
    AddAttr<std::vector<int>>("strides",
                              "strides(height, width) of pooling operator."
                              "default {1,1}")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "paddings(height, width) of pooling operator."
                              "default {0,0}")
        .SetDefault({0, 0});

    AddComment(R"DOC(
The maxPooling2d with index operation calculates the output and the mask based on
the input and ksize, strides, paddings parameters.
)DOC");
  }
};

class MaxPool3dWithIndexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MaxPool3dWithIndexOpMaker(framework::OpProto *proto,
                            framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "The input tensor of pooling operator. "
        "The format of input tensor is NCDHW. Where N is batch size, C is "
        "the number of channels, D, H and W is the depth, height and width of "
        "image.");
    AddOutput("Out",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCDHW.");
    AddOutput("Mask",
              "The Mask tensor of pooling operator."
              "The format of output tensor is also NCDHW.");

    AddAttr<std::vector<int>>(
        "ksize", "pooling size(depth, height, width) of pooling operator.");
    AddAttr<bool>(
        "globalPooling",
        "whether to use the globalPooling."
        "int constant equal to false or true"
        "default false"
        "If globalPooling = true, ksize is ignored and need not be specified.")
        .SetDefault(false);
    AddAttr<std::vector<int>>(
        "strides",
        "strides(depth, height, width) of pooling operator."
        "default {1,1,1}")
        .SetDefault({1, 1, 1});
    AddAttr<std::vector<int>>(
        "paddings",
        "paddings(depth, height, width) of pooling operator."
        "default {0,0,0}")
        .SetDefault({0, 0, 0});
    AddComment(R"DOC(
The maxpooling3d with index operation calculates the output and the mask based on
the input and ksize, strides, paddings parameters.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(maxPool2dWithIndex, ops::MaxPoolWithIndexOp,
            ops::MaxPool2dWithIndexOpMaker, pool2dWithIndex_grad,
            ops::MaxPoolWithIndexOpGrad);

REGISTER_OP_CPU_KERNEL(
    maxPool2dWithIndex,
    ops::MaxPoolWithIndexKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    maxPool2dWithIndex_grad,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CPUPlace, float>)

REGISTER_OP(maxPool3dWithIndex, ops::MaxPoolWithIndexOp,
            ops::MaxPool3dWithIndexOpMaker, maxPool3dWithIndex_grad,
            ops::MaxPoolWithIndexOpGrad);

REGISTER_OP_CPU_KERNEL(
    maxPool3dWithIndex,
    ops::MaxPoolWithIndexKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    maxPool3dWithIndex_grad,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CPUPlace, float>)
