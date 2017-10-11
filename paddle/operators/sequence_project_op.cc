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

#include "paddle/operators/sequence_project_op.h"

namespace paddle {
namespace operators {

class SequenceProjectOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    int context_length = ctx->Attrs().Get<int>("context_length");
    bool padding_trainable = ctx->Attrs().Get<bool>("padding_trainable");
    int context_start = ctx->Attrs().Get<int>("context_start");

    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceProjectOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequenceProjectOp should not be null.");

    if (padding_trainable) {
      PADDLE_ENFORCE(
          ctx->HasOutput("PaddingData"),
          "Output(PaddingData) of SequenceProjectOp should not be null.");
      // TODO(duo): get  PaddingData shape
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

class SequenceProjectOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceProjectOpMaker(framework::OpProto* proto,
                         framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "A float LoDTensor, the variable-length input of SequenceProjectOp");
    AddOutput(
        "Out",
        "A float LoDTensor, the variable-length output of SequenceProjectOp.");

    AddOutput("PaddingData",
              "A float LoDTensor, the padding data of SequenceProjectOp.");

    AddAttr<int>("context_length",
                 "(int, default 3) the stride of SequenceProjectOp.")
        .SetDefault(3)
        .GreaterThan(0);
    AddAttr<bool>("padding_trainable",
                  "(bool, default false) the padding data of SequenceProjectOp "
                  "is trainable or not.")
        .SetDefault(false);
    AddAttr<int>("context_start",
                 "(int, default 0) the xx of SequenceProjectOp.")
        .SetDefault(0);

    AddComment(R"DOC(
    SequenceProjectOp projects features of context_length time-steps of each instance.

    For a mini-batch of 2 variable lengths sentences, containing 3, and 1 time-steps:

    Assumed input (X) is a [4, M, N] float LoDTensor, and X->lod()[0] = [0, 3, 4].
    Besides, for the sake of simplicity, we assume M=1 and N=2.

    X = [[a1, a2,
          b1, b2.
          c1, c2]
         [d1, d2]]

    This is to say that input (X) has 4 words and the dimension of each word
    representation is 2. If we use zero to pad instead of learned weight to pad,
    and the context_lenth is 3, the output (Out) is:

    Out = [0,  0,  a1, a2, b1, b2;
           a1, a2, b1, b2, c1, c2;
           b1, b2, c1, c2, d1, d2;
           c1, c2, d1, d2, 0,  0]

    )DOC");
  }
};

class SequencePoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("X"), "The input X should not be null.");

    bool padding_trainable = ctx->Attrs().Get<bool>("padding_trainable");
    if (padding_trainable) {
      PADDLE_ENFORCE(
          ctx->HasOutput("PaddingData"),
          "Output(PaddingData) of SequenceProjectOp should not be null.");
      // TODO(duo): get  PaddingData shape
    }
    auto x_dims = ctx->GetInputDim("X");

    //    auto og_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    //    PADDLE_ENFORCE_EQ(og_dims.size(), x_dims.size(),
    //                      "The rank of output grad must equal to Input(X).");
    //    for (int64_t i = 1; i < og_dims.size(); ++i) {
    //      PADDLE_ENFORCE_EQ(og_dims[i], x_dims[i], "The dimension mismatch.");
    //    }
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sequence_project, ops::SequenceProjectOp,
            ops::SequenceProjectOpMaker, sequence_project_grad,
            ops::SequencePoolGradOp);

REGISTER_OP_CPU_KERNEL(
    sequence_project,
    ops::SequenceProjectKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_project_grad,
    ops::SequenceProjectGradKernel<paddle::platform::CPUPlace, float>);
