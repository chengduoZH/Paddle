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

#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
#include "paddle/operators/detail/buffer.h"

namespace paddle {
namespace operators {

using GetBuffer = paddle::operators::detail::GetBuffer;
using BufferElement = paddle::operators::detail::BufferElement;

class StageOp : public framework::OperatorBase {
 public:
  StageOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::Place &place) const override {
    auto input_var_names = Inputs("Input");
    auto buffer_capacity = Attr<int>("buffer_capacity");
    auto buffer_bytes_limit = Attr<int>("buffer_bytes_limit");

    BufferElement buffer_element;
    for (auto var_name : input_var_names) {
      // the input Vars maybe nullptr in the end of one pass.

      auto *input_var = scope.FindVar(var_name);
      PADDLE_ENFORCE(input_var != nullptr,
                     "Cannot find feed_var in scope, feed_var_name is %s",
                     feed_var_name);
      buffer_element.push_back(input_var);

      detail::Buffer *buffer =
          GetBuffer(place, buffer_capacity, buffer_bytes_limit);

      // if the requirement of overlapping data transfer and kernel operation is
      // true, we should copy data to pinned memory.
      // and then copy the pinned memory to cuda memory in another stream.

      buffer->Put(buffer_element);
    }
  }
};

class StageOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  StageOpInfoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Input", "The input of feed op");
    AddAttr<int>("buffer_capacity", "(int) The column of feed");
    AddAttr<int>("buffer_bytes_limit", "(int) The column of feed");
    //      AddAttr<std::vector<int>>("dtypes", "(int) The column of feed");
    AddComment(R"DOC(
     StageOp Operator.

     According to `buffer_capacity` and `buffer_bytes_limit` Get buffer
  )DOC");
  }

  //    framework::OpKernelType GetExpectedKernelType(
  //        const framework::ExecutionContext &ctx) const override {
  //      return framework::OpKernelType(
  //          paddle::framework::DataType::FP32,  // should be changed
  //          paddle::platform::CUDAPlace);
  //    }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(stage, paddle::operators::StageOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::StageOpInfoMaker);
