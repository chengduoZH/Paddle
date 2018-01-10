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
// #include "paddle/operators/detail/buffer.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {

class UnStageOp : public framework::OperatorBase {
 public:
  UnStageOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::Place &place) const override {
    //    detail::Buffer *buffer;
    //    paddle::operators::detail::GetBuffer(place, buffer_capacity,
    //                                         buffer_bytes_limit, buffer);
    //
    //    detail::BufferElement *buffer_element;
    //    buffer->Get(buffer_element);
    //    //
    //    auto output_var_names = Outputs("Out");
    //    // keep mind the order
    //    for (auto var_name : output_var_names) {
    //      auto *output_var = scope.FindVar(var_name);
    //      output_var->GetMutable<detail::MetaType>();
    //      output_var->Get().SharedMory(buffer_element[0]);
    //    }
  }
};

class UnStageOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  UnStageOpInfoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    //    AddOutput("Out", "The output of fetch op");
    //    AddComment(R"DOC(
    //     UnStage Operator.
    //
    //     It should not be configured by users directly.
    //
    //    )DOC");
  }

  //  framework::OpKernelType GetExpectedKernelType(
  //      const framework::ExecutionContext &ctx) const override {
  //    return framework::OpKernelType(
  //        paddle::framework::DataType::FP32,  // should be changed
  //        paddle::platform::CUDAPlace);
  //  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(unstage, paddle::operators::UnStageOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::UnStageOpInfoMaker);
