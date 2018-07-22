/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise_op_function.h"
#include "paddle/fluid/operators/math/functors.h"

namespace math = paddle::operators::math;

namespace paddle {
namespace operators {

class FusedOperatorsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override;
};

class FusedOperatorsMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class FusedOperatorsOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override;
};

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class FusedOperatorsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *in_x = ctx.Input<Tensor>("X");
    const Tensor *in_y = ctx.Input<Tensor>("Y");
    Tensor *output = ctx.Output<Tensor>("Out");

    int axis = ctx.Attr<int>("axis");
    std::vector<std::string> functors =
        ctx.Attr<std::vector<std::string>>("functor_list");

    int func_mode = GetFuncitonMode(functors);

    if (func_mode == 2) {
      //      auto unary_fun_str = functors[1];
      //      size_t pos = unary_fun_str.find(",");
      //      std::string scale_str = unary_fun_str.substr(pos,
      //      unary_fun_str.size());

      T scale = 0.1;
      using BinaryCompoundFunctor =
          math::BinaryCompoundFunctor<T, math::AddFunctor<T>,
                                      math::ScaleFunctor<T>>;

      ElementwiseComputeEx<BinaryCompoundFunctor, DeviceContext, T>(
          ctx, in_x, in_y, axis,
          BinaryCompoundFunctor(math::AddFunctor<T>(),
                                math::ScaleFunctor<T>(scale)),
          output);

    } else {
      T scale = 0.1;
      using UnaryCompoundFunctor =
          math::UnaryCompoundFunctor<T, math::ScaleFunctor<T>,
                                     math::AddFunctor<T>>;

      ElementwiseComputeEx<UnaryCompoundFunctor, DeviceContext, T>(
          ctx, in_x, in_y, axis,
          UnaryCompoundFunctor(math::ScaleFunctor<T>(scale),
                               math::AddFunctor<T>()),
          output);
    }
  }

  int GetFuncitonMode(const std::vector<std::string> &functors) const {
    std::unordered_set<std::string> unary_fun = {"scale", "relu"};
    std::unordered_set<std::string> binary_fun = {"add", "sub"};
    std::string unary_fun_str;
    int func_mode = 2;
    if (binary_fun.count(functors[0])) {
      unary_fun_str = functors[1];
    } else if (binary_fun.count(functors[1])) {
      unary_fun_str = functors[0];
      func_mode = 1;
    } else {
      PADDLE_THROW("functor list is invalid.");
    }
    size_t pos = unary_fun_str.find(",");
    PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str.substr(0, pos)), 1);
    return func_mode;
  }

  // private:
  //  enum class FuncitonMode { UnaryCompound, BinaryCompound };
};

template <typename DeviceContext, typename T>
class FusedOperatorsGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *in_x = ctx.Input<Tensor>("X");
    const Tensor *in_y = ctx.Input<Tensor>("Y");
    const Tensor *in_out = ctx.Input<Tensor>("Out");
    const Tensor *in_out_grad =
        ctx.Input<Tensor>(framework::GradVarName("Out"));

    Tensor *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor *y_grad = ctx.Output<Tensor>(framework::GradVarName("Y"));

    int axis = ctx.Attr<int>("axis");
    std::vector<std::string> functors =
        ctx.Attr<std::vector<std::string>>("functor_list");

    PADDLE_ENFORCE_EQ(functors.size(), 2);

    int64_t numel = in_x->numel();
    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(),
        static_cast<size_t>(numel));

    int func_mode = GetFuncitonMode(functors);  // TODO(zcd): get function mode

    if (func_mode == 1) {
      T scale = 0.1;
      using UnaryCompoundDxFunctor =
          math::UnaryCompoundGradDxFunctor<T, math::ScaleGradFunctor<T>,
                                           math::AddFunctor<T>,
                                           math::AddGradFunctor<T>>;
      using UnaryCompoundDyFunctor =
          math::UnaryCompoundGradDyFunctor<T, math::ScaleGradFunctor<T>,
                                           math::AddFunctor<T>,
                                           math::AddGradFunctor<T>>;

      ElemwiseGradCompute<DeviceContext, T, UnaryCompoundDxFunctor,
                          UnaryCompoundDyFunctor>(
          ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
          UnaryCompoundDxFunctor(math::ScaleGradFunctor<T>(scale),
                                 math::AddFunctor<T>(),
                                 math::AddGradFunctor<T>()),
          UnaryCompoundDyFunctor(math::ScaleGradFunctor<T>(scale),
                                 math::AddFunctor<T>(),
                                 math::AddGradFunctor<T>()));

    } else {
      T scale = 0.1;
      using BinaryCompoundDxFunctor =
          math::BinaryCompoundGradDxFunctor<T, math::AddGradFunctor<T>,
                                            math::ScaleFunctor<T>>;
      using BinaryCompoundDyFunctor =
          math::BinaryCompoundGradDyFunctor<T, math::AddGradFunctor<T>,
                                            math::ScaleFunctor<T>,
                                            math::ScaleGradFunctor<T>>;

      ElemwiseGradCompute<DeviceContext, T, BinaryCompoundDxFunctor,
                          BinaryCompoundDyFunctor>(
          ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
          BinaryCompoundDxFunctor(math::AddGradFunctor<T>(),
                                  math::ScaleFunctor<T>(scale)),
          BinaryCompoundDyFunctor(math::AddGradFunctor<T>(),
                                  math::ScaleFunctor<T>(scale),
                                  math::ScaleGradFunctor<T>(scale)));
    }
  }

  int GetFuncitonMode(const std::vector<std::string> &functors) const {
    std::unordered_set<std::string> unary_fun = {"scale", "relu"};
    std::unordered_set<std::string> binary_fun = {"add", "sub"};
    std::string unary_fun_str;
    int func_mode = 2;
    if (binary_fun.count(functors[0])) {
      unary_fun_str = functors[1];
    } else if (binary_fun.count(functors[1])) {
      unary_fun_str = functors[0];
      func_mode = 1;
    } else {
      PADDLE_THROW("functor list is invalid.");
    }
    size_t pos = unary_fun_str.find(",");
    PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str.substr(0, pos)), 1);
    return func_mode;
  }

  // private:
  //  enum class FuncitonMode { UnaryCompound, BinaryCompound };
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_operators, ops::FusedOperatorsOp,
                  ops::FusedOperatorsMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(fused_operators_grad, ops::FusedOperatorsOpGrad);
