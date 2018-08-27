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
#include <unordered_set>
#include <vector>

#include "paddle/fluid/operators/elementwise_op_function.h"
#include "paddle/fluid/operators/math/compound_functors.h"
#include "paddle/fluid/operators/math/functors.h"

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T, typename BinaryFunctor,
          typename UnaryFunctor>
static void RunBinaryCompoundFunctor(
    const framework::ExecutionContext &ctx, const BinaryFunctor &binary_functor,
    const UnaryFunctor &unary_functor, const framework::Tensor &in_x,
    const framework::Tensor &in_y, std::vector<framework::Tensor *> *outputs) {
  // Z = Binary(X, Unary(Y))
  // intermediate_out = Unary(Y)
  // out = Binary(X, Unary(Y))
  // In this case, the shape of intermediate_out and out are different.
  math::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor> compound_func(
      binary_functor, unary_functor);
  int axis = ctx.Attr<int>("axis");
  if (ctx.Attr<bool>("keep_intermediate_value")) {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T,
        math::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>,
        true /*KeepIntermediateValue*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  } else {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T,
        math::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>,
        false /*KeepIntermediateValue*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  }
}

template <typename DeviceContext, typename T, typename UnaryFunctor,
          typename BinaryFunctor>
static void RunUnaryCompoundFunctors(
    const framework::ExecutionContext &ctx, const UnaryFunctor &unary_functor,
    const BinaryFunctor &binary_functor, const framework::Tensor &in_x,
    const framework::Tensor &in_y, std::vector<framework::Tensor *> *outputs) {
  // Z = Unary(Binary(X, Y))
  // intermediate_out = Binary(X, Y)
  // out = Unary(Binary(X, Y))
  // In this case, the shape of intermediate_out and out are the same.
  int axis = ctx.Attr<int>("axis");

  math::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor> compound_func(
      unary_functor, binary_functor);

  if (ctx.Attr<bool>("keep_intermediate_value")) {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T,
        math::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>,
        true /*KeepIntermediateValue*/,
        true /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  } else {
    FusedElemwiseAndActComputeEx<
        DeviceContext, T,
        math::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>,
        false /*KeepIntermediateValue*/,
        true /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  }
}

template <typename DeviceContext, typename T, typename BinaryGradFunctor,
          typename UnaryFunctor, typename UnaryGradFunctor>
static void RunBinaryCompoundGradFunctors(
    const framework::ExecutionContext &ctx,
    const BinaryGradFunctor &binary_grad_functor,
    const UnaryFunctor &unary_functor,
    const UnaryGradFunctor &unary_grad_functor, const framework::Tensor *in_x,
    const framework::Tensor *in_y, const framework::Tensor *in_out,
    const framework::Tensor *in_intermediate_out,
    const framework::Tensor *in_out_grad, framework::Tensor *x_grad,
    framework::Tensor *y_grad) {
  // Z = Binary(X, Unary(Y))
  int axis = ctx.Attr<int>("axis");

  using BinaryCompoundDxFunctor =
      math::BinaryCompoundGradDxFunctor<T, BinaryGradFunctor, UnaryFunctor>;
  using BinaryCompoundDyFunctor =
      math::BinaryCompoundGradDyFunctor<T, BinaryGradFunctor, UnaryFunctor,
                                        UnaryGradFunctor>;

  if (in_intermediate_out) {
    FusedElemwiseAndActGradComputeEx<
        DeviceContext, T, BinaryCompoundDxFunctor, BinaryCompoundDyFunctor,
        true /*UseIntermediateOut*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, axis, x_grad,
        y_grad, BinaryCompoundDxFunctor(binary_grad_functor, unary_functor),
        BinaryCompoundDyFunctor(binary_grad_functor, unary_functor,
                                unary_grad_functor));
  } else {
    FusedElemwiseAndActGradComputeEx<
        DeviceContext, T, BinaryCompoundDxFunctor, BinaryCompoundDyFunctor,
        false /*UseIntermediateOut*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, axis, x_grad,
        y_grad, BinaryCompoundDxFunctor(binary_grad_functor, unary_functor),
        BinaryCompoundDyFunctor(binary_grad_functor, unary_functor,
                                unary_grad_functor));
  }
}

template <typename DeviceContext, typename T, typename UnaryGradFunctor,
          typename BinaryFunctor, typename BinaryGradFunctor,
          bool Recomputation = true>
static void RunUnaryCompoundGradFunctors(
    const framework::ExecutionContext &ctx,
    const UnaryGradFunctor &unary_grad_functor,
    const BinaryFunctor &binary_functor,
    const BinaryGradFunctor &binary_grad_functor, const framework::Tensor *in_x,
    const framework::Tensor *in_y, const framework::Tensor *in_out,
    const framework::Tensor *in_intermediate_out,
    const framework::Tensor *in_out_grad, framework::Tensor *x_grad,
    framework::Tensor *y_grad) {
  // Z = Unary(Binary(X, Y))
  int axis = ctx.Attr<int>("axis");

  using UnaryCompoundDxFunctor =
      math::UnaryCompoundGradDxFunctor<T, UnaryGradFunctor, BinaryFunctor,
                                       BinaryGradFunctor, Recomputation>;
  using UnaryCompoundDyFunctor =
      math::UnaryCompoundGradDyFunctor<T, UnaryGradFunctor, BinaryFunctor,
                                       BinaryGradFunctor, Recomputation>;

  if (in_intermediate_out) {
    FusedElemwiseAndActGradComputeEx<
        DeviceContext, T, UnaryCompoundDxFunctor, UnaryCompoundDyFunctor,
        true /*UseIntermediateOut*/, true /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, axis, x_grad,
        y_grad, UnaryCompoundDxFunctor(unary_grad_functor, binary_functor,
                                       binary_grad_functor),
        UnaryCompoundDyFunctor(unary_grad_functor, binary_functor,
                               binary_grad_functor));
  } else {
    FusedElemwiseAndActGradComputeEx<DeviceContext, T, UnaryCompoundDxFunctor,
                                     UnaryCompoundDyFunctor,
                                     false /*UseIntermediateOut*/,
                                     true /*SameShapeOfIntermediateOutAndOut*/>(
        ctx, in_x, in_y, in_out, in_intermediate_out, in_out_grad, axis, x_grad,
        y_grad, UnaryCompoundDxFunctor(unary_grad_functor, binary_functor,
                                       binary_grad_functor),
        UnaryCompoundDyFunctor(unary_grad_functor, binary_functor,
                               binary_grad_functor));
  }
}

class AddAndScaleFunctor {
 public:
  template <typename DeviceContext, typename T>
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor &in_x, const framework::Tensor &in_y,
                  std::vector<framework::Tensor *> *outputs) {
    // Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunBinaryCompoundFunctor<DeviceContext, T, math::AddFunctor<T>,
                             math::ScaleFunctor<T>>(
        ctx, math::AddFunctor<T>(), math::ScaleFunctor<T>(scale), in_x, in_y,
        outputs);
  }
};

class ScaleAndAddFunctor {
 public:
  template <typename DeviceContext, typename T>
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor &in_x, const framework::Tensor &in_y,
                  std::vector<framework::Tensor *> *outputs) {
    // Z = Unary(Binary(X, Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunUnaryCompoundFunctors<DeviceContext, T, math::ScaleFunctor<T>,
                             math::AddFunctor<T>>(
        ctx, math::ScaleFunctor<T>(scale), math::AddFunctor<T>(), in_x, in_y,
        outputs);
  }
};

class AddAndReluFunctor {
 public:
  template <typename DeviceContext, typename T>
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor &in_x, const framework::Tensor &in_y,
                  std::vector<framework::Tensor *> *outputs) {
    // Z = Binary(X, Unary(Y))
    RunBinaryCompoundFunctor<DeviceContext, T, math::AddFunctor<T>,
                             math::ReluFunctor<T>>(ctx, math::AddFunctor<T>(),
                                                   math::ReluFunctor<T>(), in_x,
                                                   in_y, outputs);
  }
};

class ReluAndAddFunctor {
 public:
  template <typename DeviceContext, typename T>
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor &in_x, const framework::Tensor &in_y,
                  std::vector<framework::Tensor *> *outputs) {
    // Z = Unary(Binary(X, Y))
    RunUnaryCompoundFunctors<DeviceContext, T, math::ReluFunctor<T>,
                             math::AddFunctor<T>>(ctx, math::ReluFunctor<T>(),
                                                  math::AddFunctor<T>(), in_x,
                                                  in_y, outputs);
  }
};

class MulAndScaleFunctor {
 public:
  template <typename DeviceContext, typename T>
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor &in_x, const framework::Tensor &in_y,
                  std::vector<framework::Tensor *> *outputs) {
    // Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunBinaryCompoundFunctor<DeviceContext, T, math::MulFunctor<T>,
                             math::ScaleFunctor<T>>(
        ctx, math::MulFunctor<T>(), math::ScaleFunctor<T>(scale), in_x, in_y,
        outputs);
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle

namespace math = paddle::operators::math;

REGISTER_COMPOUNDFUNCTOR(elementwise_add_and_scale, math::AddAndScaleFunctor);
REGISTER_COMPOUNDFUNCTOR(scale_and_elementwise_add, math::ScaleAndAddFunctor);
REGISTER_COMPOUNDFUNCTOR(elementwise_add_and_relu, math::AddAndReluFunctor);
REGISTER_COMPOUNDFUNCTOR(relu_and_elementwise_add, math::ReluAndAddFunctor);
REGISTER_COMPOUNDFUNCTOR(elementwise_mul_and_scale, math::MulAndScaleFunctor);
// USE_COMPOUNDFUNCTOR(elementwise_add_and_scale);
