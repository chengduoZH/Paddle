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

#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/elementwise_op_function.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

// AddFunctor
template <typename T>
struct AddFunctor {
  // out = x + y;
  inline HOSTDEVICE T operator()(T x, T y) { return x + y; }
};

template <typename T>
struct AddGradFunctor {
  inline HOSTDEVICE T operator()(T x, T y) { return 1; }

  inline HOSTDEVICE T operator()(T x, T y, T out) const { return 1; }
};

template <typename T>
struct ScaleFunctor {
  explicit ScaleFunctor(const T coeff) : coeff_(coeff) {}

  inline HOSTDEVICE T operator()(T ele) { return ele * coeff_; }

 private:
  T coeff_;
};

template <typename T>
struct ScaleGradFunctor {
  explicit ScaleGradFunctor(T coeff) : coeff_(coeff) {}

  inline HOSTDEVICE T operator()(T x) { return coeff_; }

  inline HOSTDEVICE T operator()(T x, T out) { return coeff_; }

 private:
  T coeff_;
};

template <typename T>
struct ReluFunctor {
  inline HOSTDEVICE T operator()(T x) { return x * (x > 0); }
};

template <typename T>
struct ReluGradFunctor {
  inline HOSTDEVICE T operator()(T x) { return x > 0 ? 1 : 0; }

  inline HOSTDEVICE T operator()(T x, T out) { return x > 0 ? 1 : 0; }
};

// For example: z = Binary(X, Unary(Y))
template <typename T, typename BinaryFun, typename UnaryFun>
struct BinaryCompoundFunctor {
  BinaryCompoundFunctor(const BinaryFun &binary_fun, const UnaryFun &unary_fun)
      : binary_fun_(binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y) {
    return binary_fun_(x, unary_fun_(y));
  }

 private:
  BinaryFun binary_fun_;
  UnaryFun unary_fun_;
};

template <typename T, typename DBinaryFun, typename UnaryFun,
          bool Recomputation = true>
struct BinaryCompoundGradDxFunctor {
  BinaryCompoundGradDxFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun)
      : d_binary_fun_(d_binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    // FIXME(zcd): DBinaryFun have to method to get the dx,
    // one is to use the 'out', and the other is not to use it.
    // the former method will save the time of recomputation the
    // 'out', but it must occupy the memory to store the 'out'.
    // While the later method can avoid occupying this memory,
    // but it must recompute the 'out'.

    if (Recomputation) {
      return dout * d_binary_fun_(x, unary_fun_(y));
    } else {
      return dout * d_binary_fun_(x, unary_fun_(y), out);
    }
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
};

template <typename T, typename DBinaryFun, typename UnaryFun,
          typename DUnaryFun, bool Recomputation = true>
struct BinaryCompoundGradDyFunctor {
  BinaryCompoundGradDyFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun,
                              const DUnaryFun &d_unary_fun)
      : d_binary_fun_(d_binary_fun),
        unary_fun_(unary_fun),
        d_unary_fun_(d_unary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    if (Recomputation) {
      return dout * d_binary_fun_(unary_fun_(y), x) * d_unary_fun_(y);
    } else {
      return dout * d_binary_fun_(unary_fun_(y), x, out) * d_unary_fun_(y);
    }
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
  DUnaryFun d_unary_fun_;
};

template <typename T, typename UnaryFun, typename BinaryFun>
struct UnaryCompoundFunctor {
  UnaryCompoundFunctor(const UnaryFun &unary_fun, const BinaryFun &binary_fun)
      : unary_fun_(unary_fun), binary_fun_(binary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y) {
    return unary_fun_(binary_fun_(x, y));
  }

 private:
  UnaryFun unary_fun_;
  BinaryFun binary_fun_;
};

template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun, bool Recomputation = true>
struct UnaryCompoundGradDxFunctor {
  UnaryCompoundGradDxFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(binary_fun_(x, y));
    } else {
      base = dout * d_unary_fun_(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_(x, y);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun, bool Recomputation = true>
struct UnaryCompoundGradDyFunctor {
  UnaryCompoundGradDyFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(binary_fun_(x, y));
    } else {
      base = dout * d_unary_fun_(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_(y, x);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

static bool ValidCheck(const std::string &functors) {
  std::unordered_set<std::string> unary_fun = {"scale", "relu"};
  std::unordered_set<std::string> binary_fun = {"elementwise_add"};

  size_t pos = functors.find(",");
  auto func_1 = functors.substr(0, pos);
  auto func_2 = functors.substr(pos + 1, functors.size());
  std::string unary_fun_str;
  if (binary_fun.count(func_1)) {
    unary_fun_str = func_2;
  } else if (binary_fun.count(func_2)) {
    unary_fun_str = func_1;
  } else {
    PADDLE_THROW("%s and %s are not included in fused_list.", func_1, func_2);
  }
  PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str), 1,
                    "%s is not included in fused_list.", unary_fun_str);
  return true;
}

template <typename DeviceContext, typename T, typename BinaryFunctor,
          typename UnaryFunctor>
static void RunBinaryCompoundFunctor(const framework::ExecutionContext &ctx,
                                     const BinaryFunctor &binary_functor,
                                     const UnaryFunctor &unary_functor,
                                     const framework::Tensor *in_x,
                                     const framework::Tensor *in_y,
                                     framework::Tensor *output) {
  int axis = ctx.Attr<int>("axis");

  using BinaryCompoundFunctor =
      math::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>;

  ElementwiseComputeEx<BinaryCompoundFunctor, DeviceContext, T>(
      ctx, in_x, in_y, axis,
      BinaryCompoundFunctor(binary_functor, unary_functor), output);
}

template <typename DeviceContext, typename T, typename UnaryFunctor,
          typename BinaryFunctor>
static void RunUnaryCompoundFunctors(const framework::ExecutionContext &ctx,
                                     const UnaryFunctor &unary_functor,
                                     const BinaryFunctor &binary_functor,
                                     const framework::Tensor *in_x,
                                     const framework::Tensor *in_y,
                                     framework::Tensor *output) {
  int axis = ctx.Attr<int>("axis");

  using UnaryCompoundFunctor =
      math::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>;

  ElementwiseComputeEx<UnaryCompoundFunctor, DeviceContext, T>(
      ctx, in_x, in_y, axis,
      UnaryCompoundFunctor(unary_functor, binary_functor), output);
}

template <typename DeviceContext, typename T, typename BinaryGradFunctor,
          typename UnaryFunctor, typename UnaryGradFunctor,
          bool Recomputation = true>
static void RunBinaryCompoundGradFunctors(
    const framework::ExecutionContext &ctx,
    const BinaryGradFunctor &binary_grad_functor,
    const UnaryFunctor &unary_functor,
    const UnaryGradFunctor &unary_grad_functor, const framework::Tensor *in_x,
    const framework::Tensor *in_y, const framework::Tensor *in_out,
    const framework::Tensor *in_out_grad, framework::Tensor *x_grad,
    framework::Tensor *y_grad) {
  int axis = ctx.Attr<int>("axis");

  using BinaryCompoundDxFunctor =
      math::BinaryCompoundGradDxFunctor<T, BinaryGradFunctor, UnaryFunctor,
                                        Recomputation>;
  using BinaryCompoundDyFunctor =
      math::BinaryCompoundGradDyFunctor<T, BinaryGradFunctor, UnaryFunctor,
                                        UnaryGradFunctor, Recomputation>;

  ElemwiseGradCompute<DeviceContext, T, BinaryCompoundDxFunctor,
                      BinaryCompoundDyFunctor>(
      ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
      BinaryCompoundDxFunctor(binary_grad_functor, unary_functor),
      BinaryCompoundDyFunctor(binary_grad_functor, unary_functor,
                              unary_grad_functor));
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
    const framework::Tensor *in_out_grad, framework::Tensor *x_grad,
    framework::Tensor *y_grad) {
  int axis = ctx.Attr<int>("axis");

  using UnaryCompoundDxFunctor =
      math::UnaryCompoundGradDxFunctor<T, UnaryGradFunctor, BinaryFunctor,
                                       BinaryGradFunctor, Recomputation>;
  using UnaryCompoundDyFunctor =
      math::UnaryCompoundGradDyFunctor<T, UnaryGradFunctor, BinaryFunctor,
                                       BinaryGradFunctor, Recomputation>;

  ElemwiseGradCompute<DeviceContext, T, UnaryCompoundDxFunctor,
                      UnaryCompoundDyFunctor>(
      ctx, *in_x, *in_y, *in_out, *in_out_grad, axis, x_grad, y_grad,
      UnaryCompoundDxFunctor(unary_grad_functor, binary_functor,
                             binary_grad_functor),
      UnaryCompoundDyFunctor(unary_grad_functor, binary_functor,
                             binary_grad_functor));
}

template <typename DeviceContext, typename T>
static void RunFunctors(const framework::ExecutionContext &ctx,
                        const std::string &functors,
                        const framework::Tensor *in_x,
                        const framework::Tensor *in_y,
                        framework::Tensor *output) {
  // TODO(zcd): The following code can be refined.
  if (functors == "elementwise_add,scale") {
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunBinaryCompoundFunctor<DeviceContext, T, math::AddFunctor<T>,
                             math::ScaleFunctor<T>>(
        ctx, math::AddFunctor<T>(), math::ScaleFunctor<T>(scale), in_x, in_y,
        output);
  } else if (functors == "scale,elementwise_add") {
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    RunUnaryCompoundFunctors<DeviceContext, T, math::ScaleFunctor<T>,
                             math::AddFunctor<T>>(
        ctx, math::ScaleFunctor<T>(scale), math::AddFunctor<T>(), in_x, in_y,
        output);
  } else if (functors == "elementwise_add,relu") {
    RunBinaryCompoundFunctor<DeviceContext, T, math::AddFunctor<T>,
                             math::ReluFunctor<T>>(
        ctx, math::AddFunctor<T>(), math::ReluFunctor<T>(), in_x, in_y, output);
  } else if (functors == "relu,elementwise_add") {
    RunUnaryCompoundFunctors<DeviceContext, T, math::ReluFunctor<T>,
                             math::AddFunctor<T>>(
        ctx, math::ReluFunctor<T>(), math::AddFunctor<T>(), in_x, in_y, output);
  } else {
    PADDLE_THROW("%s has not been implemented.", functors);
  }
}

template <typename DeviceContext, typename T>
static void RunGradFunctors(
    const framework::ExecutionContext &ctx, const std::string &functors,
    const framework::Tensor *in_x, const framework::Tensor *in_y,
    const framework::Tensor *in_out, const framework::Tensor *in_out_grad,
    framework::Tensor *x_grad, framework::Tensor *y_grad) {
  bool recomputation = ctx.Attr<bool>("recomputation");
  // TODO(zcd): The following code can be refined.
  if (functors == "elementwise_add_grad,scale_grad") {
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    if (recomputation) {
      RunBinaryCompoundGradFunctors<DeviceContext, T, math::AddGradFunctor<T>,
                                    math::ScaleFunctor<T>,
                                    math::ScaleGradFunctor<T>, true>(
          ctx, math::AddGradFunctor<T>(), math::ScaleFunctor<T>(scale),
          math::ScaleGradFunctor<T>(scale), in_x, in_y, in_out, in_out_grad,
          x_grad, y_grad);
    } else {
      RunBinaryCompoundGradFunctors<DeviceContext, T, math::AddGradFunctor<T>,
                                    math::ScaleFunctor<T>,
                                    math::ScaleGradFunctor<T>, false>(
          ctx, math::AddGradFunctor<T>(), math::ScaleFunctor<T>(scale),
          math::ScaleGradFunctor<T>(scale), in_x, in_y, in_out, in_out_grad,
          x_grad, y_grad);
    }
  } else if (functors == "scale_grad,elementwise_add_grad") {
    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    if (recomputation) {
      RunUnaryCompoundGradFunctors<DeviceContext, T, math::ScaleGradFunctor<T>,
                                   math::AddFunctor<T>, math::AddGradFunctor<T>,
                                   true>(ctx, math::ScaleGradFunctor<T>(scale),
                                         math::AddFunctor<T>(),
                                         math::AddGradFunctor<T>(), in_x, in_y,
                                         in_out, in_out_grad, x_grad, y_grad);
    } else {
      RunUnaryCompoundGradFunctors<DeviceContext, T, math::ScaleGradFunctor<T>,
                                   math::AddFunctor<T>, math::AddGradFunctor<T>,
                                   false>(ctx, math::ScaleGradFunctor<T>(scale),
                                          math::AddFunctor<T>(),
                                          math::AddGradFunctor<T>(), in_x, in_y,
                                          in_out, in_out_grad, x_grad, y_grad);
    }
  } else if (functors == "elementwise_add_grad,relu_grad") {
    if (recomputation) {
      RunBinaryCompoundGradFunctors<DeviceContext, T, math::AddGradFunctor<T>,
                                    math::ReluFunctor<T>,
                                    math::ReluGradFunctor<T>, true>(
          ctx, math::AddGradFunctor<T>(), math::ReluFunctor<T>(),
          math::ReluGradFunctor<T>(), in_x, in_y, in_out, in_out_grad, x_grad,
          y_grad);
    } else {
      RunBinaryCompoundGradFunctors<DeviceContext, T, math::AddGradFunctor<T>,
                                    math::ReluFunctor<T>,
                                    math::ReluGradFunctor<T>, false>(
          ctx, math::AddGradFunctor<T>(), math::ReluFunctor<T>(),
          math::ReluGradFunctor<T>(), in_x, in_y, in_out, in_out_grad, x_grad,
          y_grad);
    }
  } else if (functors == "relu_grad,elementwise_add_grad") {
    if (recomputation) {
      RunUnaryCompoundGradFunctors<DeviceContext, T, math::ReluGradFunctor<T>,
                                   math::AddFunctor<T>, math::AddGradFunctor<T>,
                                   true>(ctx, math::ReluGradFunctor<T>(),
                                         math::AddFunctor<T>(),
                                         math::AddGradFunctor<T>(), in_x, in_y,
                                         in_out, in_out_grad, x_grad, y_grad);
    } else {
      RunUnaryCompoundGradFunctors<DeviceContext, T, math::ReluGradFunctor<T>,
                                   math::AddFunctor<T>, math::AddGradFunctor<T>,
                                   false>(ctx, math::ReluGradFunctor<T>(),
                                          math::AddFunctor<T>(),
                                          math::AddGradFunctor<T>(), in_x, in_y,
                                          in_out, in_out_grad, x_grad, y_grad);
    }
  } else {
    PADDLE_THROW("%s has not been implemented.", functors);
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
