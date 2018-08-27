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

#include "paddle/fluid/operators/math/compound_functors.h"
#include <string>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace operators {
namespace math {

class AddAndScaleFunctor {
 public:
  //  template <typename DeviceContext, typename T>
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor &in_x, const framework::Tensor &in_y,
                  std::vector<framework::Tensor *> *outputs) {
    // Z = Binary(X, Unary(Y))
    //    T scale = static_cast<T>(ctx.Attr<float>("scale"));
    //    RunBinaryCompoundFunctor<DeviceContext, T, math::AddFunctor<T>,
    //      math::ScaleFunctor<T>>(
    //      ctx, math::AddFunctor<T>(), math::ScaleFunctor<T>(scale), in_x,
    //      in_y,
    //      outputs);
  }
};
}  // namespace math
}  // namespace operators
}  // namespace paddle

namespace math = paddle::operators::math;

REGISTER_COMPOUNDFUNCTOR(elementwise_add_and_scale, math::AddAndScaleFunctor);
// USE_COMPOUNDFUNCTOR(elementwise_add_and_scale);
