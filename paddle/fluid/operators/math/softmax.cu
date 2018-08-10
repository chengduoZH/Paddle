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

#define EIGEN_USE_GPU

#include <vector>

#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/math/softmax_impl.h"

namespace paddle {
namespace operators {
namespace math {

template class SoftmaxFunctor<platform::CUDADeviceContext, platform::float16>;
template class SoftmaxFunctor<platform::CUDADeviceContext, float>;
template class SoftmaxFunctor<platform::CUDADeviceContext, double>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, float>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
