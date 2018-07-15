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
#include "paddle/fluid/operators/fused_elementwise_add_relu_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    fused_elementwise_add_relu,
    ops::FusedElementwiseAddReluKernel<plat::CUDADeviceContext, float>,
    ops::FusedElementwiseAddReluKernel<plat::CUDADeviceContext, double>,
    ops::FusedElementwiseAddReluKernel<plat::CUDADeviceContext, int>,
    ops::FusedElementwiseAddReluKernel<plat::CUDADeviceContext, int64_t>,
    ops::FusedElementwiseAddReluKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    fused_elementwise_add_relu_grad,
    ops::FusedElementwiseAddReluGradKernel<plat::CUDADeviceContext, float>,
    ops::FusedElementwiseAddReluGradKernel<plat::CUDADeviceContext, double>,
    ops::FusedElementwiseAddReluGradKernel<plat::CUDADeviceContext, int>,
    ops::FusedElementwiseAddReluGradKernel<plat::CUDADeviceContext, int64_t>);
