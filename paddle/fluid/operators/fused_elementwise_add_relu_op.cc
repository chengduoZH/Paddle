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

#include "paddle/fluid/operators/fused_elementwise_add_relu_op.h"
#include "paddle/fluid/operators/elementwise_op.h"
namespace ops = paddle::operators;
REGISTER_ELEMWISE_OP(fused_elementwise_add_relu, "AddRelu",
                     "Out = max(X + Y, 0)");

REGISTER_OP_CPU_KERNEL(
    fused_elementwise_add_relu,
    ops::FusedElementwiseAddReluKernel<paddle::platform::CPUDeviceContext,
                                       float>,
    ops::FusedElementwiseAddReluKernel<paddle::platform::CPUDeviceContext,
                                       double>,
    ops::FusedElementwiseAddReluKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FusedElementwiseAddReluKernel<paddle::platform::CPUDeviceContext,
                                       int64_t>);
REGISTER_OP_CPU_KERNEL(
    fused_elementwise_add_relu_grad,
    ops::FusedElementwiseAddReluGradKernel<paddle::platform::CPUDeviceContext,
                                           float>,
    ops::FusedElementwiseAddReluGradKernel<paddle::platform::CPUDeviceContext,
                                           double>,
    ops::FusedElementwiseAddReluGradKernel<paddle::platform::CPUDeviceContext,
                                           int>,
    ops::FusedElementwiseAddReluGradKernel<paddle::platform::CPUDeviceContext,
                                           int64_t>);
