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

#include "paddle/operators/pool_cudnn_op.h"
#include "paddle/memory/memory.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedPoolingDescriptor = platform::ScopedPoolingDescriptor;
using DataLayout = platform::DataLayout;
using CUDADeviceContext = platform::CUDADeviceContext;

template <typename T>
class PoolCudnnOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override{

  };
};

template <typename T>
class PoolCudnnGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override{

  };
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_GPU_KERNEL(
    pool2d_cudnn, ops::PoolCudnnOpKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(
    pool2d_cudnn_grad,
    ops::PoolCudnnGradOpKernel<paddle::platform::GPUPlace, float>);

REGISTER_OP_GPU_KERNEL(
    pool3d_cudnn, ops::PoolCudnnOpKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(
    pool3d_cudnn_grad,
    ops::PoolCudnnGradOpKernel<paddle::platform::GPUPlace, float>);
