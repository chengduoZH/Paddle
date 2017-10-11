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
#include "paddle/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedPoolingDescriptor = platform::ScopedPoolingDescriptor;
using DataLayout = platform::DataLayout;
using PoolingMode = platform::PoolingMode;

// NOTE: copy from conv_cudnn
std::vector<int> Dims2Vector(const framework::DDim &dims) {
  std::vector<int> ret;
  for (int i = 0; i < dims.size(); i++) {
    ret.push_back(dims[i]);
  }
  return ret;
}

template <typename T>
class PoolCudnnOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");

    const Tensor *input = ctx.Input<Tensor>("X");
    Tensor *output = ctx.Output<Tensor>("Out");

    std::string pooling_type = ctx.Attr<std::string>("poolingType");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    if (ctx.Attr<bool>("globalPooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        ksize[i] = static_cast<int>(input->dims()[i + 2]);
      }
    }

    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>(ctx.GetPlace());

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedPoolingDescriptor pool_desc;

    // ------------------- cudnn pool algorithm ---------------------
    auto handle = ctx.cuda_device_context().cudnn_handle();
    T alpha = 1.0f, beta = 1.0f;

    /*
    PoolingMode pooling_mode;
    DataLayout layout;
    // In the infershape phase, pooling_type has been limited in {"max" or
    // "avg"}
    if (pooling_type == "max") {
      pooling_mode = PoolingMode::kMaximum;
    } else {
      pooling_mode = PoolingMode::kAverage;
    }
    if (ksize.size() == 2) {
      layout = DataLayout::kNCHW;
    } else {
      // TODO(chengduo)
    }
    */
    cudnnTensorDescriptor_t cudnn_input_desc =
        input_desc.descriptor<T>(DataLayout::kNCHW, Dims2Vector(input->dims()));
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        DataLayout::kNCHW, Dims2Vector(output->dims()));

    cudnnPoolingDescriptor_t cudnn_pool_desc =
        pool_desc.descriptor(PoolingMode::kMaximum, ksize, paddings, strides);

    PADDLE_ENFORCE(platform::dynload::cudnnPoolingForward(
        handle, cudnn_pool_desc, &alpha, cudnn_input_desc, input_data, &beta,
        cudnn_output_desc, output_data));

    switch (ksize.size()) {
      case 2: {
      } break;
      case 3: {
      } break;
    }
  }
};

template <typename T>
class PoolCudnnGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");

    const Tensor *input = ctx.Input<Tensor>("X");
    const Tensor *output = ctx.Input<Tensor>("Out");
    const Tensor *output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    std::string pooling_type = ctx.Attr<std::string>("poolingType");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    if (ctx.Attr<bool>("globalPooling")) {
      for (size_t i = 0; i < ksize.size(); ++i)
        ksize[i] = static_cast<int>(input->dims()[i + 2]);
    }

    const T *input_data = input->data<T>();
    const T *output_data = output->data<T>();
    const T *output_grad_data = output_grad->data<T>();
    T *input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedTensorDescriptor output_grad_desc;
    ScopedTensorDescriptor input_grad_desc;
    ScopedPoolingDescriptor pool_desc;

    // ------------------- cudnn backward algorithm ---------------------
    auto handle = ctx.cuda_device_context().cudnn_handle();
    T alpha = 1.0f, beta = 1.0f;

    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      auto temp = framework::EigenVector<T>::Flatten(*input_grad);
      temp.device(ctx.GetEigenDevice<paddle::platform::GPUPlace>()) =
          temp.constant(static_cast<T>(0));

      /*
      PoolingMode pooling_mode;
      DataLayout layout;
      // In the infershape phase, pooling_type has been limited in {"max" or
      // "avg"}
      if (pooling_type == "max") {
        pooling_mode = PoolingMode::kMaximum;
      } else {
        pooling_mode = PoolingMode::kAverage;
      }
      if (ksize.size() == 2) {
        layout = DataLayout::kNCHW;
      } else {
        // TODO(chengduo)
      }
      */
      cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
          DataLayout::kNCHW, Dims2Vector(input->dims()));
      cudnnTensorDescriptor_t cudnn_output_desc = input_desc.descriptor<T>(
          DataLayout::kNCHW, Dims2Vector(output->dims()));
      cudnnTensorDescriptor_t cudnn_input_grad_desc =
          input_grad_desc.descriptor<T>(DataLayout::kNCHW,
                                        Dims2Vector(input_grad->dims()));
      cudnnTensorDescriptor_t cudnn_output_grad_desc =
          output_grad_desc.descriptor<T>(DataLayout::kNCHW,
                                         Dims2Vector(output_grad->dims()));

      cudnnPoolingDescriptor_t cudnn_pool_desc =
          pool_desc.descriptor(PoolingMode::kMaximum, ksize, paddings, strides);

      PADDLE_ENFORCE(platform::dynload::cudnnPoolingBackward(
          handle, cudnn_pool_desc, &alpha, cudnn_output_desc, output_data,
          cudnn_output_grad_desc, output_grad_data, cudnn_input_desc,
          input_data, &beta, cudnn_input_grad_desc, input_grad_data));

      switch (ksize.size()) {
        case 2: {
        } break;
        case 3: {
        } break;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_GPU_KERNEL(pool2d_cudnn, ops::PoolCudnnOpKernel<float>);
REGISTER_OP_GPU_KERNEL(pool2d_cudnn_grad, ops::PoolCudnnGradOpKernel<float>);

REGISTER_OP_GPU_KERNEL(pool3d_cudnn, ops::PoolCudnnOpKernel<float>);
REGISTER_OP_GPU_KERNEL(pool3d_cudnn_grad, ops::PoolCudnnGradOpKernel<float>);
