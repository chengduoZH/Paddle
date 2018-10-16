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

#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename DeviceContext, typename T>
class CUDNNDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *input = context.Input<Tensor>("X");
    auto *output = context.Output<Tensor>("Out");

    if (!context.Attr<bool>("is_test")) {
      auto *mask = context.Output<Tensor>("Mask");

      float dropout_prob = context.Attr<float>("dropout_prob");
      int random_seed = context.Attr<int>("seed");

      auto &dev_ctx =
          context.template device_context<platform::CUDADeviceContext>();
      auto cudnn_handle = dev_ctx.cudnn_handle();

      ScopedTensorDescriptor input_desc;
      DataLayout layout = DataLayout::kNCHW;
      cudnnTensorDescriptor_t data_desc = input_desc.descriptor<T>(
          layout, framework::vectorize2int(input->dims()));

      size_t states_size_in_bytes_;
      CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetStatesSize(
          cudnn_handle, &states_size_in_bytes_));

      // TODO(zcd): Node that states is local var.
      Tensor states;
      states.Resize(framework::make_ddim({states_size_in_bytes_}));
      uint8_t *states_data = states->mutable_data<uint8_t>();

      ScopedDropoutDescriptor dropout_desc;
      cudnnDropoutDescriptor_t cudnn_dropout_desc =
          dropout_desc.descriptor<T>(cudnn_handle, dropout_prob, states_data,
                                     states_size_in_bytes_, random_seed);

      size_t reserve_space_size_in_bytes_;
      CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetReserveSpaceSize(
          data_desc, &reserve_space_size_in_bytes_));
      mask->Resize(framework::make_ddim({reserve_space_size_in_bytes_}));

      CUDNN_ENFORCE(platform::dynload::cudnnDropoutForward(
          cudnn_handle, dropout_desc, data_desc, input->data<T>(), data_desc,
          output->mutable_data<T>(context.GetPlace()),
          mask->mutable_data<uint8_t>(context.GetPlace()),
          reserve_space_size_in_bytes_));

    } else {
      auto X = EigenMatrix<T>::Reshape(*x, 1);
      auto Y = EigenMatrix<T>::Reshape(*y, 1);
      auto &place =
          *context.template device_context<DeviceContext>().eigen_device();
      Y.device(place) = X * (1.0f - dropout_prob);
    }
  }
};

template <typename DeviceContext, typename T>
class CUDNNDropoutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    PADDLE_ENFORCE(!context.Attr<bool>("is_test"),
                   "GradOp is only callable when is_test is false");

    auto *grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto *grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto *mask = context.Input<Tensor>("Mask");

    grad_x->mutable_data<T>(context.GetPlace());
    void *mask_data = mask->mutable_data<void>(context.GetPlace());

    float dropout_prob = context.Attr<float>("dropout_prob");
    int random_seed = context.Attr<int>("seed");

    auto &dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    auto cudnn_handle = dev_ctx.cudnn_handle();

    ScopedTensorDescriptor input_desc;
    DataLayout layout = DataLayout::kNCHW;
    cudnnTensorDescriptor_t data_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(grad_y->dims()));

    size_t states_size_in_bytes_;
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetStatesSize(
        cudnn_handle, &states_size_in_bytes_));

    // TODO(zcd): Node that states is local var.
    Tensor states;
    states.Resize(framework::make_ddim({states_size_in_bytes_}));
    uint8_t *states_data = states->mutable_data<uint8_t>();

    size_t reserve_space_size_in_bytes_;
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetReserveSpaceSize(
        data_desc, &reserve_space_size_in_bytes_));

    ScopedDropoutGradDescriptor dropout_grad_desc;
    cudnnDropoutDescriptor_t cudnn_dropout_grad_desc =
        dropout_grad_desc.descriptor<T>(cudnn_handle, dropout_prob, states_data,
                                        states_size_in_bytes_, random_seed);

    CUDNN_ENFORCE(platform::dynload::cudnnDropoutBackward(
        cudnn_handle, cudnn_dropout_grad_desc, data_desc, grad_y->data<T>(),
        data_desc, grad_x->mutable_data<T>(), mask_data,
        reserve_space_size_in_bytes_));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

namespace plat = paddle::platform;
REGISTER_OP_KERNEL(dropout, CUDNN, plat::CUDAPlace,
                   ops::CUDNNDropoutKernel<float>,
                   ops::CUDNNDropoutKernel<double>);

REGISTER_OP_KERNEL(dropout_grad, CUDNN, plat::CUDAPlace,
                   ops::CUDNNDropoutGradKernel<float>,
                   ops::CUDNNDropoutGradKernel<double>);
