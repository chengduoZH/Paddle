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

#pragma once
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/im2col.h"
#include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class SequenceProjectKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    // need discuss, is it necessary to set zeros ?
    // Because if padding_trainable is false, padding data should be zeros.
    auto temp = framework::EigenVector<T>::Flatten(*out);
    temp.device(context.GetEigenDevice<Place>()) =
        temp.constant(static_cast<T>(0));

    auto place = context.GetEigenDevice<Place>();

    int context_start = context.Attr<int>("context_start");
    int context_length = context.Attr<int>("context_length");
    bool padding_trainable = context.Attr<bool>("padding_trainable");
    int context_stride = context.Attr<int>("context_stride");

    // InferShape by in_lod
    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");
    auto lod_level_0 = in->lod()[0];
    int64_t input_width = in->dims()[1];
    int64_t output_width = out->dims()[1];
    int64_t padding_width = 0;
    PADDLE_ENFORCE(input_width * context_length == output_width,
                   "Input size and pooling size should be consistent.");

    const LoDTensor* padding_data = nullptr;
    if (padding_trainable) {
      padding_data = context.Input<LoDTensor>("PaddingData");
      PADDLE_ENFORCE_EQ(padding_data->dims().size(), 2UL,
                        "Only support one level sequence now.");
      padding_width = padding_data->dims()[1];
      PADDLE_ENFORCE(padding_width == input_width,
                     "Input size and pooling size should be consistent.");
    }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int sequence_height, sequence_width;
    int input_row_begin, input_row_end;

    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kOCF, Place, float>
        im2col_ocf;

    for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
      input_row_begin = (context_start > 0)
                            ? static_cast<int>(lod_level_0[i]) + context_start
                            : static_cast<int>(lod_level_0[i]);
      input_row_end = static_cast<int>(lod_level_0[i + 1]);

      Tensor out_t = out->Slice<T>(static_cast<int>(lod_level_0[i]),
                                   static_cast<int>(lod_level_0[i + 1]));

      sequence_height = static_cast<int>(out_t.dims()[0]);
      sequence_width = static_cast<int>(in->dims()[1]);

      std::vector<int64_t> output_shape(
          {sequence_height, 1, 1, context_length,
           sequence_width});  // output_height, output_width,
      // input_channels, filter_height, filter_width
      out_t.Resize(framework::make_ddim(output_shape));

      if (input_row_begin < input_row_end) {
        Tensor in_t = in->Slice<T>(input_row_begin, input_row_end);
        std::vector<int64_t> input_shape(
            {1, input_row_end - input_row_begin,
             sequence_width});  // input_channels, input_height, input_width
        in_t.Resize(framework::make_ddim(input_shape));

        im2col_ocf(context.device_context(), in_t, out_t,
                   /*stride_height*/ context_stride, /*stride_width*/ 0, up_pad,
                   down_pad);
      }

      if (padding_trainable) {
        // add up trainable data
        out_t.Resize(framework::make_ddim(
            {sequence_height * context_length, sequence_width}));

        if (up_pad > 0) {  // add up pad
          //          int pad_size =
          //              std::min(-context_start,
          //                       static_cast<int>(lod_level_0[i + 1] -
          //                       lod_level_0[i]));
          for (int k = 0; k < up_pad; ++k) {
            Tensor out_t_sub = out_t.Slice<T>(
                k * context_length, k * context_length + (up_pad - k));
            Tensor w_sub = padding_data->Slice<T>(k, up_pad);
            // in this block, using EigenVector<T>::Flatten is ok too.
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            out_t_sub_e.device(place) = w_sub_e;
          }
        }
        if (down_pad > 0) {  // add down pad
          int down_pad_begin_row =
              std::max(0,
                       (sequence_height - context_start - context_length) + 1) +
              1;
          int padding_size =
              (sequence_height - context_start - context_length) > 0
                  ? 1
                  : context_length - (sequence_height - context_start);
          int padding_begin = sequence_height > context_start
                                  ? 0
                                  : context_start - sequence_height;

          for (int t = 0; t + down_pad_begin_row <= sequence_height; ++t) {
            Tensor out_t_sub = out_t.Slice<T>(
                (down_pad_begin_row + t) * context_length - t - padding_size,
                (down_pad_begin_row + t) * context_length);
            Tensor w_sub = padding_data->Slice<T>(
                up_pad + padding_begin,
                up_pad + padding_begin + t + padding_size);
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            out_t_sub_e.device(place) = w_sub_e;
          }
        }
      }
      out_t.Resize(framework::make_ddim(
          {sequence_height, context_length * sequence_width}));
    }
  }
};

template <typename Place, typename T>
class SequenceProjectGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    //    auto* in = context.Input<LoDTensor>("X");
    auto* out_g = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
    in_g->mutable_data<T>(context.GetPlace());
    auto place = context.GetEigenDevice<Place>();

    int context_start = context.Attr<int>("context_start");
    int context_length = context.Attr<int>("context_length");
    bool padding_trainable = context.Attr<bool>("padding_trainable");
    int context_stride = context.Attr<bool>("context_stride");

    // InferShape by in_lod
    PADDLE_ENFORCE_EQ(in_g->lod().size(), 1UL,
                      "Only support one level sequence now.");
    auto lod_g_level_0 = in_g->lod()[0];
    int64_t input_width = in_g->dims()[1];
    int64_t output_width = out_g->dims()[1];
    int64_t padding_width = 0;
    PADDLE_ENFORCE(input_width * context_length == output_width,
                   "Input size and pooling size should be consistent.");

    LoDTensor* padding_data = nullptr;
    if (padding_trainable) {
      padding_data = context.Output<LoDTensor>("PaddingData");
      padding_data->mutable_data<T>(context.GetPlace());
      PADDLE_ENFORCE_EQ(padding_data->dims().size(), 2UL,
                        "Only support one level sequence now.");
      padding_width = padding_data->dims()[1];
      PADDLE_ENFORCE(padding_width == input_width,
                     "Input size and pooling size should be consistent.");
    }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);

    paddle::operators::math::Col2ImFunctor<
        paddle::operators::math::ColFormat::kOCF, Place, float>
        col2im_ocf;

    for (int i = 0; i < static_cast<int>(lod_g_level_0.size()) - 1; ++i) {
      int row_begin = lod_g_level_0[i] + context_start;
      int row_end = lod_g_level_0[i + 1] + context_start;

      int input_row_bebin, input_row_end;
      input_row_bebin = row_begin > lod_g_level_0[i] ? row_begin : 0;
      input_row_end =
          row_end > lod_g_level_0[i + 1] ? lod_g_level_0[i + 1] : row_end;

      Tensor in_g_t = in_g->Slice<T>(input_row_bebin, input_row_end);
      Tensor out_g_t = out_g->Slice<T>(static_cast<int>(lod_g_level_0[i]),
                                       static_cast<int>(lod_g_level_0[i + 1]));

      int sequence_height = out_g_t.dims()[0];
      int sequence_width = out_g_t.dims()[1];

      if (padding_trainable) {
        // add up trainable data
        out_g_t.Resize(framework::make_ddim(
            {sequence_height * context_length, sequence_width}));

        if (row_begin < lod_g_level_0[i]) {  // add up pad
          for (int k = 0; k < up_pad; ++k) {
            Tensor out_t_sub = out_g_t.Slice<T>(
                k * context_length, k * context_length + (up_pad - k));
            Tensor w_sub = padding_data->Slice<T>(k, up_pad);
            // in this block, using EigenVector<T>::Flatten is ok too.
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            w_sub_e.device(place) = w_sub_e + out_t_sub_e;
            // out_t_sub_e.device(place) = 0;
          }
        }
        if (row_end > lod_g_level_0[i + 1]) {  // add down pad
          int down_pad_begin_row;
          int padding_rows = 0;
          if (row_begin <= lod_g_level_0[i]) {
            down_pad_begin_row =
                (sequence_height + up_pad - context_length) / context_stride;
            padding_rows =
                context_length - ((sequence_height + up_pad) % context_length);
          } else {
            down_pad_begin_row =
                (sequence_height - context_start - context_length) /
                context_stride;

            padding_rows = context_length -
                           ((sequence_height - context_start) % context_length);
          }

          for (int t = 1; t + down_pad_begin_row < sequence_height; ++t) {
            Tensor out_t_sub = out_g_t.Slice<T>(
                (down_pad_begin_row + t) * context_length * sequence_width -
                    (padding_rows - t + 1) * sequence_width,
                (down_pad_begin_row + t) * context_length * sequence_width);
            Tensor w_sub = padding_data->Slice<T>(
                up_pad + 1, up_pad + 1 + padding_rows + t);
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            out_t_sub_e.device(place) = w_sub_e;
          }
        }
      }

      std::vector<int64_t> output_shape(
          {sequence_height, 1, 1, context_length,
           sequence_width});  // output_height, output_width,
      // input_channels, filter_height, filter_width
      out_g_t.Resize(framework::make_ddim(output_shape));

      std::vector<int64_t> input_shape(
          {1, input_row_end - input_row_bebin,
           sequence_width});  // input_channels, input_height, input_width
      in_g_t.Resize(framework::make_ddim(input_shape));

      col2im_ocf(context.device_context(), in_g_t, out_g_t,
                 /*stride_height*/ context_stride, /*stride_width*/ 0, up_pad,
                 down_pad);
      // out_g_t back to orign size
    }
  }
};

}  // namespace operators
}  // namespace paddle
