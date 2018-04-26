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

#pragma once
#include <vector>
#include "paddle/fluid/operators/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct MulFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a * b; }
};

template <typename DeviceContext, typename T>
class ElementwiseMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          MulFunctor<T>(), z);

    {
      std::vector<T> xv;
      framework::TensorToVector(*x, ctx.device_context(), &xv);
      ctx.device_context().Wait();
      T total = 0.0;
      for (T v : xv) {
        T v1 = v;
        if (v1 < 0) {
          v1 = -v1;
        }
        total += v1;
      }
      fprintf(stderr, "fw_elementwise_mul_x: %f\n", static_cast<double>(total));
      VLOG(1) << "fw_elementwise_mul_x:" << total << " " << x->dims();
    }
    {
      std::vector<T> xv;
      framework::TensorToVector(*y, ctx.device_context(), &xv);
      ctx.device_context().Wait();
      T total = 0.0;
      for (T v : xv) {
        T v1 = v;
        if (v1 < 0) {
          v1 = -v1;
        }
        total += v1;
      }
      fprintf(stderr, "fw_elementwise_mul_y: %f\n", static_cast<double>(total));
      VLOG(1) << "fw_elementwise_mul_y:" << total << " " << y->dims();
    }
    {
      std::vector<T> xv;
      framework::TensorToVector(*z, ctx.device_context(), &xv);
      ctx.device_context().Wait();
      T total = 0.0;
      for (T v : xv) {
        T v1 = v;
        if (v1 < 0) {
          v1 = -v1;
        }
        total += v1;
      }
      fprintf(stderr, "fw_elementwise_mul_z: %f\n", static_cast<double>(total));
      VLOG(1) << "fw_elementwise_mul_z:" << total << " " << z->dims();
    }
  }
};

template <typename T>
struct MulGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * y; }
};

template <typename T>
struct MulGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * x; }
};

template <typename DeviceContext, typename T>
class ElementwiseMulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, MulGradDX<T>(), MulGradDY<T>());

    {
      std::vector<T> xv;
      framework::TensorToVector(*dx, ctx.device_context(), &xv);
      ctx.device_context().Wait();
      T total = 0.0;
      for (T v : xv) {
        T v1 = v;
        if (v1 < 0) {
          v1 = -v1;
        }
        total += v1;
      }
      fprintf(stderr, "bk_elementwise_mul_dx: %f\n",
              static_cast<double>(total));
      VLOG(1) << "bk_elementwise_mul_dx:" << total << " " << dx->dims();
    }
    {
      std::vector<T> xv;
      framework::TensorToVector(*dy, ctx.device_context(), &xv);
      ctx.device_context().Wait();
      T total = 0.0;
      for (T v : xv) {
        T v1 = v;
        if (v1 < 0) {
          v1 = -v1;
        }
        total += v1;
      }
      fprintf(stderr, "bk_elementwise_mul_dy: %f\n",
              static_cast<double>(total));
      VLOG(1) << "bk_elementwise_mul_dy:" << total << " " << dy->dims();
    }
    {
      std::vector<T> xv;
      framework::TensorToVector(*dout, ctx.device_context(), &xv);
      ctx.device_context().Wait();
      T total = 0.0;
      for (T v : xv) {
        T v1 = v;
        if (v1 < 0) {
          v1 = -v1;
        }
        total += v1;
      }
      fprintf(stderr, "bk_elementwise_mul_dout: %f\n",
              static_cast<double>(total));
      VLOG(1) << "bk_elementwise_mul_dout:" << total << " " << dout->dims();
    }
  }
};
}  // namespace operators
}  // namespace paddle
