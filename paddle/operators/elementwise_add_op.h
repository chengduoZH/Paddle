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

#include "paddle/operators/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

template <typename T>
void Add_same(T* z, const T* x, const T* y, int64_t num) {
  for (int64_t i = 0; i < num; ++i) {
    z[i] = x[i] + y[i];
  }
}

template <typename T>
void Add_RowWise(T* z, const T* x, const T* y, int64_t x_num, int64_t y_num) {
  for (int64_t i = 0, j = 0; i < x_num; ++i, ++j) {
    if (UNLIKELY(j == y_num)) {
      j = 0;
    }
    z[i] = x[i] + y[j];
  }
}

template <typename T>
void Add_MidWise(T* z, const T* x, const T* y, int64_t x_num, int64_t y_num,
                 int64_t post) {
  for (int64_t i = 0, j = 0, y_i = 0; i < x_num; ++i, ++j) {
    y_i = j / post;
    if (UNLIKELY(y_i == y_num)) {
      j = 0;
      y_i = 0;
    }
    z[i] = x[i] + y[y_i];
  }
}

template <typename DeviceContext, typename T>
class ElementwiseAddKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());
    TransformFunctor<AddFunctor<T>, T, DeviceContext> functor(
        x, y, z, ctx.template device_context<DeviceContext>(), AddFunctor<T>());

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Rank of first input must >= rank of second input.");

    if (x_dims == y_dims) {
      //      functor.Run();
      Add_same(z->data<T>(), x->data<T>(), y->data<T>(),
               paddle::framework::product(x_dims));
      return;
    }

    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
    PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                   "Axis should be in range [0, x_dims)");

    int pre, n, post;
    get_mid_dims(x_dims, y_dims, axis, pre, n, post);
    if (post == 1) {
      Add_RowWise(z->data<T>(), x->data<T>(), y->data<T>(),
                  paddle::framework::product(x_dims), n);
      //      functor.RunRowWise(n, pre);
      return;
    } else {
      Add_MidWise(z->data<T>(), x->data<T>(), y->data<T>(),
                  paddle::framework::product(x_dims), n, post);
      //      functor.RunMidWise(n, pre, post);
      return;
    }
  }
};

template <typename T>
struct ElementwiseAddGradFunctor {
  template <typename Device, typename X, typename Y, typename Z, typename dX,
            typename dY, typename dZ>
  void operator()(Device d, X x, Y y, Z z, dX dx, dY dy, dZ dz) {
    auto dz_e = framework::EigenVector<T>::Flatten(*dz);
    if (dx) {
      auto dx_e = framework::EigenVector<T>::Flatten(*dx);
      dx_e.device(d) = dz_e;
    }
    if (dy) {
      auto dy_e = framework::EigenVector<T>::Flatten(*dy);
      dy_e.device(d) = dz_e;
    }
  }
};

template <typename T>
struct ElementwiseAddOneGradFunctor {
  template <typename Device, typename X, typename Y, typename Z, typename dX,
            typename dY, typename dZ>
  void operator()(Device d, X x, Y y, Z z, dX dx, dY dy, dZ dz) {
    auto dz_e = framework::EigenVector<T>::Flatten(*dz);
    if (dx) {
      auto dx_e = framework::EigenVector<T>::Flatten(*dx);
      dx_e.device(d) = dz_e;
    }
    if (dy) {
      auto dy_e = framework::EigenVector<T>::Flatten(*dy);
      dy_e.device(d) = dz_e.sum();
    }
  }
};

template <typename T>
struct ElementwiseAddBroadCastGradFunctor {
  template <typename Device, typename X, typename Y, typename Z, typename dX,
            typename dY, typename dZ, typename Pre, typename N>
  void operator()(Device d, X x, Y y, Z z, dX dx, dY dy, dZ dz, Pre pre, N n) {
    auto dz_e = framework::EigenVector<T>::Flatten(*dz);
    if (dx) {
      auto dx_e = framework::EigenVector<T>::Flatten(*dx);
      dx_e.device(d) = dz_e;
    }

    if (dy) {
      auto dy_e = framework::EigenVector<T>::Flatten(*dy);
      dy_e.device(d) = dz_e.reshape(Eigen::DSizes<int, 2>(pre, n))
                           .sum(Eigen::array<int, 1>{{0}});
    }
  }
};

template <typename T>
struct ElementwiseAddBroadCast2GradFunctor {
  template <typename Device, typename X, typename Y, typename Z, typename dX,
            typename dY, typename dZ, typename Pre, typename N, typename Post>
  void operator()(Device d, X x, Y y, Z z, dX dx, dY dy, dZ dz, Pre pre, N n,
                  Post post) {
    auto dz_e = framework::EigenVector<T>::Flatten(*dz);
    if (dx) {
      auto dx_e = framework::EigenVector<T>::Flatten(*dx);
      dx_e.device(d) = dz_e;
    }

    if (dy) {
      auto dy_e = framework::EigenVector<T>::Flatten(*dy);
      dy_e.device(d) = dz_e.reshape(Eigen::DSizes<int, 3>(pre, n, post))
                           .sum(Eigen::array<int, 2>{{0, 2}});
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseAddGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElementwiseGradCompute<DeviceContext, T, ElementwiseAddGradFunctor<T>,
                           ElementwiseAddOneGradFunctor<T>,
                           ElementwiseAddBroadCastGradFunctor<T>,
                           ElementwiseAddBroadCast2GradFunctor<T>>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle
