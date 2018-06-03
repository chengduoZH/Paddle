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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FuseVarsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_vars = context.MultiInputVar("X");
    auto fused_var = context.OutputVar("FusedX");
    auto out_vars = context.MultiOutputVar("Y");

    int64_t total_numel = 0;
    std::vector<size_t> vars_in_gpu;
    framework::LoDTensor first_in_tensor;

    for (size_t i = 0; i < in_vars.size(); ++i) {
      PADDLE_ENFORCE_EQ(in_vars[i], out_vars[i]);
      PADDLE_ENFORCE(in_vars[i]->IsType<paddle::framework::LoDTensor>());
      auto in_t = in_vars[i]->Get<paddle::framework::LoDTensor>();
      if (platform::is_gpu_place(in_t.place())) {
        if (vars_in_gpu.size() == 0) {
          vars_in_gpu.push_back(i);
          first_in_tensor = in_t;
        } else {
          PADDLE_ENFORCE(
              platform::is_same_place(in_t.place(), first_in_tensor.place()));
          PADDLE_ENFORCE_EQ(in_t.type().hash_code(),
                            first_in_tensor.type().hash_code());
        }
        auto numel = in_vars[i]->Get<paddle::framework::LoDTensor>().numel();
        PADDLE_ENFORCE_GE(numel, 0);
        total_numel += numel;
      }
    }

    if (vars_in_gpu.size() != 0) {
      auto fused_t = fused_var->GetMutable<paddle::framework::LoDTensor>();
      fused_t->Resize(paddle::framework::DDim({total_numel}))
          .mutable_data(first_in_tensor.place());

      int s = 0;
      for (size_t i = 0; i < vars_in_gpu.size(); ++i) {
        auto out_t = out_vars[vars_in_gpu[i]]
                         ->GetMutable<paddle::framework::LoDTensor>();
        int64_t mem_size = in_vars[vars_in_gpu[i]]
                               ->Get<paddle::framework::LoDTensor>()
                               .numel();
        out_t->ShareDataWith(out_t->Slice(s, s + mem_size));
        s += mem_size;
      }
    } else {
      //
    }
  }
};
}  // namespace operators
}  // namespace paddle
