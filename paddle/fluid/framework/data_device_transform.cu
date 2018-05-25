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

#include "paddle/fluid/framework/data_device_transform.h"

namespace paddle {
namespace framework {

extern proto::VarType::Type ToDataType(std::type_index type);

template <typename T>
__global__ void KeDataTransFromCPUToGPU(const T in_value, T* out_value) {
  out_value[0] = in_value;
}

void DataTransFromCPUToGPU(const Tensor& src, const platform::Place& dst_place,
                           const platform::DeviceContext& dev_ctx,
                           Tensor* dst) {
  src.check_memory_size();
  auto type = src.type();

  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  dst->mutable_data(dst_place, type);

  auto& cuda_ctx =
      reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx);

  switch (ToDataType(type)) {
    case proto::VarType::FP32:
      KeDataTransFromCPUToGPU<<<1, 1, 0, cuda_ctx.stream()>>><float>(
          src.data<float>()[0], dst->mutable_data<float>(dst_place));
      break;
    case proto::VarType::FP64:
      KeDataTransFromCPUToGPU<<<1, 1, 0, cuda_ctx.stream()>>><double>(
          src.data<double>()[0], dst->mutable_data<double>(dst_place));
      break;
    case proto::VarType::INT32:
      KeDataTransFromCPUToGPU<<<1, 1, 0, cuda_ctx.stream()>>><int>(
          src.data<int>()[0], dst->mutable_data<int>(dst_place));
      break;
    case proto::VarType::INT64:
      KeDataTransFromCPUToGPU<<<1, 1, 0, cuda_ctx.stream()>>><int64_t>(
          src.data<int64_t>()[0], dst->mutable_data<int64_t>(dst_place));
      break;
    default:
      PADDLE_THROW("Not supported %d", type);
  }
}

static const platform::DeviceContext* GetDeviceContext(
    const platform::Place& src_place, const platform::Place& dst_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  if (platform::is_gpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    return pool.Get(src_place);
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    return pool.Get(dst_place);
  } else {
    PADDLE_THROW(
        "Currently, model parallelism is only supported between CPU and CUDA");
  }
}

void TransDataDevice(const Tensor& in, const platform::Place& dst_place,
                     Tensor* out) {
  VLOG(3) << "DeviceTransform in, src_place " << in.place()
          << " dst_place: " << dst_place << "  " << in.dims();
  auto* dev_ctx = GetDeviceContext(in.place(), dst_place);

  // FIXME(zcd): TransDataDevice is used to transform data from GPU to CPU and
  // the enforced checkings have been done in GetDeviceContext, so the
  // `dev_ctx->Wait()` is necessary. But `dev_ctx->Wait()` will make the program
  // slow, especially the number of elements is one. So one solution is to use a
  // cuda kernel to complete the copy operation when the transforming is from
  // CPU to GPU and the number of elements is one.
  if (platform::is_cpu_place(in.place()) && platform::is_gpu_place(dst_place) &&
      in.numel() == 1) {
    PADDLE_ENFORCE(platform::is_gpu_place(dev_ctx->GetPlace()));
#ifdef __NVCC__
    DataTransFromCPUToGPU(in, dst_place, *dev_ctx, out);
#endif
  } else {
    TensorCopy(in, dst_place, *dev_ctx, out);
    dev_ctx->Wait();
  }
}

}  // namespace framework
}  // namespace paddle
