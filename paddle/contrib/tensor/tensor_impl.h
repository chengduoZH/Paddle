// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "tc/core/tensor.h"

namespace tc {
namespace aten {

static DLDataType getDLDataType(const std::type_index& type) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = paddle::framework::SizeOfType(type);
  switch (type) {
    case std::type_index(typeid(double)):
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case std::type_index(typeid(float)):
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case std::type_index(typeid(int)):
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    default:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
  return dtype;
}

static std::type_index getDType(const DLDataType& dtype) {
  std::type_index type;
  switch (dtype) {
    case DLDataTypeCode::kDLFloat:
      type = std::type_index(typeid(double));
    case DLDataTypeCode::kDLFloat:
      type = std::type_index(typeid(float));
    case DLDataTypeCode::kDLInt:
      type = std::type_index(typeid(int));
      break;
    default:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
  return type;
}

static DLContext getDLContext(const paddle::framework::Tensor& src,
                              const int64_t& device_id) {
  DLContext ctx;
  ctx.device_id = device_id;
  if (paddle::platform::is_gpu_place(src.place())) {
    ctx.device_type = DLDeviceType::kDLGPU;
  } else {
    ctx.device_type = DLDeviceType::kDLCPU;
  }
  return ctx;
}

struct PTenDLMTensor {
  paddle::framework::Tensor handle;
  DLManagedTensor tensor;
};

// This function returns a shared_ptr to memory managed DLpack tensor
// constructed
// out of ATen tensor
DLManagedTensor* toDLPack(const paddle::framework::Tensor& src) {
  PTenDLMTensor* paddle_dl_tensor(new PTenDLMTensor);
  paddle_dl_tensor->handle = src;
  paddle_dl_tensor->tensor.manager_ctx = paddle_dl_tensor;
  paddle_dl_tensor->tensor.deleter = nullptr;
  paddle_dl_tensor->tensor.dl_tensor.data = src.data<void>();

  int64_t device_id =
      boost::get<paddle::platform::CUDAPlace>(src.place()).device;

  paddle_dl_tensor->tensor.dl_tensor.ctx = getDLContext(src, device_id);
  paddle_dl_tensor->tensor.dl_tensor.ndim = src.dims().size();
  paddle_dl_tensor->tensor.dl_tensor.dtype = getDLDataType(src.type());
  paddle_dl_tensor->tensor.dl_tensor.shape =
      const_cast<int64_t*>(paddle::framework::vectorize(src.dims()).data());
  paddle_dl_tensor->tensor.dl_tensor.strides =
      nullptr;  // const_cast<int64_t*>(src.strides().data());
  paddle_dl_tensor->tensor.dl_tensor.byte_offset = 0;
  return &(paddle_dl_tensor->tensor);
}

std::vector<int64_t> ToVector(int64_t* shape, int dims) {
  std::vector<int64_t> vec(dims);
  vec.insert(vec.begin(), shape, shape + dims);
  return vec;
}

paddle::framework::Tensor fromDLPack(const DLManagedTensor* src) {
  paddle::framework::Tensor tensor;

  int dev_id = paddle_dl_tensor->tensor.dl_tensor.ctx.device_id;
  auto dim = ToVector(src->dl_tensor.shape, src->dl_tensor.ndim);
  auto type = getDType(paddle_dl_tensor->tensor.dl_tensor.dtype);

  tensor.Resize(paddle::framework::make_ddim(dim));
  tensor.mutable_data(paddle::platform::CUDAPlace(dev_id), type);

  std::swap(tensor.data<void>(), src->dl_tensor.data);
  return tensor;
}

inline std::vector<DLTensorUPtr> makeDLTensors(
    const std::vector<paddle::framework::Tensor>& tensors) {
  std::vector<DLTensorUPtr> dlTensors;
  for (auto tensor : tensors) {
    auto dlMTensor = at::toDLPack(tensor);
    dlTensors.push_back(makeDLTensor(&(dlMTensor->dl_tensor)));
    dlMTensor->deleter(dlMTensor);
  }
  return dlTensors;
}

inline std::vector<DLConstTensorUPtr> makeDLConstTensors(
    const std::vector<paddle::framework::Tensor>& tensors) {
  std::vector<DLConstTensorUPtr> dlTensors;
  for (auto tensor : tensors) {
    auto dlMTensor = at::toDLPack(tensor);
    dlTensors.push_back(makeDLConstTensor(&(dlMTensor->dl_tensor)));
    dlMTensor->deleter(dlMTensor);
  }
  return dlTensors;
}

}  // namespace aten
}  // namespace tc
