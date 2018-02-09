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

#include "paddle/framework/chan_desc.h"
#include "gtest/gtest.h"

namespace paddle {
namespace framework {
TEST(ChanEleDesc, putdata) {
  ChanEleDesc chan_ele;
  auto* data = chan_ele.AddMetaData();
  data->SetType(proto::VarDesc_VarType_LOD_TENSOR);
  data->SetLoDLevel(0);
  data->SetDataType(proto::FP32);
  data->SetShape({1000, 784});

  data = chan_ele.AddMetaData();
  data->SetType(proto::VarDesc_VarType_LOD_TENSOR);
  data->SetLoDLevel(0);
  data->SetDataType(proto::FP32);
  data->SetShape({10, 4});
}

}  // namespace framework
}  // namespace paddle
