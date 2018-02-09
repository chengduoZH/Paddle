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

#include <vector>
#include "glog/logging.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/var_desc.h"

namespace paddle {
namespace framework {

class ChanEleDesc {
 public:
  ChanEleDesc(){};

  explicit ChanEleDesc(const proto::ChanEleDesc &desc);

  ChanEleDesc(const ChanEleDesc &o);

  explicit ChanEleDesc(const std::string &binary_str);

  VarDesc *AddChanMetaData();

  VarDesc *AppendChanMetaData(const VarDesc &ele);

  VarDesc *MutableChanMetaData(size_t idx) { return chan_ele_[idx].get(); }

  //  std::vector<VarDesc *> AllMetaData() const { return chan_ele_; }

  const VarDesc &AddChanMetaData(size_t idx) const { return *chan_ele_[idx]; }

  size_t Size() const { return chan_ele_.size(); }

  proto::ChanEleDesc *Proto();

 private:
  proto::ChanEleDesc desc_;

  std::vector<std::unique_ptr<VarDesc>> chan_ele_;
};
}  // namespace framework
}  // namespace paddle
