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

namespace paddle {
namespace framework {

ChanEleDesc::ChanEleDesc(const proto::ChanEleDesc &desc) {
  desc_ = desc;
  for (auto &meta_data_desc : *desc_.mutable_meta_datas()) {
    chan_ele_.emplace_back(new VarDesc(meta_data_desc));
  }
}

ChanEleDesc::ChanEleDesc(const ChanEleDesc &o) {
  desc_ = o.desc_;
  for (int i = 0; i < desc_.meta_datas_size(); ++i) {
    auto *meta_data = desc_.mutable_meta_datas(i);
    chan_ele_.emplace_back(new VarDesc(*meta_data));
  }
}

ChanEleDesc::ChanEleDesc(const std::string &binary_str) {
  PADDLE_ENFORCE(desc_.ParseFromString(binary_str),
                 "Fail to parse chan_ele_desc from binary string.");
  for (auto &meta_data_desc : *desc_.mutable_meta_datas()) {
    chan_ele_.emplace_back(new VarDesc(meta_data_desc));
  }
}

VarDesc *ChanEleDesc::AddChanMetaData() {
  std::string name = "temp";  // TODO(zcd): add prefix "chan_"
  auto *var = new VarDesc(name);
  return var;
}

VarDesc *ChanEleDesc::AppendChanMetaData(const VarDesc &var) {
  chan_ele_.emplace_back(new VarDesc(var));
  return chan_ele_.back().get();
}

proto::ChanEleDesc *ChanEleDesc::Proto() {
  for (auto &chan_ele : chan_ele_) {
    chan_ele->Flush();
  }
  return &desc_;
}

ChanEleDesc::ChanEleDesc() {}

////
//
// ChanEleDesc *ChanDesc::AppendChanEle(const ChanEleDesc &chan_ele) {
//  chan_.emplace_back(new ChanEleDesc(chan_ele));
//  return chan_.back().get();
//}
//
// proto::ChanDesc *ChanDesc::Proto() {
//  for (auto &block : chan_) {
//    block->Flush();
//  }
//  return &desc_;
//}
//
// ChanDesc::ChanDesc() {
//}
//
// ChanDesc::ChanDesc(const ChanDesc &o) {
//  desc_ = o.desc_;
//
//  for (int i = 0; i < desc_.blocks_size(); ++i) {
////    auto *chan_ele = desc_.mutable_blocks(i);
//    chan_.emplace_back(new ChanEleDesc(*o.chan_[i]));
//  }
//}
//
// ChanDesc::ChanDesc(const proto::ChanDesc &desc) {
//  desc_ = desc;
//  for (auto &block_desc : *desc_.mutable_blocks()) {
//    chan_.emplace_back(new BlockDesc(this, &block_desc));
//  }
//}
//
// ChanDesc::ChanDesc(const std::string &binary_str) {
//  PADDLE_ENFORCE(desc_.ParseFromString(binary_str),
//                 "Fail to parse program_desc from binary string.");
//  for (auto &block_desc : *desc_.mutable_blocks()) {
//    blocks_.emplace_back(new BlockDesc(this, &block_desc));
//  }
//}

}  // namespace framework
}  // namespace paddle
