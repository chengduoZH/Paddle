//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/graph_op_fusion_builder.h"
#include "paddle/fluid/framework/op_info.h"

namespace paddle {
namespace framework {
namespace details {

GraphOpFusionBuilder::GraphOpFusionBuilder(const std::string &loss_var_name)
    : loss_var_name_(loss_var_name) {}

ProgramDesc *GraphOpFusionBuilder::Build(ProgramDesc *program) const {
  // 1. 获取输出变量和op之间的对应关系
  //  for (auto *var : program->Block(0).AllVars()) {
  //    all_vars_.emplace(var->Name(), var);
  //  }
  //
  //  std::unordered_map<std::string, >
  //
  //  for (auto *op : program->Block(0).AllOps()) {
  //  }

  // 2. 对graph进行拓扑排序

  // 3. 从输出到输入检测可以fuse的op

  // 3.1 op可以fuse的规则：目前只是对scale + element + moment进行合并

  // 将FC(mul + add_bias + relu)

  return program;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
