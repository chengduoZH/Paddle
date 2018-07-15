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

#include "paddle/fluid/framework/details/ssa_graph_builder.h"

namespace paddle {
namespace framework {
namespace details {
struct SSAGraph;

class OpFusionBuilder : public SSAGraphBuilder {
 public:
  explicit OpFusionBuilder(std::unique_ptr<SSAGraphBuilder>&& builder)
      : builder_(std::move(builder)) {}

  std::unique_ptr<SSAGraph> Build(const ProgramDesc& program) const override {
    auto graph = builder_->Build(program);

    // 1. 获取输出变量和op之间的对应关系

    // 2. 对graph进行拓扑排序

    // 3. 从输出到输入检测可以fuse的op

    // 3.1 op可以fuse的规则：目前只是对scale + element + moment进行合并

    // 将FC(mul + add_bias + relu)

    return graph;
  }

  int GetVarDeviceID(const std::string& var_name) const override {
    return builder_->GetVarDeviceID(var_name);
  }

  bool IsValidGraph(const SSAGraph* graph) const;

  void AnalysisDeepId(const SSAGraph* graph) const;

 private:
  std::unique_ptr<SSAGraphBuilder> builder_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
