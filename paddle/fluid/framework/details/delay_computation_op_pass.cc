// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

class DelayCompuationOpPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override {
    std::vector<ir::Node*> sorted_ops = ir::TopologySortOperations(*graph);

    std::vector<ir::Node*> ops;
    std::vector<ir::Node*> vars;
    auto nodes = graph->ReleaseNodes();
    std::unordered_map<ir::Node*, size_t> nodes_map;

    for (size_t i = 0; i < nodes.size(); ++i) {
      auto* n = nodes[i].get();
      nodes_map.emplace(n, i);

      if (n->IsWrappedBy<OpHandleBase>()) {
        ops.emplace_back(n);
      } else if (n->IsWrappedBy<VarHandleBase>()) {
        vars.emplace_back(n);
      } else {
        PADDLE_THROW("Error.");
      }
    }

    for (auto& node : sorted_ops) {
      //      node->
    }
  }
};
}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delay_all_reduce_op_pass,
              paddle::framework::details::DelayCompuationOpPass);
