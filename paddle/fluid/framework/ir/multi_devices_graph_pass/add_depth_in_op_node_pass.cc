//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/fused_all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/op_graph_view.h"
namespace paddle {
namespace framework {
namespace ir {

class AddDepthInOpNodePass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override {
    // get all the op node
    auto all_ops = ir::FilterByNodeWrapper<details::OpHandleBase>(*graph);
    OpGraphView graph_view(all_ops);

    int64_t depth = 0;
    auto op_deps = graph_view.GetPrecedingDepNum();
    size_t op_num = op_deps.size();

    std::unordered_set<details::OpHandleBase *> visited_ops;
    std::queue<details::OpHandleBase *> ready_ops;
    for (auto iter = op_deps.begin(); iter != op_deps.end();) {
      if (iter->second != 0) {
        ++iter;
        continue;
      }
      ready_ops.push(iter->first);
      visited_ops.insert(iter->first);
      iter->first->SetDepth(depth);
      op_deps.erase(iter++);
    }

    while (true) {
      std::queue<details::OpHandleBase *> next_ready_ops;
      ++depth;
      while (!ready_ops.empty()) {
        auto *cur_op = ready_ops.front();
        ready_ops.pop();

        auto &pending_ops = graph_view.PendingOps(cur_op);
        for (auto *pending_op : pending_ops) {
          if (visited_ops.count(pending_op) > 0) {
            continue;
          }

          if (--op_deps.at(pending_op) == 0) {
            visited_ops.insert(pending_op);
            pending_op->SetDepth(depth);
            op_deps.erase(pending_op);
            next_ready_ops.push(pending_op);
          }
        }
      }
      if (next_ready_ops.size() > 0) {
        std::swap(ready_ops, next_ready_ops);
      } else {
        break;
      }
    }

    PADDLE_ENFORCE_EQ(visited_ops.size(), op_num, "There are unvisited ops");
    PADDLE_ENFORCE(op_deps.empty(), "There are unvisited ops");
    //     1. not split op according to device

    // from top to bottom, set the depth

    // from bottom to top, set the depth

    // 2.  split op according to device

    // from top to bottom, set the depth

    // from bottom to top, set the depth
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(add_depth_in_no_node_pass,
              paddle::framework::ir::AddDepthInOpNodePass);
