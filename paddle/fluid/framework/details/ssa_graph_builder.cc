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
#include "paddle/fluid/framework/details/ssa_graph_builder.h"
#include <deque>
#include <utility>
#include <vector>

namespace paddle {
namespace framework {
namespace details {
void SSAGraphBuilder::PolishGraphToSupportDataHazards(SSAGraph *graph) {
  for (auto &var_map : graph->vars_) {
    for (auto &name_pair : var_map) {
      if (name_pair.second.size() <= 1) {
        continue;
      }
      auto it_new = name_pair.second.rbegin();
      auto it_old = name_pair.second.rbegin();
      ++it_old;
      for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
        auto *write_op = (*it_new)->generated_op_;
        auto &read_ops = (*it_old)->pending_ops_;

        for (auto *read_op : read_ops) {
          // Manually add a dependency var from read_op to write_op;
          if (read_op == write_op) {
            // Read Write is the same op.
            continue;
          }

          auto *dep_var = new DummyVarHandle();
          read_op->AddOutput(dep_var);
          write_op->AddInput(dep_var);
          graph->dep_vars_.emplace(dep_var);
        }
      }
    }
  }
}

void SSAGraphBuilder::RemoveIndependentGraph(SSAGraph *graph) {
  std::unordered_set<VarHandleBase *> visited_var;
  std::unordered_set<OpHandleBase *> visited_op;
  std::vector<std::unordered_set<OpHandleBase *>> op_groups;

  auto var_iter = graph->dep_vars_.begin();

  while (visited_op.size() != graph->ops_.size()) {
    std::unordered_set<OpHandleBase *> current_group;

    VarHandleBase *unvisited = nullptr;
    for (; var_iter != graph->dep_vars_.end(); ++var_iter) {
      if (visited_var.count(var_iter->get()) == 0) {
        unvisited = var_iter->get();
        visited_var.emplace(unvisited);
      }
    }

    if (unvisited == nullptr ||
        unvisited->pending_ops_.size() ==
            0) {  // TODO(zcd): for detail error info
      PADDLE_THROW("The graph has some error");
    }

    std::deque<OpHandleBase *> need_visite;

    for (auto op : unvisited->pending_ops_) {
      if (visited_op.count(op) == 0) {
        need_visite.push_back(op);
      }
    }

    while (!need_visite.empty()) {
      auto op = need_visite.front();
      need_visite.pop_back();

      visited_op.emplace(op);
      current_group.emplace(op);

      for (auto &out_var : op->Outputs()) {
        for (auto pend_op : out_var->pending_ops_) {
          if (visited_op.count(pend_op) == 0) {
            need_visite.push_back(pend_op);
            visited_op.emplace(pend_op);
            current_group.emplace(op);
          }
        }
      }

      for (auto &in_var : op->Inputs()) {
        auto gen_op = in_var->generated_op_;
        if (gen_op && visited_op.count(gen_op) == 0) {
          visited_op.emplace(gen_op);
          current_group.emplace(op);
        }
      }
    }

    op_groups.emplace_back(current_group);
  }
  if (op_groups.size() > 1) {
    VLOG(2) << "op_groups:size:" << op_groups.size();
  }
}

VarHandle *SSAGraphBuilder::CreateOrGetLatestVarHandle(
    SSAGraph *graph, const std::string &each_var_name,
    const platform::Place &place, size_t place_offset) {
  auto &var_holders = graph->vars_[place_offset];
  auto &var_holder = var_holders[each_var_name];
  VarHandle *var = nullptr;
  if (var_holder.empty()) {
    var = new VarHandle(0, place_offset, each_var_name, place);
    var_holder.emplace_back(var);
  } else {
    var = var_holder.rbegin()->get();
  }
  return var;
}

void SSAGraphBuilder::CreateOpOutput(SSAGraph *graph, OpHandleBase *op_handle,
                                     const std::string &each_var_name,
                                     const platform::Place &place,
                                     size_t place_offset) {
  auto &vars = graph->vars_[place_offset][each_var_name];
  size_t version = vars.size();
  auto var = new VarHandle(version, place_offset, each_var_name, place);
  vars.emplace_back(var);
  op_handle->AddOutput(var);
}

void SSAGraphBuilder::AddOutputToLeafOps(SSAGraph *graph) {
  for (auto &op : graph->ops_) {
    if (!op->Outputs().empty()) {
      continue;
    }
    auto *dummy_leaf = new DummyVarHandle();
    graph->dep_vars_.emplace(dummy_leaf);
    op->AddOutput(dummy_leaf);
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
