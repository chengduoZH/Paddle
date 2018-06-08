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
#include <utility>

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

std::unique_ptr<SSAGraph> SSAGraphBuilder::BuildAndCheck(
    const ProgramDesc &program) {
  std::unique_ptr<SSAGraph> graph = Build(program);
  PADDLE_ENFORCE(IsValidGraph(graph.get()));
  return std::move(graph);
}

bool SSAGraphBuilder::IsValidGraph(const SSAGraph *graph) const {
  std::unordered_map<OpHandleBase *, size_t> pending_ops;
  std::unordered_set<VarHandleBase *> pending_vars;
  std::unordered_set<VarHandleBase *> ready_vars;
  std::unordered_set<OpHandleBase *> ready_ops;

  auto insert_pending_var = [&](VarHandleBase *var) {
    pending_vars.insert(var);
    if (var->generated_op_ == nullptr) {
      ready_vars.emplace(var);
    }
  };

  for (auto &var_map : graph->vars_) {
    for (auto &name_pair : var_map) {
      for (auto &version_pair : name_pair.second) {
        insert_pending_var(version_pair.get());
      }
    }
  }

  for (auto &var : graph->dep_vars_) {
    insert_pending_var(var.get());
  }

  for (auto &op : graph->ops_) {
    if (op->Inputs().empty()) {
      ready_ops.insert(op.get());
    } else {
      pending_ops.insert({op.get(), op.get()->NoDupInputSize()});
    }
  }

  auto run_all_ops = [&](std::unordered_set<OpHandleBase *> &set) {
    for (auto *op : set) {
      for (auto out : op->Outputs()) {
        ready_vars.emplace(out);
      }
    }
    set.clear();
  };

  while (!pending_vars.empty()) {
    if (ready_ops.empty()) {
      if (VLOG_IS_ON(4)) {
        for (auto pend_var : pending_vars) {
          VLOG(4) << pend_var->DebugString();
        }
      }
      return false;
    }

    run_all_ops(ready_ops);

    for (auto ready_var : ready_vars) {
      pending_vars.erase(ready_var);
      for (auto *op : ready_var->pending_ops_) {
        auto &deps = --pending_ops[op];
        if (deps == 0) {
          ready_ops.insert(op);
        }
      }
    }
    ready_vars.clear();
  }
  return true;
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
