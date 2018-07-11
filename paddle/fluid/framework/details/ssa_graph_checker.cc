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

#include "paddle/fluid/framework/details/ssa_graph.h"
#include <map>
#include <string>
#include "paddle/fluid/framework/details/ssa_graph_checker.h"

namespace paddle {
namespace framework {
namespace details {

bool SSAGraghBuilderWithChecker::IsValidGraph(const SSAGraph *graph) const {
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
    std::stringstream sout;
    for (auto *op : set) {
      for (auto out : op->Outputs()) {
        ready_vars.emplace(out);
      }
      sout << op->Name() << " " << op->DebugString();
    }

    VLOG(2) << "\n Op groups:\n " << sout.str() << "\n";

    set.clear();
  };

  while (!pending_vars.empty()) {
    run_all_ops(ready_ops);

    if (ready_vars.empty()) {
      return false;
    }

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

void SSAGraghBuilderWithChecker::AnalysisDeepId(const SSAGraph *graph) const {
  std::unordered_map<OpHandleBase *, size_t> pending_ops;
  std::unordered_set<VarHandleBase *> pending_vars;
  std::unordered_set<VarHandleBase *> ready_vars;
  std::unordered_set<OpHandleBase *> ready_ops;

  std::map<int, std::unordered_set<OpHandleBase *>> m_dep_ops;

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

  int depth = 0;

  for (auto &op : graph->ops_) {
    if (op->Inputs().empty()) {
      ready_ops.insert(op.get());
    } else {
      pending_ops.insert({op.get(), op.get()->NoDupInputSize()});
    }
  }

  auto get_depth = [&](VarHandleBase *var) -> int {
    if (var->generated_op_) {
      if (var->generated_op_->deep_id == -1) {
        PADDLE_THROW("This graph has some err");
      } else {
        return var->generated_op_->deep_id;
      }
    }
    return 0;
  };

  auto run_all_ops = [&](std::unordered_set<OpHandleBase *> &set) {
    if (set.size()) {
      ++depth;
    }

    for (auto *op : set) {
      for (auto out : op->Outputs()) {
        ready_vars.emplace(out);
      }

      int deep = depth;
      for (auto &in_var : op->Inputs()) {
        if (deep < get_depth(in_var)) {
          deep = get_depth(in_var);
        }
      }
      op->deep_id = deep;
      m_dep_ops[deep].emplace(op);
    }
    set.clear();
  };

  while (!pending_vars.empty()) {
    run_all_ops(ready_ops);

    if (ready_vars.empty()) {
      PADDLE_THROW("exception");
    }

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

  VLOG(2) << "The depth:" << m_dep_ops.size();
  for (auto &dep_ops : m_dep_ops) {
    VLOG(2) << "Depth:" << dep_ops.first << ", num:" << dep_ops.second.size();
    std::stringstream sout;
    for (auto &op : dep_ops.second) {
      sout << op->Name() << "(depth: " << op->deep_id << ")  "
           << op->DebugString();
    }
    VLOG(2) << "\n" << sout.str();
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
