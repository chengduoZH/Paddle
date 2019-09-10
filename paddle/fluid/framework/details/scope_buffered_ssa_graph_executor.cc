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

#include "paddle/fluid/framework/details/scope_buffered_ssa_graph_executor.h"
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/profiler.h"
namespace paddle {
namespace framework {
namespace details {
ScopeBufferedSSAGraphExecutor::ScopeBufferedSSAGraphExecutor(
    ExecutionStrategy strategy, std::vector<Scope *> local_scopes,
    std::vector<Scope *> local_exec_scopes, std::vector<VariableInfo> var_infos,
    std::vector<platform::Place> places,
    std::unique_ptr<SSAGraphExecutor> &&underlying_executor)
    : strategy_(std::move(strategy)),
      underlying_executor_(std::move(underlying_executor)),
      local_scopes_(std::move(local_scopes)),
      local_exec_scopes_(std::move(local_exec_scopes)),
      var_infos_(std::move(var_infos)),
      places_(std::move(places)) {
  PADDLE_ENFORCE_EQ(local_scopes_.size(), local_exec_scopes_.size());
  pre_local_exec_scopes_.resize(local_exec_scopes_.size());
  post_local_exec_scopes_.resize(local_exec_scopes_.size());
  PrepareLocalExeScopes();
}

static void CaculateAllocations(const std::vector<Scope *> &local_scopes,
                                const std::vector<Scope *> &local_exe_scopes,
                                const std::vector<platform::Place> &places) {
  size_t scope_idx = 0;
  for (auto &scope : local_scopes) {
    VLOG(1) << "scope " << scope << ", scope_idx " << scope_idx
            << ", local scopes num " << scope->kids().size()
            << ", Recursive local scope num: "
            << scope->RecursiveGetLocalScope().size();
    size_t cpu_bytes = 0, gpu_bytes = 0;
    AnalysisScope(*scope, &cpu_bytes, &gpu_bytes);
    VLOG(1) << "!!!!!!!!! " << scope << "  bytes: "
            << static_cast<double>(cpu_bytes + gpu_bytes) / 1024.0 / 1024.0 /
                   1024.0
            << " GB"
            << "  cpu bytes: "
            << static_cast<double>(cpu_bytes) / 1024.0 / 1024.0 / 1024.0
            << " GB"
            << "  gpu bytes: "
            << static_cast<double>(gpu_bytes) / 1024.0 / 1024.0 / 1024.0
            << " GB";
    cpu_bytes = 0, gpu_bytes = 0;
    PrintMemoryUsage(scope, &cpu_bytes, &gpu_bytes);
    VLOG(1) << "!!!!!!!!! " << scope << "  bytes(included local scope): "
            << static_cast<double>(cpu_bytes + gpu_bytes) / 1024.0 / 1024.0 /
                   1024.0
            << " GB"
            << "  cpu bytes: "
            << static_cast<double>(cpu_bytes) / 1024.0 / 1024.0 / 1024.0
            << " GB"
            << "  gpu bytes: "
            << static_cast<double>(gpu_bytes) / 1024.0 / 1024.0 / 1024.0
            << " GB";
    auto local_exe_scope = local_exe_scopes[scope_idx];
    VLOG(1) << "local_exe_scope " << local_exe_scope;
    cpu_bytes = 0, gpu_bytes = 0;
    PrintMemoryUsage(local_exe_scope, &cpu_bytes, &gpu_bytes);
    VLOG(1) << "!!!!!!!!! " << local_exe_scope << " bytes: "
            << static_cast<double>(cpu_bytes + gpu_bytes) / 1024.0 / 1024.0 /
                   1024.0
            << " GB"
            << "  cpu bytes: "
            << static_cast<double>(cpu_bytes) / 1024.0 / 1024.0 / 1024.0
            << " GB"
            << "  gpu bytes: "
            << static_cast<double>(gpu_bytes) / 1024.0 / 1024.0 / 1024.0
            << " GB";
    scope_idx++;
  }
}

struct TensorVisitor {
  void operator()(LoDTensor *lod_tensor) {
    if (lod_tensor->IsInitialized()) {
      if (platform::is_cpu_place(lod_tensor->place())) {
        lod_tensor->clear();
      }
    }
  }

  void operator()(SelectedRows *selectedrows) {
    if (selectedrows->value().IsInitialized()) {
      if (platform::is_cpu_place(selectedrows->value().place())) {
        selectedrows->mutable_value()->clear();
      }
    }
  }

  void operator()(LoDTensorArray *array) {
    std::vector<memory::Allocation *> result;
    for (auto &tensor : *array) {
      if (tensor.IsInitialized()) {
        if (platform::is_cpu_place(tensor.place())) {
          tensor.clear();
        }
      }
    }
  }
};

template <typename Func>
void VisitVariable(Variable *var, Func *func) {
  if (var->IsType<LoDTensor>()) {
    (*func)(var->GetMutable<LoDTensor>());
  } else if (var->IsType<SelectedRows>()) {
    (*func)(var->GetMutable<SelectedRows>());
  } else if (var->IsType<LoDTensorArray>()) {
    (*func)(var->GetMutable<LoDTensorArray>());
  }
}

FeedFetchList ScopeBufferedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  if (drop_scope_counter_ == 0) {
    platform::RecordEvent e("InitLocalVars");
    InitVariables();
  }

  // collect local execution scope
  for (size_t scope_id = 0; scope_id < local_exec_scopes_.size(); ++scope_id) {
    pre_local_exec_scopes_.at(scope_id).clear();
    auto scopes = local_exec_scopes_.at(scope_id)->kids();
    pre_local_exec_scopes_.at(scope_id).insert(scopes.begin(), scopes.end());
  }

  std::vector<framework::LoDTensor> fetch_data;
  std::exception_ptr eptr = nullptr;
  try {
    fetch_data = underlying_executor_->Run(fetch_tensors);
  } catch (...) {
    eptr = std::current_exception();
  }

  // collect local execution scope
  for (size_t scope_id = 0; scope_id < local_exec_scopes_.size(); ++scope_id) {
    post_local_exec_scopes_.at(scope_id).clear();
    auto scopes = local_exec_scopes_.at(scope_id)->kids();
    post_local_exec_scopes_.at(scope_id).insert(scopes.begin(), scopes.end());
  }

  history_local_exec_scopes_.emplace_back();
  auto &incr_local_exec_scopes = history_local_exec_scopes_.back();
  incr_local_exec_scopes.resize(local_exec_scopes_.size());
  for (size_t scope_id = 0; scope_id < local_exec_scopes_.size(); ++scope_id) {
    std::set_difference(
        post_local_exec_scopes_.at(scope_id).begin(),
        post_local_exec_scopes_.at(scope_id).end(),
        pre_local_exec_scopes_.at(scope_id).begin(),
        pre_local_exec_scopes_.at(scope_id).end(),
        std::inserter(incr_local_exec_scopes.at(scope_id),
                      incr_local_exec_scopes.at(scope_id).begin()));
    post_local_exec_scopes_.at(scope_id).clear();
    pre_local_exec_scopes_.at(scope_id).clear();
    std::stringstream out;
    out << scope_id << " kids: ";
    for (auto &scope : incr_local_exec_scopes.at(scope_id)) {
      out << scope << ", ";
    }
    VLOG(1) << out.str();

    TensorVisitor tensor_visitor;
    for (auto &scope : incr_local_exec_scopes.at(scope_id)) {
      auto var_set = scope->GetLocalVars();
      for (auto &var : var_set) {
        PADDLE_ENFORCE(var->IsInitialized());
        VisitVariable(var, &tensor_visitor);
      }
    }
  }

  size_t history_step = history_local_exec_scopes_.size();
  if (fetch_tensors.size() && history_step >= 2) {
    VLOG(1) << "fetch_tensors.size() && history_step >= 2; " << history_step;
    for (size_t i = 0; i < history_step - 1; ++i) {
      auto &pre_incr_local_exec_scopes = history_local_exec_scopes_.front();
      for (size_t scope_idx = 0; scope_idx < pre_incr_local_exec_scopes.size();
           ++scope_idx) {
        for (auto scope : pre_incr_local_exec_scopes[scope_idx]) {
          VLOG(1) << "delete: " << scope;
          local_exec_scopes_.at(scope_idx)->DeleteScope(scope);
        }
      }
      history_local_exec_scopes_.pop_front();
    }
  }
  VLOG(1) << "history_local_exec_scopes_";

  CaculateAllocations(local_scopes_, local_exec_scopes_, places_);
  ++drop_scope_counter_;
  if (drop_scope_counter_ == strategy_.num_iteration_per_drop_scope_) {
    DropLocalExeScopes();
    VLOG(1) << "DropLocalExeScopes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
    CaculateAllocations(local_scopes_, local_exec_scopes_, places_);
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  } else {
    return fetch_data;
  }
}

void ScopeBufferedSSAGraphExecutor::InitVariables() {
  for (auto &info : tmp_var_infos_) {
    for (auto &pair : info) {
      InitializeVariable(pair.first, pair.second);
    }
  }

  const ir::Graph &graph = Graph();
  if (graph.Has(details::kProgramDescs)) {
    auto &program_descs =
        graph.Get<details::ProgramDescs>(details::kProgramDescs);
    // Init vars
    auto &fused_grad_vars = graph.Get<details::FusedVars>(details::kFusedVars);
    for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
      for (auto &var_name : fused_grad_vars) {
        auto var = local_exec_scopes_[i]->Var(var_name);
        var->GetMutable<LoDTensor>();
      }
    }

    for (auto &program_desc : program_descs) {
      for (auto &op_desc : program_desc.Block(0).AllOps()) {
        for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
          auto op = OpRegistry::CreateOp(*op_desc);
          op->Run(*local_exec_scopes_[i], places_[i]);
        }
      }
    }
  }
}

void ScopeBufferedSSAGraphExecutor::DropLocalExeScopes() {
  platform::RecordEvent drop_scope_event("DropLocalExeScopes");
  drop_scope_counter_ = 0;
  for (auto &p : places_) {
    platform::DeviceContextPool::Instance().Get(p)->Wait();
  }

  for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
    local_exec_scopes_[i]->EraseVarsExcept(preserve_vars_[i]);
    local_exec_scopes_[i]->DropKids();
    for (auto &preserve_var : preserve_vars_[i]) {
      preserve_var->Clear();
    }
    VLOG(3) << "Drop local execution scope: " << local_scopes_[i];
  }
}

void ScopeBufferedSSAGraphExecutor::PrepareLocalExeScopes() {
  // Create local scopes.
  preserve_vars_.resize(local_scopes_.size());
  tmp_var_infos_.resize(local_scopes_.size());

  for (auto it = local_scopes_.rbegin(); it != local_scopes_.rend(); ++it) {
    size_t idx = local_scopes_.size() - 1 - (it - local_scopes_.rbegin());
    auto *scope = local_scopes_[idx];
    auto *local_scope = local_exec_scopes_[idx];

    for (auto &info : var_infos_) {
      if (info.persistable_) {  // Persistable
        auto var = scope->FindVar(info.name_);
        if (var != nullptr) {
          VLOG(2)
              << info.name_
              << " has been initialized beforehand in global scope, skipped";
          continue;
        }
        InitializeVariable(scope->Var(info.name_), info.type_);
      } else {
        Variable *tmp_var = local_scope->Var(info.name_);
        preserve_vars_[idx].emplace(tmp_var);
        tmp_var_infos_[idx].emplace_back(tmp_var, info.type_);
      }
    }
  }
}

bool ScopeBufferedSSAGraphExecutor::NeedCreateLocalExeScope() {
  return drop_scope_counter_ == 0;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
