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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/profiler.h"
namespace paddle {
namespace framework {
namespace details {
ScopeBufferedSSAGraphExecutor::ScopeBufferedSSAGraphExecutor(
    ExecutionStrategy strategy, std::vector<Scope *> local_scopes,
    std::vector<VariableInfo> var_infos, std::vector<platform::Place> places,
    std::unique_ptr<SSAGraphExecutor> &&underlying_executor)
    : strategy_(std::move(strategy)),
      underlying_executor_(std::move(underlying_executor)),
      local_scopes_(std::move(local_scopes)),
      var_infos_(std::move(var_infos)),
      places_(std::move(places)) {}

FeedFetchList ScopeBufferedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  std::unordered_map<Variable *, std::pair<std::string, int>> var_names;
  std::unordered_map<LoDTensor *, std::pair<std::string, int>> tensor_names;
  std::unordered_map<void *, std::pair<std::string, int>> data_names;

  if (drop_scope_counter_ == 0) {
    // Create local scopes.
    int scope_idx = local_scopes_.size() - 1;
    for (auto it = local_scopes_.rbegin(); it != local_scopes_.rend();
         ++it, --scope_idx) {
      auto &scope = *it;
      Scope &local_scope = scope->NewScope();
      *scope->Var(details::kLocalExecScopeName)->GetMutable<Scope *>() =
          &local_scope;

      for (auto &info : var_infos_) {
        if (scope->FindVar(info.name_) != nullptr) {
          continue;
        }

        if (info.persistable_) {  // Persistable
          InitializeVariable(scope->Var(info.name_), info.type_);
        } else {
          InitializeVariable(local_scope.Var(info.name_), info.type_);
        }
      }
      {
        std::stringstream out;
        for (auto &var : scope->LocalVarNames()) {
          auto *var_ptr = scope->FindVar(var);
          out << var << "(" << var_ptr << "), ";
          if (var_names.count(var_ptr)) {
            VLOG(10) << "Find same var: " << var << " with"
                     << data_names[var_ptr].first << "_"
                     << data_names[var_ptr].second;
          }
          var_names.emplace(var_ptr, std::make_pair(var, scope_idx));
          if (var_ptr->IsInitialized() &&
              var_ptr->IsType<framework::LoDTensor>()) {
            auto *tensor = var_ptr->GetMutable<LoDTensor>();
            if (tensor_names.count(tensor)) {
              VLOG(10) << "Find tensor var: " << var << " with"
                       << data_names[tensor].first << "_"
                       << data_names[tensor].second;
            }
            tensor_names.emplace(tensor, std::make_pair(var, scope_idx));
            if (tensor->IsInitialized()) {
              void *data = tensor->data<void>();
              if (data_names.count(data)) {
                VLOG(10) << "Find data var: " << var << " with"
                         << data_names[data].first << "_"
                         << data_names[data].second;
              }
              data_names.emplace(data, std::make_pair(var, scope_idx));
            }
          }
        }
        VLOG(10) << out.str();
      }
      {
        std::stringstream out;
        for (auto &var : local_scope.LocalVarNames()) {
          auto *var_ptr = local_scope.FindVar(var);
          out << var << "(" << var_ptr << "), ";
          if (var_names.count(var_ptr)) {
            VLOG(10) << "Find same var: " << var << " with"
                     << data_names[var_ptr].first << "_"
                     << data_names[var_ptr].second;
          }
          var_names.emplace(var_ptr, std::make_pair(var, scope_idx));
          if (var_ptr->IsInitialized() &&
              var_ptr->IsType<framework::LoDTensor>()) {
            auto *tensor = var_ptr->GetMutable<LoDTensor>();
            if (tensor_names.count(tensor)) {
              VLOG(10) << "Find tensor var: " << var << " with"
                       << data_names[tensor].first << "_"
                       << data_names[tensor].second;
            }
            tensor_names.emplace(tensor, std::make_pair(var, scope_idx));
            if (tensor->IsInitialized()) {
              void *data = tensor->data<void>();
              if (data_names.count(data)) {
                VLOG(10) << "Find data var: " << var << " with"
                         << data_names[data].first << "_"
                         << data_names[data].second;
              }
              data_names.emplace(data, std::make_pair(var, scope_idx));
            }
          }
        }
        VLOG(10) << out.str();
      }
    }
  }
  std::vector<framework::LoDTensor> fetch_data;
  std::exception_ptr eptr = nullptr;
  try {
    fetch_data = underlying_executor_->Run(fetch_tensors);
  } catch (...) {
    eptr = std::current_exception();
  }

  platform::RecordEvent e("ScopeBufferedSSAGraphExecutorAfterRun");
  ++drop_scope_counter_;

  bool stream_end = false;
  if (!fetch_tensors.empty()) {
    WaitComputationalStreams();
    stream_end = true;
  }

  if (drop_scope_counter_ == strategy_.num_iteration_per_drop_scope_) {
    if (!stream_end) {
      WaitComputationalStreams();
    }

    for (auto &scope : local_scopes_) {
      auto &local_scope =
          *scope->Var(details::kLocalExecScopeName)->GetMutable<Scope *>();
      scope->DeleteScope(local_scope);
    }

    drop_scope_counter_ = 0;
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  } else {
    return fetch_data;
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
