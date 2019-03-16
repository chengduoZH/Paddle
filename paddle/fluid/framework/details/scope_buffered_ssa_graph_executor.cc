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
  if (drop_scope_counter_ == 0) {
    // Create local scopes.
    for (auto it = local_scopes_.rbegin(); it != local_scopes_.rend(); ++it) {
      auto &scope = *it;
      Scope &local_scope = scope->NewScope();
      *scope->Var(details::kLocalExecScopeName)->GetMutable<Scope *>() =
          &local_scope;

      std::stringstream out2;
      int64_t begin = 0, end = var_infos_.size() / 4 * 2;
      int64_t i = -1;
      for (auto &info : var_infos_) {
        ++i;
        if (scope->FindVar(info.name_) != nullptr) {
          continue;
        }

        if (info.persistable_ || (i >= begin && i < end)) {  //
          //        Persistable
          //        if (info.persistable_) {
          if (VLOG_IS_ON(10)) {
            out2 << info.name_ << ",";
          }
          InitializeVariable(scope->Var(info.name_), info.type_);
        } else {
          InitializeVariable(local_scope.Var(info.name_), info.type_);
        }
      }
      if (VLOG_IS_ON(10)) {
        {
          auto vars = local_scope.LocalVarNames();
          create_vars_[scope].clear();
          create_vars_[scope].insert(vars.begin(), vars.end());
          std::stringstream out;
          for (auto var : vars) {
            out << var << ", ";
          }
          VLOG(10) << "create local scope of " << scope << ", " << vars.size()
                   << "  :" << out.str() << "";
        }
        VLOG(10) << "create persistable_ for " << scope << ", " << out2.str()
                 << "";
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

  for (auto &scope : local_scopes_) {
    auto &local_scope =
        *scope->Var(details::kLocalExecScopeName)->GetMutable<Scope *>();
    local_scope->DropKids();
  }

  if (drop_scope_counter_ == strategy_.num_iteration_per_drop_scope_) {
    if (!stream_end) {
      WaitComputationalStreams();
    }

    for (auto &scope : local_scopes_) {
      auto &local_scope =
          *scope->Var(details::kLocalExecScopeName)->GetMutable<Scope *>();
      {
        std::stringstream out;
        std::stringstream out2;
        std::unordered_set<std::string> vars2;
        auto vars = local_scope->LocalVarNames();
        for (auto var : vars) {
          out << var << ", ";
          if (create_vars_[scope].count(var) == 0) {
            out2 << var << ", ";
            vars2.insert(var);
          }
        }
        VLOG(10) << "delete local scope of " << scope << ", " << vars.size()
                 << ", " << out.str() << "";
        VLOG(10) << "not in local scope " << scope << ", " << vars2.size()
                 << ", " << out2.str() << "";
      }
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
