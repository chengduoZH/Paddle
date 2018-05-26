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

#include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"
#include "paddle/fluid/framework/threadpool.h"
// #include "paddle/timer/Stat.h"

namespace paddle {
namespace framework {
namespace details {
ThreadedSSAGraphExecutor::ThreadedSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    std::unique_ptr<SSAGraph> &&graph)
    : SSAGraphExecutor(std::move(graph)),
      pool_(strategy.num_threads_ >= 2 ? new ::ThreadPool(strategy.num_threads_)
                                       : nullptr),
      local_scopes_(local_scopes),
      places_(places),
      fetch_ctxs_(places),
      running_ops_(0),
      strategy_(strategy),
      thread_cnt_(strategy.num_threads_) {}

void ThreadedSSAGraphExecutor::RunOp(
    std::atomic<int> *total_ops, BlockingQueue<OpHandleBase *> *ready_ops,
    std::unordered_map<OpHandleBase *, std::atomic<size_t>> *pending_op_deps,
    int dev_id) {
  bool timeout;
  std::deque<OpHandleBase *> local_ops;
  OpHandleBase *current_op = nullptr;
  //  auto name = "ThreadedSSAGraphExecutor::RunOp";
  //  auto stat = getStat(name);
  //  TimerOnce timer(stat.get(), name, 1 * 1LU);
  while (true) {
    // 1. If current_op is nullptr, get a runnable op from pending_ops.
    if (current_op == nullptr && local_ops.size() == 0) {
      if ((*total_ops) <= 0) break;
      current_op = ready_ops->Pop(1, &timeout);
      if (timeout) continue;
    }
    if (current_op == nullptr) {
      current_op = local_ops.front();
      local_ops.pop_front();
    }

    // 2. Run the current op.
    try {
      VLOG(10) << current_op << " " << current_op->Name() << " : "
               << current_op->DebugString();
      current_op->Run(strategy_.use_event_);
      --(*total_ops);
      VLOG(10) << current_op << " " << current_op->Name() << " Done ";
    } catch (platform::EnforceNotMet ex) {
      exception_.reset(new platform::EnforceNotMet(ex));
    } catch (...) {
      LOG(FATAL) << "Unknown exception catched";
    }
    auto released_vars = current_op->Outputs();

    // 3. Decrease the dependency of pending_op_deps. And find the runnable op.
    current_op = nullptr;
    for (auto ready_var : released_vars) {
      for (auto *op : ready_var->pending_ops_) {
        auto dep_num = --pending_op_deps->at(op);
        if (dep_num == 0) {
          bool push_into_ready_ops =
              current_op != nullptr ||
              (op->IsMultiDeviceTransfer() && strategy_.allow_op_delay_);
          if (push_into_ready_ops) {
            // ready_ops->Push(op);
            local_ops.emplace_back(op);
          } else {
            current_op = op;
          }
        }
      }
    }
  }
  platform::DeviceContextPool::Instance()
      .Get(platform::CUDAPlace(dev_id))
      ->Wait();
}

FeedFetchList ThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  // Step 1. Insert FetchOps
  std::vector<std::unique_ptr<FetchOpHandle>> fetch_ops;
  std::unordered_set<std::unique_ptr<VarHandleBase>> fetch_dependencies;
  FeedFetchList fetch_data(fetch_tensors.size());

  InsertFetchOps(fetch_tensors, &fetch_ops, &fetch_dependencies, &fetch_data);

  // Step 2. Collect ready_ops and pending_op_deps
  //  BlockingQueue<OpHandleBase *> ready_ops;  // read and write
  //  std::unordered_map<OpHandleBase *, std::atomic<size_t>>
  //      pending_op_deps;  // only read
  const size_t dev_cnt = places_.size();
  std::vector<BlockingQueue<OpHandleBase *>> ready_ops(dev_cnt);
  std::vector<std::unordered_map<OpHandleBase *, std::atomic<size_t>>>
      pending_op_deps(dev_cnt);

  auto get_device_id = [dev_cnt](OpHandleBase *op) -> int {
    int dev_id = -1;
    for (auto var : op->Outputs()) {
      auto var_h = dynamic_cast<VarHandle *>(var);
      if (var_h) {
        dev_id = boost::get<platform::CUDAPlace>(var_h->place_).device;
        break;
      }
    }
    if (dev_id == -1) {
      for (auto var : op->Inputs()) {
        auto var_h = dynamic_cast<VarHandle *>(var);
        if (var_h) {
          dev_id = boost::get<platform::CUDAPlace>(var_h->place_).device;
          break;
        }
      }
    }
    PADDLE_ENFORCE(dev_id >= 0 && dev_id < dev_cnt);
    return dev_id;
  };

  for (auto &op : graph_->ops_) {
    int dev_id = get_device_id(op.get());
    if (op->Inputs().empty()) {
      ready_ops[dev_id].Push(op.get());
    } else {
      pending_op_deps[dev_id][op.get()] = op->NoDupInputSize();
    }
  }
  for (auto &op : fetch_ops) {
    int dev_id = get_device_id(op.get());
    pending_op_deps[dev_id][op.get()] = op->NoDupInputSize();
  }

  // move some pending op to ready ops
  for (auto &var_map : graph_->vars_) {
    for (auto &name_pair : var_map) {
      for (auto &version_pair : name_pair.second) {
        if (version_pair->generated_op_ == nullptr) {
          for (auto pending_op : version_pair->pending_ops_) {
            int dev_id = get_device_id(pending_op);
            --pending_op_deps[dev_id][pending_op];
            if (pending_op_deps[dev_id][pending_op] == 0) {
              ready_ops[dev_id].Push(pending_op);
            }
          }
        }
      }
    }
  }

  for (auto &var : graph_->dep_vars_) {
    if (var->generated_op_ == nullptr) {
      for (auto pending_op : var->pending_ops_) {
        int dev_id = get_device_id(pending_op);
        --pending_op_deps[dev_id][pending_op];
        if (pending_op_deps[dev_id][pending_op] == 0) {
          ready_ops[dev_id].Push(pending_op);
        }
      }
    }
  }

  for (int i = 0; i < ready_ops.size(); ++i) {
    bool timeout = false;
    auto ops = ready_ops[i].PopAll(1, &timeout);
    PADDLE_ENFORCE(timeout == false);
    for (auto op : ops) {
      PADDLE_ENFORCE_EQ(i, get_device_id(op));
    }
    ready_ops[i].Extend(ops);
  }

  for (int i = 0; i < pending_op_deps.size(); ++i) {
    for (auto iter_op = pending_op_deps[i].begin();
         iter_op != pending_op_deps[i].end(); iter_op++) {
      PADDLE_ENFORCE_EQ(i, get_device_id((*iter_op).first));
    }
  }

  // according to total_ops to know whether the loop is over
  std::atomic<int> total_ops(
      static_cast<int>(graph_->ops_.size() + fetch_ops.size()));

  // Step 3. Execution
  std::vector<std::thread> workers;
  workers.resize(dev_cnt);
  for (size_t i = 0; i < dev_cnt; ++i) {
    workers[i] = std::thread(
        [&total_ops, &ready_ops, &pending_op_deps, i, dev_cnt, this] {
          RunOp(&total_ops, &ready_ops[i], &pending_op_deps[i], i);
        });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  PADDLE_ENFORCE(total_ops <= 0);

  // Wait FetchOps.
  if (!fetch_ops.empty()) {
    fetch_ops.clear();
  }

  return fetch_data;
}

void ThreadedSSAGraphExecutor::InsertFetchOps(
    const std::vector<std::string> &fetch_tensors,
    std::vector<std::unique_ptr<FetchOpHandle>> *fetch_ops,
    std::unordered_set<std::unique_ptr<VarHandleBase>> *fetch_dependencies,
    FeedFetchList *fetch_data) {
  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;

  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &var_map : graph_->vars_) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        fetched_vars[fetch_var_name].push_back(it->second.rbegin()->get());
      }
    }
  }

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors[i];
    auto &vars = fetched_vars.at(var_name);
    auto *op = new FetchOpHandle(fetch_data, i, &local_scopes_);
    fetch_ops->emplace_back(op);

    for (auto &p : places_) {
      op->SetDeviceContext(p, fetch_ctxs_.Get(p));
    }

    for (auto *var : vars) {
      op->AddInput(var);
    }

    auto *fetch_dummy = new DummyVarHandle();
    op->AddOutput(fetch_dummy);
    fetch_dependencies->emplace(fetch_dummy);
  }
}

void ThreadedSSAGraphExecutor::InsertPendingOp(
    std::unordered_map<OpHandleBase *, size_t> *pending_ops,
    OpHandleBase *op_instance) const {
  pending_ops->insert({op_instance, op_instance->NoDupInputSize()});
}

void ThreadedSSAGraphExecutor::InsertPendingVar(
    std::unordered_set<VarHandleBase *> *pending_vars,
    BlockingQueue<VarHandleBase *> *ready_vars, VarHandleBase *var) const {
  pending_vars->insert(var);
  if (var->generated_op_ == nullptr) {
    ready_vars->Push(var);
  }
}
void ThreadedSSAGraphExecutor::RunOp(
    BlockingQueue<VarHandleBase *> *ready_var_q, details::OpHandleBase *op) {
  auto op_run = [ready_var_q, op, this] {
    try {
      VLOG(10) << op << " " << op->Name() << " : " << op->DebugString();
      op->Run(strategy_.use_event_);
      VLOG(10) << op << " " << op->Name() << " Done ";
      running_ops_--;
      ready_var_q->Extend(op->Outputs());
      VLOG(10) << op << " " << op->Name() << "Signal posted";
    } catch (platform::EnforceNotMet ex) {
      exception_.reset(new platform::EnforceNotMet(ex));
    } catch (...) {
      LOG(FATAL) << "Unknown exception catched";
    }
  };
  if (pool_) {
    pool_->enqueue(op_run);
  } else {
    op_run();
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
