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

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace platform {
class NCCLContextMap;
}

namespace framework {
class Scope;
namespace details {

class MultiDevSSAGraphBuilderBase : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

  void CreateOpHandleIOs(ir::Graph *result, ir::Node *node,
                         size_t device_id) const;

  virtual void Init() const;

  virtual bool IsDistTrain(const std::vector<ir::Node *> &ops) const;

  virtual void Prepare() const;

  virtual std::vector<ir::Node *> SortOperations(const ir::Graph &graph) const;

  int GetVarDeviceID(const std::string &varname) const;

  bool IsScaleLossOp(ir::Node *node) const;

  int CreateRPCOp(ir::Graph *result, ir::Node *node) const;

  int CreateDistTrainOp(ir::Graph *result, ir::Node *node) const;

  void CreateComputationalOps(ir::Graph *result, ir::Node *node,
                              size_t num_places) const;

  void CreateScaleLossGradOp(ir::Graph *result,
                             const std::string &loss_grad_name,
                             ir::Node *out_var_node,
                             proto::VarType::Type dtype) const;

  VarHandle *CreateReduceOp(ir::Graph *result, const std::string &og,
                            int dst_dev_id) const;

  void CreateComputationalOp(ir::Graph *result, ir::Node *node,
                             int dev_id) const;

  void InsertAllReduceOp(ir::Graph *result, const std::string &og) const;

  void InsertDataBalanceOp(ir::Graph *result,
                           const std::vector<std::string> &datas) const;

  void CreateBroadcastOp(ir::Graph *result, const std::string &p_name,
                         size_t src_dev_id) const;

  virtual void CreateCollectionOp(ir::Graph *result, bool is_dist_train,
                                  const std::string &p_name,
                                  const std::string &g_name) const = 0;

  virtual bool PreProcess(ir::Graph *result, ir::Node *node) { return false; }
  virtual bool IsPreProcessNode(ir::Node *node) const { return false; }
  int GetOpDeviceID(ir::Node *node) const;

  void CreateFusedBroadcastOp(
      ir::Graph *result,
      const std::vector<std::unordered_set<std::string>> &bcast_varnames) const;

  size_t GetAppropriateDeviceID(
      const std::vector<std::string> &var_names) const;

  void SetCommunicationContext(OpHandleBase *op_handle,
                               const platform::Place &p) const;

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  mutable platform::NCCLContextMap *nccl_ctxs_;
#endif

  mutable std::string loss_var_name_;
  mutable std::vector<platform::Place> places_;
  mutable std::vector<Scope *> local_scopes_;

  mutable BuildStrategy strategy_;
  mutable std::unordered_map<std::string, VarDesc *> all_vars_;

  mutable std::vector<int64_t> balance_vars_;
  mutable std::vector<std::unordered_set<std::string>> bcast_var_name_set_;
  mutable std::unordered_map<std::string, int> sharded_var_device_;
};

class AllReduceSSAGraphBuilder : public MultiDevSSAGraphBuilderBase {
 protected:
  virtual void CreateCollectionOp(ir::Graph *result, bool is_dist_train,
                                  const std::string &p_name,
                                  const std::string &g_name) const;

  bool IsSparseGradient(const std::string &og) const;

  //  virtual bool IsDistTrain(const std::vector<ir::Node *> &ops) const {
  //    return false;
  //  }
};

class ReduceSSAGraphBuilder : public MultiDevSSAGraphBuilderBase {
 protected:
  virtual void Init() const {
    MultiDevSSAGraphBuilderBase::Init();
    sharded_var_device_.clear();
  }

  virtual void Prepare() const {
    MultiDevSSAGraphBuilderBase::Prepare();
    sharded_var_device_.clear();
  }

  virtual void CreateCollectionOp(ir::Graph *result, bool is_dist_train,
                                  const std::string &p_name,
                                  const std::string &g_name) const;

  int GetOpDeviceID(ir::Node *node,
                    std::unordered_map<std::string, std::vector<ir::Node *>>
                        *delay_ops) const;

  virtual bool IsPreProcessNode(ir::Node *node) const {
    bool flag = MultiDevSSAGraphBuilderBase::IsPreProcessNode(node);
    flag = flag || (MultiDevSSAGraphBuilderBase::GetOpDeviceID(node) != -1);
    return flag;
  }

  virtual bool PreProcess(ir::Graph *result, ir::Node *node) const;

  virtual std::vector<ir::Node *> SortOperations(const ir::Graph &graph) const;

  std::vector<ir::Node *> SortForReduceMode(
      const std::vector<ir::Node *> &topo_ops) const;

  //  virtual bool IsDistTrain(const std::vector<ir::Node *> &ops) const {
  //    return false;
  //  }

  //  mutable std::unordered_map<std::string, int> sharded_var_device_;
};

class DistSSAGraphBuilder : public MultiDevSSAGraphBuilderBase {
 protected:
  virtual bool PreProcess(ir::Graph *result, ir::Node *node) const;
  virtual bool IsPreProcessNode(ir::Node *node) const;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
