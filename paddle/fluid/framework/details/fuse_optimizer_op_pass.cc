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

#include "paddle/fluid/framework/details/fuse_optimizer_op_pass.h"
#include <algorithm>
#include <unordered_set>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
namespace details {

std::unique_ptr<ir::Graph> FuseOptimizerOpPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  ir::Graph &result = *graph;

  auto &places = Get<const std::vector<platform::Place>>(kPlaces);
  auto &local_scopes = Get<const std::vector<Scope *>>(kLocalScopes);

  const std::string fuse_op_type = GetOpType();
  const std::vector<std::string> aux_var_names = GetAuxiliaryVarNames();

  // Step 1: Get the specified op and auxiliary variables.
  std::vector<ir::Node *> topo_nodes = ir::TopologySortOperations(result);
  std::unordered_map<std::string, std::vector<std::string>> aux_var_set;
  std::vector<ir::Node *> opt_ops;
  for (auto &node : topo_nodes) {
    GetSpecifiedOpsAndVars(fuse_op_type, aux_var_names, node, &opt_ops,
                           &aux_var_set);
  }

  VLOG(10) << "Find " << fuse_op_type << " operators: " << opt_ops.size();
  if (opt_ops.size() == 0) {
    return std::move(graph);
  }

  if (result.Has(kFusedOptType)) {
    VLOG(10)
        << "Currently only support fusing one type optimizer op. Has fused "
        << result.Get<FusedOptType>(kFusedOptType);
    return std::move(graph);
  } else {
    result.Set(kFusedOptType, new FusedOptType);
  }
  result.Get<FusedOptType>(kFusedOptType) = fuse_op_type;

  // Step 2: Insert fused_var_name to FusedVars, and the FusedVars need be
  // initialized in scopes before execution.
  if (!result.Has(kFusedVars)) {
    result.Set(kFusedVars, new FusedVars);
  }
  std::unordered_map<std::string, std::string> fused_vars_name;
  fused_vars_name.reserve(aux_var_names.size() + 1);
  auto &fused_var_set = result.Get<FusedVars>(kFusedVars);
  const std::string prefix(kFusedVarNamePrefix);
  // NOTE: the fused_var_name should be unique.
  for (auto &var_name : aux_var_names) {
    auto fused_var_name = prefix + "_" + fuse_op_type + "_" + var_name + "_" +
                          aux_var_set[var_name][0];
    VLOG(10) << fused_var_name;
    fused_vars_name.emplace(var_name, fused_var_name);
    PADDLE_ENFORCE_EQ(fused_var_set.count(fused_var_name), 0);
    fused_var_set.insert(fused_var_name);
  }

  // Step 3: Get the fused Gradient's name
  auto &params_grads = result.Get<ParamsAndGrads>(kParamsAndGrads);
  if (!result.Has(kFusedGrads)) {
    PADDLE_THROW(
        "The alloc_continuous_space_for_grad_pass should be called before this "
        "pass.");
  }
  auto &fused_grad = result.Get<FusedGrads>(kFusedGrads);
  auto &fused_vars = result.Get<FusedVars>(kFusedVars);
  auto iter = std::find(fused_vars.begin(), fused_vars.end(), fused_grad);
  PADDLE_ENFORCE(iter != fused_vars.end(), "Not find the fused_grad.");
  fused_vars_name.emplace("Grad", fused_grad);

  // Step 4: Sort the parameters and auxiliary variables according
  // to parameters' name to make variables' name correspond correctly.
  PADDLE_ENFORCE(result.Has(kParamsAndGrads), "Does't find kParamsAndGrads.");
  PADDLE_ENFORCE_EQ(params_grads.size(), aux_var_set.begin()->second.size(),
                    "The size of params_grads and aux_var_set are not equal.");
  SortParametersAndAuxVars(params_grads, &aux_var_set, &opt_ops);

  // Step 5: Alloc continuous space for Parameters and AuxiliaryVar(e.g.
  // Moment1, Moment2, Beta1Pow, Beta2Pow) of all the optimizer ops separately.
  InitFusedVarsAndAllocSpaceForVars(places, local_scopes, aux_var_names,
                                    aux_var_set, fused_vars_name);

  // Step 6: Fuse optimizer Ops and Scale Ops
  FuseOptimizerOps(aux_var_set, fused_vars_name, opt_ops, &result);

  // Step 7: Remove optimizer Ops
  for (auto &opt_op : opt_ops) {
    graph->RemoveNode(opt_op);
  }

  return std::move(graph);
}

void FuseOptimizerOpPass::InitFusedVarsAndAllocSpaceForVars(
    const std::vector<platform::Place> &places,
    const std::vector<Scope *> &local_scopes,
    const std::vector<std::string> &aux_var_names,
    const std::unordered_map<std::string, std::vector<std::string>>
        &aux_var_set,
    const std::unordered_map<std::string, std::string> &fused_vars_name) const {
  VLOG(10) << "Init FusedVars.";
  // Alloc parameters and auxiliary vars in the respective scope.
  size_t idx = local_scopes.size();
  for (auto iter = local_scopes.rbegin(); iter != local_scopes.rend();
       ++iter, --idx) {
    auto &scope = *iter;
    for (auto &var_name : aux_var_names) {
      auto fused_var_name = fused_vars_name.at(var_name);
      VLOG(10) << "Init " << fused_var_name;
      PADDLE_ENFORCE(scope->FindVar(fused_var_name) == nullptr,
                     "%s has exist in scope[%d]", fused_var_name, idx);
      scope->Var(fused_var_name)->GetMutable<LoDTensor>();
    }
  }

  ProgramDesc program_desc;
  auto *global_block = program_desc.MutableBlock(0);
  for (auto &var_name : aux_var_names) {
    AppendAllocContinuousSpace(aux_var_set.at(var_name),
                               fused_vars_name.at(var_name), true,
                               global_block);
  }

  for (size_t i = 0; i < local_scopes.size(); ++i) {
    for (auto &op_desc : global_block->AllOps()) {
      auto op = OpRegistry::CreateOp(*op_desc);
      op->Run(*local_scopes[i], places[i]);
    }
  }
}

void FuseOptimizerOpPass::SortParametersAndAuxVars(
    const std::vector<std::pair<std::string, std::string>> &params_grads,
    std::unordered_map<std::string, std::vector<std::string>> *aux_vars_set,
    std::vector<ir::Node *> *ops) const {
  PADDLE_ENFORCE_NE(aux_vars_set->count("Param"), static_cast<size_t>(0));
  auto &param_vec = aux_vars_set->at("Param");

  std::vector<size_t> param_sort_idx;
  param_sort_idx.reserve(param_vec.size());

  for (auto &p_g : params_grads) {
    auto iter = std::find(param_vec.begin(), param_vec.end(), p_g.first);
    PADDLE_ENFORCE(iter != param_vec.end());
    auto idx = std::distance(param_vec.begin(), iter);
    param_sort_idx.emplace_back(idx);
  }

  for (auto &aux_vars : *aux_vars_set) {
    std::vector<std::string> sorted_vars;
    sorted_vars.reserve(aux_vars.second.size());
    for (size_t i = 0; i < aux_vars.second.size(); ++i) {
      sorted_vars.emplace_back(aux_vars.second.at(param_sort_idx[i]));
    }
    std::swap(aux_vars.second, sorted_vars);

    std::stringstream out;
    for (auto &var_name : aux_vars.second) {
      out << var_name << " ";
    }
    VLOG(10) << aux_vars.first << ": " << out.str();
  }

  std::vector<ir::Node *> sorted_ops;
  sorted_ops.reserve(ops->size());
  for (size_t i = 0; i < ops->size(); ++i) {
    sorted_ops.emplace_back(ops->at(param_sort_idx[i]));
  }
  std::swap(*ops, sorted_ops);
}

void FuseOptimizerOpPass::GetSpecifiedOpsAndVars(
    const std::string &op_type, const std::vector<std::string> &aux_vars_name,
    ir::Node *node, std::vector<ir::Node *> *ops,
    std::unordered_map<std::string, std::vector<std::string>> *aux_args_name)
    const {
  if (node->Op()->Type() != op_type) return;

  for (auto &var_n : aux_vars_name) {
    auto arg_names = node->Op()->Input(var_n);
    PADDLE_ENFORCE_EQ(arg_names.size(), static_cast<size_t>(1));
    (*aux_args_name)[var_n].emplace_back(arg_names[0]);
    VLOG(10) << var_n << ", " << arg_names[0];
  }
  ops->emplace_back(node);
}

void FuseOptimizerOpPass::AppendAllocContinuousSpace(
    const std::vector<std::string> &args, const std::string &out_arg,
    bool copy_data, BlockDesc *global_block) const {
  auto op_desc = global_block->AppendOp();
  op_desc->SetType("alloc_continuous_space");
  op_desc->SetInput("Input", args);
  op_desc->SetOutput("Output", args);
  op_desc->SetOutput("FusedOutput", {out_arg});
  op_desc->SetAttr("copy_data", copy_data);
  op_desc->SetAttr("check_name", true);
}

void FuseOptimizerOpPass::InserInputAndOutputForOptOps(
    const std::vector<ir::Node *> &opt_ops, ir::Node *opt_node) const {
  std::unordered_set<ir::Node *> inputs;
  std::unordered_set<ir::Node *> outputs;
  for (auto opt_op : opt_ops) {
    // set inputs
    inputs.insert(opt_op->inputs.begin(), opt_op->inputs.end());
    for (auto &input : opt_op->inputs) {
      replace(input->outputs.begin(), input->outputs.end(), opt_op, opt_node);
    }
    // set outputs
    outputs.insert(opt_op->outputs.begin(), opt_op->outputs.end());
    for (auto &output : opt_op->outputs) {
      replace(output->inputs.begin(), output->inputs.end(), opt_op, opt_node);
    }
  }
  opt_node->inputs.insert(opt_node->inputs.begin(), inputs.begin(),
                          inputs.end());
  opt_node->outputs.insert(opt_node->outputs.begin(), outputs.begin(),
                           outputs.end());
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
