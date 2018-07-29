/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <unordered_set>

#include "paddle/fluid/framework/ir/op_fusion_pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {
const char kOpDescs[] = "op_descs";
typedef std::vector<std::unique_ptr<OpDesc>> OpDescs;

std::unique_ptr<ir::Graph> OpFusionPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  graph->Set<OpDescs>(kOpDescs, new OpDescs);

  const std::unordered_set<ir::Node *> &nodes = graph->Nodes();

  std::vector<ir::Node *> topo_order;
  PADDLE_ENFORCE(GetTopoOrder(nodes, &topo_order), "");

  if (VLOG_IS_ON(10)) {
    std::stringstream out;
    for (auto &node : topo_order) {
      out << node->Op()->Type() << ", ";
    }
    VLOG(10) << out.str();
  }

  std::unordered_map<const Node *, Node *> internal_nodes;
  std::unordered_set<ir::Node *> need_removed_nodes;

  for (auto iter_node = topo_order.rbegin(); iter_node != topo_order.rend();
       iter_node++) {
    auto cur_node = *iter_node;

    std::unordered_set<Node *> tobe_fused;

    if (SetupFusion(cur_node, internal_nodes, &tobe_fused)) {
      Node *new_node =
          FuseOperators(cur_node, tobe_fused, &need_removed_nodes, graph.get());

      need_removed_nodes.insert(tobe_fused.begin(), tobe_fused.end());
      need_removed_nodes.emplace(cur_node);

      for (auto &sub_node : tobe_fused) {
        PADDLE_ENFORCE_EQ(internal_nodes.count(sub_node), 0);
        internal_nodes.emplace(sub_node, new_node);
      }
      PADDLE_ENFORCE_EQ(internal_nodes.count(cur_node), 0);
      internal_nodes.emplace(cur_node, new_node);

      PADDLE_ENFORCE(new_node->inputs.size() == 0 &&
                     new_node->outputs.size() == 0);
    }
  }

  // Release unnecessary node
  for (auto &node : need_removed_nodes) {
    graph->ReleaseNode(node);
  }
  return graph;
}

bool OpFusionPass::SetupFusion(
    const NodePtr node,
    const std::unordered_map<const Node *, InternalNodePtr> &internal_nodes,
    std::unordered_set<Node *> *tobe_fused) const {
  PADDLE_ENFORCE(!node->IsVariable(), "Node should not be variable.");

  bool need_fusion = false;

  NodePtr cur_node = node;
  if (internal_nodes.count(node)) {
    cur_node = internal_nodes.at(node);
  }

  std::vector<NodePtr> &inputs = cur_node->inputs;

  for (auto it = inputs.begin(); it != inputs.end(); ++it) {
    auto in_var = *it;
    PADDLE_ENFORCE(in_var->IsVariable(), "in_var should be a variable.");
    PADDLE_ENFORCE(in_var->inputs.size() == 1,
                   "in_var's generation op should be one.");

    auto in_var_gen_op = in_var->inputs[0];
    if (IsFusible(cur_node, in_var_gen_op)) {
      need_fusion = true;
      tobe_fused->insert(in_var_gen_op);
    }
  }
  return need_fusion;
}

bool OpFusionPass::IsFusible(const NodePtr n1, const NodePtr n2) const {
  PADDLE_ENFORCE(!n1->IsVariable(), "n1 should not be Variable.");
  PADDLE_ENFORCE(!n2->IsVariable(), "n2 should not be Variable.");

  VLOG(10) << n1->Op()->Type() << ", " << n2->Op()->Type();

  //  if (n2->outputs.size() == 1 &&
  //      n2->outputs[0]->outputs.size() == 1 && ...) {
  //    return true;
  //  }
  // TODO(zcd): hard code
  bool case1 = (n1->Op()->Type() == "scale" || n1->Op()->Type() == "relu") &&
               (n2->Op()->Type() == "elementwise_add");
  bool case2 = (n2->Op()->Type() == "scale" || n2->Op()->Type() == "relu") &&
               (n1->Op()->Type() == "elementwise_add");
  //  bool case3 =
  //    (n1->Op()->Type() == "scale_grad" || n1->Op()->Type() == "relu_grad") &&
  //    (n2->Op()->Type() == "elementwise_add_grad");
  //  bool case4 =
  //    (n2->Op()->Type() == "scale_grad" || n2->Op()->Type() == "relu_grad") &&
  //    (n1->Op()->Type() == "elementwise_add_grad");
  if (case1 || case2) {
    return true;
  }
  return false;
}

Node *OpFusionPass::FuseOperators(
    const NodePtr cur_node, const std::unordered_set<NodePtr> &tobe_fused,
    std::unordered_set<ir::Node *> *need_removed_nodes,
    ir::Graph *graph) const {
  //  Create OpDesc
  graph->Get<OpDescs>(kOpDescs).emplace_back(new framework::OpDesc());
  auto *fused_op_desc = graph->Get<OpDescs>(kOpDescs).back().get();

  if (tobe_fused.size() == 1) {
    // Init OpDesc
    if (IsElemwiseAndActivation(cur_node, tobe_fused)) {
      FuseElemwiseAndActivation(cur_node, tobe_fused, fused_op_desc);
    } else {
      PADDLE_THROW(
          "Currently, only support fusing elementwise and activation "
          "operator.");
    }
  } else {
    PADDLE_ENFORCE("Currently only support fusing two operators.");
  }

  // Create Node
  auto fused_node = graph->CreateOpNode(fused_op_desc);

  // new_node input
  for (auto &var : cur_node->inputs) {
    auto &in_var_gen_node = var->inputs[0];
    if (tobe_fused.count(in_var_gen_node)) {
      // var should be removed.
      need_removed_nodes->emplace(var);
      for (auto &in_var : in_var_gen_node->inputs) {
        PADDLE_ENFORCE(in_var->IsVariable());
        fused_node->inputs.emplace_back(in_var);

        PADDLE_ENFORCE(in_var->outputs.size() == 1);
        in_var->outputs.clear();
        in_var->outputs.emplace_back(fused_node);
      }
      //  var should be removed.
      for (auto &out_var : in_var_gen_node->outputs) {
        PADDLE_ENFORCE(out_var->IsVariable());
        need_removed_nodes->emplace(out_var);
      }
    } else {
      fused_node->inputs.emplace_back(var);

      PADDLE_ENFORCE(var->outputs.size() == 1);
      var->outputs.clear();
      var->outputs.emplace_back(fused_node);
    }
  }

  // new_node output
  for (auto &cur_output : cur_node->outputs) {
    PADDLE_ENFORCE(cur_output->IsVariable());
    fused_node->outputs.emplace_back(cur_output);

    cur_output->inputs.clear();
    cur_output->inputs.emplace_back(fused_node);
  }

  return fused_node;
}

bool OpFusionPass::IsForward(
    const NodePtr node, const std::unordered_set<Node *> &tobe_fused) const {
  auto op_role = boost::get<int>(
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));

  for (auto &node : tobe_fused) {
    PADDLE_ENFORCE_EQ(op_role, boost::get<int>(node->Op()->GetAttr(
                                   OpProtoAndCheckerMaker::OpRoleAttrName())),
                      "Currently, only support fusing the same role operators");
  }
  return static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                               OpProtoAndCheckerMaker::OpRoleAttrName())) &
                           static_cast<int>(OpRole::kForward));
}

// temporally
static bool IsActivation(std::string op_type) {
  static std::unordered_set<std::string> activations = {"relu", "scale"};
  return activations.count(op_type) == 1;
}

// temporally
static bool IsElemwise(std::string op_type) {
  static std::unordered_set<std::string> elementwise = {"elementwise_add"};
  return elementwise.count(op_type) == 1;
}

bool OpFusionPass::IsElemwiseAndActivation(
    const NodePtr node, const std::unordered_set<Node *> &tobe_fused) const {
  PADDLE_ENFORCE_EQ(tobe_fused.size(), 1);
  auto inside_op = *tobe_fused.begin();
  auto outside_op_name = node->Op()->Type();
  auto inside_op_name = inside_op->Op()->Type();

  return (IsActivation(outside_op_name) && IsElemwise(inside_op_name)) ||
         (IsActivation(inside_op_name) && IsElemwise(outside_op_name));
}

void OpFusionPass::FuseElemwiseAndActivation(
    const NodePtr node, const std::unordered_set<Node *> &tobe_fused,
    OpDesc *op_desc) const {
  auto outside_op_type = node->Op()->Type();
  auto out_argument_names = node->Op()->OutputArgumentNames();

  auto intra_node = *tobe_fused.begin();
  auto intra_op_type = intra_node->Op()->Type();

  auto fused_operator = outside_op_type + "," + intra_op_type;

  if (IsForward(node, tobe_fused)) {
    op_desc->SetType("fused_elemwise_activation");
    op_desc->SetAttr("functor_list", fused_operator);

    if (IsElemwise(outside_op_type)) {
      auto in_args = intra_node->Op()->InputArgumentNames();
      auto out_args = intra_node->Op()->OutputArgumentNames();
      PADDLE_ENFORCE_EQ(in_args.size(), 1);
      PADDLE_ENFORCE_EQ(out_args.size(), 1);

      auto cur_in_args = node->Op()->InputArgumentNames();
      PADDLE_ENFORCE_EQ(cur_in_args.size(), 2);

      op_desc->SetInput("Y", in_args);

      if (cur_in_args[0] == out_args[0]) {
        op_desc->SetInput("X", {cur_in_args[1]});
      } else if (cur_in_args[1] == out_args[0]) {
        op_desc->SetInput("X", {cur_in_args[0]});
      } else {
        PADDLE_THROW("exception");
      }
      for (auto &m_ele : intra_node->Op()->GetAttrMap()) {
        op_desc->SetAttr(m_ele.first, m_ele.second);
      }
      op_desc->SetAttr("axis", boost::get<int>(node->Op()->GetAttr("axis")));
      op_desc->SetOutput("Out", node->Op()->OutputArgumentNames());
    } else {
      op_desc->SetInput("Y", intra_node->Op()->Input("Y"));
      op_desc->SetInput("X", intra_node->Op()->Input("X"));
      op_desc->SetOutput("Out", node->Op()->OutputArgumentNames());

      for (auto &m_ele : intra_node->Op()->GetAttrMap()) {
        op_desc->SetAttr(m_ele.first, m_ele.second);
      }

      op_desc->SetAttr("axis",
                       boost::get<int>(intra_node->Op()->GetAttr("axis")));
    }
  } else {  // is backward
            //    op_desc->SetType("fused_elemwise_activation_grad");
            //
            //    std::unordered_map<std::string, Attribute> attr;
            //    attr["functor_list"] = fused_operator;
            //
            //    if (IsElemwise(outside_op_type)) {
            //      op_desc->SetInput("X", {});
            //      attr["axis"] = fused_operator;
            //
            //      op_desc->SetAttrMap({});
            //    } else {
            //      op_desc->SetInput("X", {});
            //      attr["axis"] = fused_operator;
            //      op_desc->SetAttrMap({});
            //    }
    //    op_desc->SetOutput("Out", InputGrad("X"));
  }
}

bool OpFusionPass::GetTopoOrder(const std::unordered_set<ir::Node *> &nodes,
                                std::vector<ir::Node *> *topo_order) const {
  auto topo = *topo_order;

  std::unordered_map<Node *, size_t> pending_ops;
  std::unordered_set<Node *> pending_vars;
  std::unordered_set<Node *> ready_vars;
  std::unordered_set<Node *> ready_ops;

  auto insert_var = [&](Node *var) {
    pending_vars.insert(var);
    if (var->inputs.empty()) {
      ready_vars.emplace(var);
    }
  };

  auto insert_op = [&](Node *node) {
    if (node->inputs.empty()) {
      ready_ops.insert(node);
    } else {
      pending_ops.insert({node, node->NoDupInputSize()});
    }
  };

  for (auto &node : nodes) {
    if (node->IsVariable()) {
      insert_var(node);
    } else {
      insert_op(node);
    }
  }

  auto run_all_ops = [&](std::unordered_set<Node *> &set) {
    for (auto *op : set) {
      PADDLE_ENFORCE(!op->IsVariable());
      for (auto out : op->outputs) {
        ready_vars.emplace(out);
      }
    }
    topo.insert(topo.begin(), set.begin(), set.end());
    set.clear();
  };

  while (!pending_vars.empty()) {
    run_all_ops(ready_ops);

    if (ready_vars.empty()) {
      return false;
    }

    for (auto ready_var : ready_vars) {
      pending_vars.erase(ready_var);
      for (auto *op : ready_var->outputs) {
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

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(op_fusion_pass, paddle::framework::ir::OpFusionPass);
