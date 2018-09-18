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

#include "paddle/fluid/framework/ir/fuse_elewise_add_act_pass.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> FuseElewiseAddActPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  std::unordered_set<std::string> act_types = {"relu", "scale"};
  graph = FuseActElewiseAdd(std::move(graph), act_types);
  graph = FuseElewiseAddAct(std::move(graph), act_types);
  //  {
  //    std::unordered_set<std::string> in_place_act_types = {"relu_grad"};
  //    // std::unordered_set<std::string> no_in_place_act_types =
  //    {"scale_grad"};
  //    //  graph = FuseActElewiseAdd(std::move(graph), act_types);
  //    graph = FuseElewiseAddActGrad1(std::move(graph), in_place_act_types);
  //  }

  // Remove the removable intermediate_out.
  RemoveIntermediateOut(graph.get());

  return graph;
}

void FuseElewiseAddActPass::RemoveIntermediateOut(Graph *graph) const {
  std::unordered_set<const Node *> need_removed_nodes;
  for (auto upstream_node : graph->Nodes()) {
    if (upstream_node->IsVar()) continue;
    if (upstream_node->Name() == "fused_elemwise_activation") {
      bool save_intermediate_out = boost::get<bool>(
          upstream_node->Op()->GetAttr("save_intermediate_out"));
      auto intermediate_out_args =
          upstream_node->Op()->Output("IntermediateOut");
      PADDLE_ENFORCE(
          save_intermediate_out && !intermediate_out_args.empty(),
          "The %s should save the intermediate_out in the fusing stage.",
          upstream_node->Name());

      // If the intermediate_out's output is only
      // fused_elemwise_activation_grad, but the fused_elemwise_activation_grad
      // doesn't use the intermediate_out.
      auto upstream_node_outputs = upstream_node->outputs;
      for (auto out : upstream_node_outputs) {
        if (out->Name() == intermediate_out_args[0]) {
          if (out->outputs.size() == 0) {
            upstream_node->outputs =
                this->RemoveNode(out, upstream_node->outputs);
            need_removed_nodes.insert(std::move(out));
            upstream_node->Op()->SetAttr("save_intermediate_out", false);
          }
        }
      }
    } else if (upstream_node->Name() == "fused_elemwise_activation_grad") {
      auto intermediate_out_grad_args =
          upstream_node->Op()->Output("IntermediateOut@GRAD");
      PADDLE_ENFORCE(
          !intermediate_out_grad_args.empty(),
          "The %s should save the intermediate_out in the fusing stage.",
          upstream_node->Name());
      auto upstream_node_outputs = upstream_node->outputs;
      for (auto &out : upstream_node_outputs) {
        if (out->Name() == intermediate_out_grad_args[0] &&
            out->outputs.empty()) {
          upstream_node->Op()->SetOutput("IntermediateOut@GRAD", {});
          upstream_node->outputs =
              this->RemoveNode(out, upstream_node->outputs);
          need_removed_nodes.insert(std::move(out));
        }
      }
    }
  }
  GraphSafeRemoveNodes(graph, need_removed_nodes);
}

// f1(f2(x,y))
std::unique_ptr<ir::Graph> FuseElewiseAddActPass::FuseElewiseAddAct(
    std::unique_ptr<ir::Graph> graph,
    const std::unordered_set<std::string> &act_types) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("ealewise_add_act", graph.get());

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("ealewise_add_act/x")
                ->AsInput()
                ->assert_is_op_input("elementwise_add", "X");
  patterns::ElewiseAddAct elewise_add_act_pattern(gpd.mutable_pattern(),
                                                  "elementwise_add");

  elewise_add_act_pattern(x, act_types);

  int found_elewise_add_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FuseElewiseAddAct fuse";
    GET_IR_NODE_FROM_SUBGRAPH(ele_y, ele_y, elewise_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_out, elewise_add_out,
                              elewise_add_act_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, elewise_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, act, elewise_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add, ele_add, elewise_add_act_pattern);

    std::string ele_x_n = subgraph.at(x)->Name();
    std::string ele_y_n = ele_y->Name();
    std::string ele_out_n = ele_out->Name();
    std::string act_out_n = act_out->Name();

    Node *elewise_add_act_node = CreateFuseElewiseAddActNode(
        g, act, ele_add, ele_x_n, ele_y_n, ele_out_n, act_out_n);

    VLOG(4) << "\n\t " << ele_x_n << " and " << ele_y_n << " -> "
            << ele_add->Name() << " -> " << ele_out_n << "\n"
            << "\t " << ele_out_n << " -> " << act->Name() << " -> "
            << act_out_n;

    ReLinkNodes(g, ele_out, ele_add, act, elewise_add_act_node);
    found_elewise_add_act_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_elewise_add_act_count);
  return graph;
}

std::unique_ptr<ir::Graph> FuseElewiseAddActPass::FuseActElewiseAdd(
    std::unique_ptr<ir::Graph> graph,
    const std::unordered_set<std::string> &act_types) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("act_elewise_add", graph.get());

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("act_elewise_add/x")
                ->AsInput()
                ->assert_is_ops_input(act_types, "X");
  patterns::ActElewiseAdd act_elewise_add_pattern(gpd.mutable_pattern(),
                                                  "act_elewise_add");

  act_elewise_add_pattern(x, act_types);

  int found_elewise_add_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FuseElewiseAddAct fuse";

    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, act_elewise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_x, x, act_elewise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_out, elewise_add_out,
                              act_elewise_add_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(act, act, act_elewise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add, ele_add, act_elewise_add_pattern);

    OpDesc desc;
    std::string act_i_n = subgraph.at(x)->Name();
    std::string act_o_n = act_out->Name();
    std::string elewise_add_x_n = ele_x->Name();
    std::string elewise_add_out_n = ele_out->Name();

    Node *elewise_add_act_node = CreateFuseElewiseAddActNode(
        g, ele_add, act, elewise_add_x_n, act_i_n, act_o_n, elewise_add_out_n);

    VLOG(4) << "\t " << act_i_n << " -> " << act->Name() << " -> " << act_o_n
            << "\n"
            << "\t " << act_o_n << " and " << elewise_add_x_n << " -> "
            << ele_add->Name() << " -> " << elewise_add_out_n;

    ReLinkNodes(g, act_out, act, ele_add, elewise_add_act_node);
    found_elewise_add_act_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_elewise_add_act_count);
  return graph;
}

// f1(f2(x,y)), act is inplace
std::unique_ptr<ir::Graph> FuseElewiseAddActPass::FuseElewiseAddActGrad1(
    std::unique_ptr<ir::Graph> graph,
    const std::unordered_set<std::string> &act_types) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("ealewise_add_act_grad", graph.get());

  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // ele_add_grad: in["Y", "Out@GRAD"], out["X@GRAD", "Y@GRAD"]
  GraphPatternDetector gpd;
  auto *d_act_out = gpd.mutable_pattern()
                        ->NewNode("ealewise_add_act_grad_inplace/x")
                        ->AsInput()
                        ->assert_is_ops_input(act_types, "Out@GRAD");
  patterns::ElewiseAddActGrad1 elewise_add_act_grad_pattern(
      gpd.mutable_pattern(), "elewise_add_act_grad_inplace");
  elewise_add_act_grad_pattern(d_act_out, act_types);

  int found_elewise_add_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FuseElewiseAddActGrad1 fuse";

    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_grad, act_grad, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_itermediate_out, d_itermediate_out,
                              elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_y, ele_y, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_grad, ele_add_grad,
                              elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_ele_x, d_ele_x, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_ele_y, d_ele_y, elewise_add_act_grad_pattern);

    std::string d_act_out_n = subgraph.at(d_act_out)->Name();
    std::string act_out_n = act_out->Name();
    std::string d_itermediate_out_n = d_itermediate_out->Name();
    std::string ele_y_n = ele_y->Name();
    std::string d_ele_x_n = d_ele_x->Name();
    std::string d_ele_y_n = d_ele_y->Name();

    OpDesc desc;
    desc.SetType("fused_elemwise_activation_grad");
    desc.SetInput("IntermediateOut", {});
    desc.SetInput("X", {});
    desc.SetInput("Y", std::vector<std::string>({ele_y_n}));
    desc.SetInput("Out", std::vector<std::string>({act_out_n}));
    desc.SetInput("Out@GRAD", std::vector<std::string>({d_act_out_n}));
    desc.SetOutput("X@GRAD", std::vector<std::string>({d_ele_x_n}));
    desc.SetOutput("Y@GRAD", std::vector<std::string>({d_ele_y_n}));
    desc.SetOutput("IntermediateOut@GRAD",
                   std::vector<std::string>({d_itermediate_out_n}));

    desc.SetAttr("save_intermediate_out", false);
    desc.SetAttr("functor_list",
                 std::vector<std::string>(
                     {act_grad->Op()->Type(), ele_add_grad->Op()->Type()}));

    for (auto &n : {act_grad->Op(), ele_add_grad->Op()}) {
      for (auto &m_ele : n->GetAttrMap()) {
        desc.SetAttr(m_ele.first, m_ele.second);
      }
    }

    auto fused_node = g->CreateOpNode(&desc);

    VLOG(4) << "\n\t " << d_act_out_n << " and " << act_out_n << " -> "
            << act_grad->Name() << " -> " << d_itermediate_out_n << "\n\t "
            << d_itermediate_out_n << " and " << act_out_n << " -> "
            << ele_add_grad->Name() << " -> " << d_itermediate_out_n;

    for (auto in : act_grad->inputs) {
      fused_node->inputs.emplace_back(in);
      in->outputs = this->ReplaceNode(act_grad, fused_node, in->outputs);
    }

    std::unordered_set<const Node *> nodes2delete;
    for (auto out : act_grad->outputs) {
      if (out->IsCtrlVar()) {
        auto result_iter = std::find_if(
            ele_add_grad->inputs.begin(), ele_add_grad->inputs.end(),
            [&out](const Node *node) -> bool { return node == out; });

        if (result_iter == ele_add_grad->inputs.end()) {
          IR_OP_VAR_LINK(fused_node, out);
        } else {
          nodes2delete.emplace(out);
        }
      } else {
        PADDLE_ENFORCE(out == d_itermediate_out);
        IR_OP_VAR_LINK(fused_node, out);
      }
    }

    for (auto in : ele_add_grad->inputs) {
      if (in == d_itermediate_out || nodes2delete.count(in)) {
        continue;
      }
      fused_node->inputs.emplace_back(in);
      in->outputs = this->ReplaceNode(ele_add_grad, fused_node, in->outputs);
    }

    for (auto out : ele_add_grad->outputs) {
      IR_OP_VAR_LINK(fused_node, out);
    }

    nodes2delete.insert(std::move(act_grad));
    nodes2delete.insert(std::move(ele_add_grad));

    GraphSafeRemoveNodes(g, nodes2delete);
    found_elewise_add_act_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_elewise_add_act_count);
  return graph;
}

Node *FuseElewiseAddActPass::CreateFuseElewiseAddActNode(
    Graph *g, const Node *op_1, const Node *op_2, const std::string &ele_x_n,
    const std::string &ele_y_n, const std::string &ele_out_n,
    const std::string &act_out_n) const {
  OpDesc desc;
  desc.SetInput("X", std::vector<std::string>({ele_x_n}));
  desc.SetInput("Y", std::vector<std::string>({ele_y_n}));
  desc.SetOutput("Out", std::vector<std::string>({act_out_n}));
  desc.SetOutput("IntermediateOut", std::vector<std::string>({ele_out_n}));
  desc.SetType("fused_elemwise_activation");
  desc.SetAttr("save_intermediate_out", true);
  desc.SetAttr("functor_list", std::vector<std::string>(
                                   {op_1->Op()->Type(), op_2->Op()->Type()}));

  // Set attrs
  for (auto &n : {op_1->Op(), op_2->Op()}) {
    for (auto &m_ele : n->GetAttrMap()) {
      desc.SetAttr(m_ele.first, m_ele.second);
    }
  }

  auto elewise_add_act_node = g->CreateOpNode(&desc);
  return elewise_add_act_node;
}

// f1(x,f2(y))
void FuseElewiseAddActPass::ReLinkNodes(Graph *graph,
                                        const Node *intermediate_out,
                                        Node *op_1, Node *op_2,
                                        Node *fused_op) const {  // delete act
  for (auto in : op_1->inputs) {
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_1, fused_op, in->outputs);
  }

  std::unordered_set<const Node *> nodes2delete;
  for (auto out : op_1->outputs) {
    if (out->IsCtrlVar()) {
      auto result_iter = std::find_if(
          op_2->inputs.begin(), op_2->inputs.end(),
          [&out](const Node *node) -> bool { return node == out; });

      if (result_iter == op_2->inputs.end()) {
        IR_OP_VAR_LINK(fused_op, out);
      } else {
        nodes2delete.emplace(out);
      }
    } else {
      PADDLE_ENFORCE(out == intermediate_out);
      IR_OP_VAR_LINK(fused_op, out);
    }
  }

  for (auto in : op_2->inputs) {
    if (in == intermediate_out || nodes2delete.count(in)) {
      continue;
    }
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_2, fused_op, in->outputs);
  }

  for (auto out : op_2->outputs) {
    IR_OP_VAR_LINK(fused_op, out);
  }

  nodes2delete.insert(std::move(op_1));
  nodes2delete.insert(std::move(op_2));

  GraphSafeRemoveNodes(graph, nodes2delete);
}

std::vector<Node *> FuseElewiseAddActPass::ReplaceNode(
    Node *cur_node, Node *new_node, const std::vector<Node *> &nodes) const {
  std::vector<Node *> new_list(nodes.size());
  bool has_replaced = false;
  std::transform(nodes.begin(), nodes.end(), new_list.begin(),
                 [&](Node *node) -> Node * {
                   if (node == cur_node) {
                     has_replaced = true;
                     return new_node;
                   }
                   return node;
                 });
  PADDLE_ENFORCE(has_replaced, "Not find %s in the node list.",
                 cur_node->Name());
  return new_list;
}

std::vector<Node *> FuseElewiseAddActPass::RemoveNode(
    Node *trg_node, const std::vector<Node *> &nodes) const {
  std::vector<Node *> new_list(nodes.size());
  auto end_iter =
      std::copy_if(nodes.begin(), nodes.end(), new_list.begin(),
                   [&](Node *node) -> bool { return node != trg_node; });
  new_list.resize(
      static_cast<uint64_t>(std::distance(new_list.begin(), end_iter)));
  return new_list;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_elewise_add_act_pass,
              paddle::framework::ir::FuseElewiseAddActPass);
