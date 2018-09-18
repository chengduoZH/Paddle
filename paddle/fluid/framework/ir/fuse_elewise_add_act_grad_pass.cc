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

#include "paddle/fluid/framework/ir/fuse_elewise_add_act_grad_pass.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> FuseElewiseAddActGradPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  std::unordered_set<std::string> in_place_act_types = {"relu_grad"};
  std::unordered_set<std::string> no_in_place_act_types = {"scale_grad"};
  //  graph = FuseActElewiseAdd(std::move(graph), act_types);
  graph = FuseElewiseAddActGrad1(std::move(graph), in_place_act_types);
  return graph;
}

// f1(f2(x,y)), act is inplace
std::unique_ptr<ir::Graph> FuseElewiseAddActGradPass::FuseElewiseAddActGrad1(
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
  elewise_add_act_grad_pattern(x, act_types);

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
    op_desc->SetInput("IntermediateOut", {});
    op_desc->SetInput("X", {});
    op_desc->SetInput("Y", std::vector<std::string>({ele_y_n}));
    op_desc->SetInput("Out", std::vector<std::string>({act_out_n}));
    op_desc->SetInput("Out@GRAD", std::vector<std::string>({d_act_out_n}));
    op_desc->SetOutput("X@GRAD", std::vector<std::string>({d_ele_x_n}));
    op_desc->SetOutput("Y@GRAD", std::vector<std::string>({d_ele_y_n}));
    op_desc->SetOutput("IntermediateOut@GRAD",
                       std::vector<std::string>({d_itermediate_out_n}));

    op_desc->SetAttr("save_intermediate_out", false);
    op_desc->SetAttr("functor_list",
                     std::vector<std::string>(
                         {act_grad->Op()->Type(), ele_add_grad->Op()->Type()}));

    for (auto &n : {act_grad->Op(), ele_add_grad->Op()}) {
      for (auto &m_ele : n->GetAttrMap()) {
        desc.SetAttr(m_ele.first, m_ele.second);
      }
    }

    auto elewise_add_act_grad_node = g->CreateOpNode(&desc);

    VLOG(4) << "\n\t " << d_act_out_n << " and " << act_out_n << " -> "
            << act_grad->Name() << " -> " << d_itermediate_out_n << "\n\t "
            << d_itermediate_out_n << " and " << act_out_n << " -> "
            << act_grad->Name() << " -> " << d_itermediate_out_n;

    for (auto in : act_grad->inputs) {
      elewise_add_act_grad_node->inputs.emplace_back(in);
      in->outputs = this->ReplaceNode(op_1, fused_op, in->outputs);
    }

    std::unordered_set<const Node *> nodes2delete;
    for (auto out : act_grad->outputs) {
      if (out->IsCtrlVar()) {
        auto result_iter = std::find_if(
            ele_add_grad->inputs.begin(), ele_add_grad->inputs.end(),
            [&out](const Node *node) -> bool { return node == out; });

        if (result_iter == ele_add_grad->inputs.end()) {
          IR_OP_VAR_LINK(fused_op, out);
        } else {
          nodes2delete.emplace(out);
        }
      } else {
        PADDLE_ENFORCE(out == d_itermediate_out);
        IR_OP_VAR_LINK(fused_op, out);
      }
    }

    for (auto in : ele_add_grad->inputs) {
      if (in == d_itermediate_out || nodes2delete.count(in)) {
        continue;
      }
      fused_op->inputs.emplace_back(in);
      in->outputs = this->ReplaceNode(ele_add_grad, fused_op, in->outputs);
    }

    for (auto out : ele_add_grad->outputs) {
      IR_OP_VAR_LINK(fused_op, out);
    }

    found_elewise_add_act_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_elewise_add_act_count);
  return graph;
}

std::vector<Node *> FuseElewiseAddActGradPass::ReplaceNode(
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

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_elewise_add_act_pass,
              paddle::framework::ir::FuseElewiseAddActGradPass);
