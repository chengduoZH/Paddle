// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/platform/enforce.h"
namespace paddle {
namespace framework {
namespace ir {

TEST(GraphTest, TestFuseSGDOps) {
  // void Test() {
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sgd");
  op->SetInput("Param", {"a"});
  op->SetInput("Grad", {"b"});
  op->SetInput("LearningRate", {"c"});
  op->SetOutput("ParamOut", {"a"});

  op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sgd");
  op->SetInput("Param", {"a"});
  op->SetInput("Grad", {"e"});
  op->SetInput("LearningRate", {"c"});
  op->SetOutput("ParamOut", {"a"});

  prog.MutableBlock(0)->Var("a")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("b")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("d")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("e")->SetType(proto::VarType::LOD_TENSOR);

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("fuse_sgd_op_pass");

  try {
    graph.reset(pass->Apply(graph.release()));
  } catch (paddle::platform::EnforceNotMet err) {
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fuse_sgd_op_pass);
