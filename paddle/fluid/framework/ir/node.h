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

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
namespace ir {

class Node {
 public:
  enum class Type { kOperation, kVariable };
  static constexpr char kControlDepVarName[] = "__control_var";

  explicit Node(const std::string& name, Type type, int id = -1)
      : name_(name),
        var_desc_(nullptr),
        op_desc_(nullptr),
        type_(type),
        id_(id) {}

  explicit Node(VarDesc* var_desc, int id = -1)
      : name_(var_desc->Name()),
        var_desc_(new VarDesc(*var_desc)),
        op_desc_(nullptr),
        type_(Type::kVariable),
        id_(id) {}

  explicit Node(OpDesc* op_desc, int id = -1)
      : name_(op_desc->Type()),
        var_desc_(nullptr),
        op_desc_(new OpDesc(*op_desc, op_desc->Block())),
        type_(Type::kOperation),
        id_(id) {}

  Type NodeType() const { return type_; }

  std::string Name() const { return name_; }

  size_t NoDupInputSize() const {
    std::unordered_set<Node*> res;
    for (auto* var : inputs) {
      res.emplace(var);
    }
    return res.size();
  }

  VarDesc* Var() {
    PADDLE_ENFORCE(type_ == Type::kVariable);
    return var_desc_.get();
  }

  OpDesc* Op() {
    PADDLE_ENFORCE(IsOp());
    return op_desc_.get();
  }

  int id() const { return id_; }

  bool IsOp() const { return type_ == Type::kOperation; }
  bool IsVar() const { return type_ == Type::kVariable; }

  std::vector<Node*> inputs;
  std::vector<Node*> outputs;

 protected:
  const std::string name_;
  std::unique_ptr<VarDesc> var_desc_;
  std::unique_ptr<OpDesc> op_desc_;
  Type type_;
  int id_;

 private:
  DISABLE_COPY_AND_ASSIGN(Node);
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
