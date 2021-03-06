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

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

// Returns true if the two dimensions are compatible.
// A dimension is compatible with the other if:
// 1. The length of the dimensions are same.
// 2. Each non-negative number of the two dimentions are same.
// 3. For negative number in a dimention, it means unknown so it is compatible
//    with any number.
bool DimensionIsCompatibleWith(const framework::DDim& first,
                               const framework::DDim& second) {
  int dim_size = first.size();
  if (dim_size != second.size()) {
    return false;
  }
  for (int i = 0; i < dim_size; ++i) {
    if (first[i] >= 0 && second[i] >= 0 && first[i] != second[i]) {
      return false;
    }
  }
  return true;
}

class ReadInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Reader"),
                   "The ReadOp must take a reader as input.");
    PADDLE_ENFORCE(ctx->HasOutputs("Out"),
                   "The ReadOp should be assigned with output.");
    if (!ctx->IsRuntime() && ctx->Attrs().Get<bool>("infer_out")) {
      std::vector<framework::DDim> reader_dims = ctx->GetReaderDims("Reader");
      std::vector<std::string> out_names = ctx->Outputs("Out");
      PADDLE_ENFORCE_EQ(
          reader_dims.size(), out_names.size(),
          "The reader's dim number doesn't match the output number.");
      ctx->SetOutputsDim("Out", reader_dims);
      auto in_desc =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("Reader")[0]);
      auto in_lod_levels = in_desc->GetLoDLevels();
      auto out_var_ptrs = ctx->GetOutputVarPtrs("Out");
      PADDLE_ENFORCE_EQ(in_lod_levels.size(), out_var_ptrs.size(),
                        "LoDLevels of Input(Reader) must be the same as the "
                        "number of Outputs(Out).");
      for (size_t i = 0; i < out_var_ptrs.size(); ++i) {
        auto* out_desc = boost::get<framework::VarDesc*>(out_var_ptrs[i]);
        out_desc->SetLoDLevel(in_lod_levels[i]);
      }
    }
  }
};

class ReadInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    bool infer_out = boost::get<bool>(ctx->GetAttr("infer_out"));
    if (infer_out) {
      std::string reader_name = ctx->Input("Reader")[0];
      std::vector<std::string> out_names = ctx->Output("Out");
      auto dtypes = ctx->GetDataTypes(reader_name);
      PADDLE_ENFORCE_EQ(dtypes.size(), out_names.size());
      for (size_t i = 0; i < dtypes.size(); ++i) {
        ctx->SetType(out_names[i], framework::proto::VarType::LOD_TENSOR);
        ctx->SetDataType(out_names[i], dtypes[i]);
      }
    }
  }
};

class ReadOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    VLOG(3) << "read op in";
    framework::ReaderHolder* reader =
        detail::Ref(scope.FindVar(Input("Reader")),
                    "Cannot find reader variable %s", Input("Reader"))
            .GetMutable<framework::ReaderHolder>();
    std::vector<std::string> out_arg_names = Outputs("Out");
    std::vector<framework::LoDTensor> ins;

    // For profiling
    platform::RecordEvent record_event(Type());

    reader->ReadNext(&ins);
    if (ins.empty()) {
      VLOG(3) << "throw_eof_exp";
      PADDLE_THROW_EOF();
    }
    PADDLE_ENFORCE_EQ(ins.size(), out_arg_names.size(),
                      "input size and output size of read_op do not match");

    const std::vector<framework::DDim>& shapes = reader->Shapes();
    const std::vector<framework::proto::VarType::Type>& var_types =
        reader->VarTypes();
    const std::vector<bool>& need_check_feed = reader->NeedCheckFeed();
    PADDLE_ENFORCE_EQ(out_arg_names.size(), need_check_feed.size(),
                      "output size of read_op and the number of feeded "
                      "variables of reader do not match");

    for (size_t i = 0; i < out_arg_names.size(); ++i) {
      auto* out =
          scope.FindVar(out_arg_names[i])->GetMutable<framework::LoDTensor>();
      if (need_check_feed[i]) {
        auto in_dims = ins[i].dims();
        PADDLE_ENFORCE_EQ(DimensionIsCompatibleWith(shapes[i], in_dims), true,
                          "The feeded Variable %s should have dimensions = %d, "
                          "shape = [%s], but received feeded shape [%s]",
                          out_arg_names[i], shapes[i].size(), shapes[i],
                          in_dims);
        PADDLE_ENFORCE_EQ(
            ins[i].type(), var_types[i],
            "The data type of feeded Variable %s must be %s, but received %s",
            out_arg_names[i], var_types[i], ins[i].type());
      }
      out->ShareDataWith(ins[i]);
      out->set_lod(ins[i].lod());
    }
  }
};

class ReadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Reader", "(ReaderHolder) The executed reader.");
    AddOutput("Out", "(LoDTensor) The output data.").AsDuplicable();
    AddAttr<bool>(
        "throw_eof_exp",
        "If set true, an exception will be thrown when the Reader "
        "yields empty (which means there is no next data).\n"
        "NOTES: This flag must be true always. It will be set to false"
        " only when the data-balance is enabled in ParallelExecutor"
        " and it is set by ParallelExecutor instance, not users.")
        .SetDefault(true);
    AddAttr<bool>("infer_out", "").SetDefault(true);
    AddComment(R"DOC(
      Read Operator

      Execute a given reader once and output data.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(read, ops::ReadOp, ops::ReadInferShape, ops::ReadOpMaker,
                  paddle::framework::EmptyGradOpMaker, ops::ReadInferVarType);
