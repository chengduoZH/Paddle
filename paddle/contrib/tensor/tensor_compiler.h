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

#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/time.h"
#include "tc/utils/compiler_options.h"

namespace tc {
namespace pten {

std::vector<tc::DLTensorUPtr> inferOutputTensorInfo(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<paddle::framework::Tensor>& inputs);

std::vector<paddle::framework::Tensor> prepareOutputs(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<paddle::framework::Tensor>& inputs);

template <typename Backend>
std::unique_ptr<typename Backend::ExecutorType> compile(
    const std::string& tc,
    const std::string& entryPoint,
    const std::vector<paddle::framework::Tensor>& inputs,
    const typename Backend::MappingOptionsType& options,
    const CompilerOptions& compilerOptions = CompilerOptions());

template <typename Executor>
void run(const Executor& executor,
         const std::vector<paddle::framework::Tensor>& inputs,
         std::vector<paddle::framework::Tensor>& outputs);

template <typename Executor>
ProfilingInfo profile(const Executor& executor,
                      const std::vector<paddle::framework::Tensor>& inputs,
                      std::vector<paddle::framework::Tensor>& outputs);

template <typename Executor>
void uncheckedRun(const Executor& executor,
                  const std::vector<paddle::framework::Tensor>& inputs,
                  std::vector<paddle::framework::Tensor>& outputs);
}  // namespace pten
}  // namespace tc

#include "paddle/contrib/tensor/tensor_compiler_impl.h"
