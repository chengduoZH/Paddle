# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest

import numpy as np
import paddle.fluid.core as core
import os
import paddle.fluid as fluid

from paddle.fluid import compiler


def simple_fc_net():
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    fc_result = fluid.layers.fc(
        img,
        size=200,
        act='tanh',
        param_attr=fluid.ParamAttr(name='weight'),
        bias_attr=fluid.ParamAttr(
            name='bias', initializer=fluid.initializer.Constant(value=1.0)))
    return fc_result


def init_data():
    np.random.seed(5)
    img = np.random.random(size=[32, 784]).astype(np.float32)
    return img


class TestMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def fun(self, model, use_cuda, img, fuse_parameter=False):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = 1  # Fix random seed
        main.random_seed = 1
        with fluid.program_guard(main, startup):
            fc_output = model()

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)

        exec_strategy = fluid.ExecutionStrategy()
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = False
        build_strategy.fuse_parameters_pass = fuse_parameter

        mult_graph = compiler.CompiledProgram(main).with_data_parallel(
            build_strategy=build_strategy, exec_strategy=exec_strategy)

        res = exe.run(mult_graph,
                      feed={"image": img},
                      fetch_list=[fc_output.name])
        return res

    def test_simple_fc(self):
        img = init_data()
        for use_cuda in [True, False]:
            if use_cuda and not core.is_compiled_with_cuda():
                continue
            output = self.fun(simple_fc_net,
                              use_cuda,
                              img,
                              fuse_parameter=False)
            output2 = self.fun(simple_fc_net,
                               use_cuda,
                               img,
                               fuse_parameter=True)

            for loss in zip(output, output2):
                self.assertAlmostEqual(loss[0], loss[1], delta=1e-6)


if __name__ == '__main__':
    unittest.main()
