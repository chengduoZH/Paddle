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

import paddle
import paddle.fluid as fluid
import numpy as np
import paddle.fluid.profiler as profiler
import time
import contextlib


@contextlib.contextmanager
def profile_context(profile=True):
    if profile:
        # for CPU profile, replace 'All' with 'CPU'.
        with profiler.profiler('All', 'total', '/tmp/profile_file'):
            yield
    else:
        yield


batch_size = 64 * 64
seq_fea = 512


def train(use_cuda, test_eigen=False):
    main_program = fluid.Program()
    startup = fluid.Program()
    with fluid.program_guard(main_program, startup):
        x = fluid.layers.data(name='x', shape=[-1, seq_fea], dtype='float32')
        y = fluid.layers.selu_eigen(x=x) if test_eigen else fluid.layers.selu(
            x=x)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program):
        PASS_NUM = 100
        # Warm up
        for _ in range(10):
            exe.run(main_program,
                    feed={
                        'x': np.random.random(
                            (batch_size, seq_fea)).astype("float32")
                    })

        with profile_context(profile=True):
            for _ in range(PASS_NUM):
                exe.run(main_program,
                        feed={
                            'x': np.random.random(
                                (batch_size, seq_fea)).astype("float32")
                        })

    train_loop(main_program)


train(use_cuda=True)
train(use_cuda=True, test_eigen=True)
