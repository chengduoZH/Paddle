#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid as fluid

x1 = fluid.layers.data(
    name='x1',
    shape=[1000, 1000, 1000],
    dtype='float32',
    append_batch_size=False)
x2 = fluid.layers.data(
    name='x2',
    shape=[1000, 1000, 1000],
    dtype='float32',
    append_batch_size=False)
y = fluid.layers.concat(input=[x1, x2], axis=0)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

x = np.random.random((100, 1000, 1000)).astype('float32')
y = np.random.random((100, 1000, 1000)).astype('float32')
tensor_x = fluid.core.LoDTensor()
tensor_x.set(x, place)
tensor_y = fluid.core.LoDTensor()
tensor_y.set(y, place)

PASS_NUM = 100
with fluid.profiler.profiler("GPU", 'total', "/temp/Paddle") as prof:
    for pass_id in range(PASS_NUM):

        exe.run(fluid.default_main_program(),
                feed={"x1": tensor_x,
                      "x2": tensor_y})
