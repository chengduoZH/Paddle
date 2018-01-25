#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
fluid.default_startup_program().random_seed = 111

batch = [128]
data_shape = [3, 224, 224]
label_shape = [1]

images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
conv2d = fluid.layers.conv2d(
    input=images, num_filters=3, filter_size=3, use_cudnn=False)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

for i in range(1):
    print("iteration : %d" % i)
    x = np.random.random(batch + data_shape).astype("float32")
    tensor_x = core.LoDTensor()
    tensor_x.set(x, place)
    exe.run(fluid.default_main_program(), feed={'pixel': tensor_x})
