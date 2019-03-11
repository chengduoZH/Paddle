# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# Define Input for model
image = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype="float32")
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

label = fluid.layers.cast(x=label, dtype='int32')
label = fluid.layers.cast(x=label, dtype='int64')

# Define the model
conv1 = fluid.layers.conv2d(input=image, filter_size=5, num_filters=20)
relu1 = fluid.layers.relu(conv1)
pool1 = fluid.layers.pool2d(input=relu1, pool_size=2, pool_stride=2)
conv2 = fluid.layers.conv2d(input=pool1, filter_size=5, num_filters=50)
relu2 = fluid.layers.relu(conv2)
pool2 = fluid.layers.pool2d(input=relu2, pool_size=2, pool_stride=2)

predict = fluid.layers.fc(input=pool2, size=10, act='softmax')

# Get the loss
cost = fluid.layers.cross_entropy(input=predict, label=label)

avg_cost = fluid.layers.mean(x=cost)

opt = fluid.optimizer.AdamOptimizer()

# Add operations(backward operators) to minimize avg_loss 
opt.minimize(avg_cost)

place = fluid.CPUPlace()  # fluid.CUDAPlace(0)    
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

batch_acc = fluid.layers.accuracy(input=predict, label=label)

import numpy as np

train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=128)

for epoch_id in range(5):
    for batch_id, data in enumerate(train_reader()):
        img_data = np.array(
            [x[0].reshape([1, 28, 28]) for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).reshape(
            [len(img_data), 1]).astype("int64")
        loss, acc = exe.run(fluid.default_main_program(),
                            feed={'pixel': img_data,
                                  'label': y_data},
                            fetch_list=[avg_cost, batch_acc])
        print("epoch: %d, batch = %d, Loss = %f, Accuracy = %f" %
              (epoch_id, batch_id, loss, acc))

# from paddle.fluid import compiler
#
# compiled_program = compiler.CompiledProgram(fluid.default_main_program())
# compiled_program.with_data_parallel(loss_name=avg_cost.name)
#
# for epoch_id in range(5):
#     for batch_id, data in enumerate(train_reader()):
#         img_data = np.array([x[0].reshape([1, 28, 28]) for x in data]).astype('float32')
#         y_data = np.array([x[1] for x in data]).reshape([len(img_data), 1]).astype("int64")
#         loss, acc = exe.run(compiled_program,
#                             feed={'pixel': img_data, 'label': y_data},
#                             fetch_list=[avg_cost, batch_acc])
#         print("epoch: %d, batch = %d, Loss = %s, Accuracy = %s" %
#               (epoch_id, batch_id, loss, acc))
