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

from __future__ import print_function

import paddle.fluid as fluid
import paddle
import unittest
import numpy

from paddle.fluid.layers.control_flow import lod_rank_table
from paddle.fluid.layers.control_flow import max_sequence_len
from paddle.fluid.layers.control_flow import lod_tensor_to_array
from paddle.fluid.layers.control_flow import array_to_lod_tensor
from paddle.fluid.layers.control_flow import shrink_memory


class TestDynRNN(unittest.TestCase):
    def setUp(self):
        self.word_dict = paddle.dataset.imdb.word_dict()
        self.BATCH_SIZE = 2
        self.train_data = paddle.batch(
            paddle.dataset.imdb.train(self.word_dict),
            batch_size=self.BATCH_SIZE)

    def test_train_dyn_rnn(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=1)
            sent_emb = fluid.layers.embedding(
                input=sentence, size=[len(self.word_dict), 32], dtype='float32')

            rnn = fluid.layers.DynamicRNN()

            with rnn.block():
                in_ = rnn.step_input(sent_emb)
                mem = rnn.memory(shape=[100], dtype='float32')
                out_ = fluid.layers.fc(input=[in_, mem], size=100, act='tanh')

                rnn2 = fluid.layers.DynamicRNN()
                with rnn2.block():
                    in2_ = rnn.step_input(in_)
                    mem2 = rnn.memory(shape=[100], dtype='float32')
                    out2_ = fluid.layers.fc(input=[in2_, mem2],
                                            size=100,
                                            act='tanh')
                    rnn.update_memory(mem2, out2_)
                    rnn.output(out2_)

                rnn.update_memory(mem, out_)
                out2_ = rnn2()
                out2_ = fluid.layers.sum(x=[out_, out2_])
                rnn.output(out2_)

            last = fluid.layers.sequence_last_step(input=rnn())
            logits = fluid.layers.fc(input=last, size=1, act=None)
            label = fluid.layers.data(name='label', shape=[1], dtype='float32')
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=logits, label=label)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.Adam(1e-3)
            sgd.minimize(loss=loss)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(startup_program)
        feeder = fluid.DataFeeder(feed_list=[sentence, label], place=cpu)
        data = next(self.train_data())
        loss_0 = exe.run(main_program,
                         feed=feeder.feed(data),
                         fetch_list=[loss])[0]
        for _ in range(100):
            val = exe.run(main_program,
                          feed=feeder.feed(data),
                          fetch_list=[loss])[0]
        # loss should be small after 100 mini-batch
        self.assertLess(val[0], loss_0[0])


if __name__ == '__main__':
    unittest.main()
