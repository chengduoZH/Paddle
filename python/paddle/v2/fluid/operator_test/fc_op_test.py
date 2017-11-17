import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.layers as nn
from paddle.v2.fluid.framework import Program
from paddle.v2.fluid.executor import Executor

startup_program = Program()
main_program = Program()

place = core.CPUPlace()

x = nn.data(
    'x', [13], main_program=main_program, startup_program=startup_program)

y_predict = nn.fc(x,
                  1,
                  main_program=main_program,
                  startup_program=startup_program)

exe = Executor(place)
exe.run(startup_program)

x_data = np.array(np.random.random((1, 13))).astype("float32")
tensor_x = core.LoDTensor()
tensor_x.set(x_data, place)

outs = exe.run(main_program, feed={'x': tensor_x}, fetch_list=[y_predict])

print np.array(outs[0])
