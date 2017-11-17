import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid.layers as nn
from paddle.v2.fluid.framework import Program
from paddle.v2.fluid.executor import Executor

startup_program = Program()
main_program = Program()

place = core.CPUPlace()

# define net
# input = nn.data(name='input',
#                 shape=[3,5,5],
#                 data_type='float32',
#                 main_program = main_program,
#                 startup_program=startup_program )

# conv = nn.conv2d(
#     input=input,
#     filter_size=[3,3],
#     num_filters=2,
#     stride=[1,1],
#     padding=[0,0],
#     bias_attr=False,
#     main_program = main_program,
#     startup_program=startup_program )

input = nn.data(
    'input', [3, 5, 5],
    main_program=main_program,
    startup_program=startup_program)

conv = nn.conv2d(
    input,
    2,
    None, [3, 3],
    None,
    1, [1, 1], [0, 0],
    False,
    main_program=main_program,
    startup_program=startup_program)

# prepare data
x_data = np.array(np.random.random((2, 3, 5, 5))).astype("float32")
tensor_x = core.LoDTensor()
tensor_x.set(x_data, place)

# prepare execution
exe = Executor(place)
exe.run(startup_program)

# execution
outs = exe.run(main_program, feed={'input': tensor_x}, fetch_list=[conv])

print np.array(outs[0])
