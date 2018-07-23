#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest

# scale + add
#   TestElementwiseAddOp
#   TestFusedOperatorsOp_scalar
#   TestFusedOperatorsOp_scalar2
#   TestFusedOperatorsOp_Vector
#   TestFusedOperatorsOp_broadcast_0
#   TestFusedOperatorsOp_broadcast_1
#   TestFusedOperatorsOp_broadcast_2
#   TestFusedOperatorsOp_broadcast_3
#   TestFusedOperatorsOp_broadcast_4
#   TestFusedOperatorsOp_rowwise_add_0
#   TestFusedOperatorsOp_rowwise_add_1
#   TestFusedOperatorsOp_channelwise_add


class TestElementwiseAddOp(OpTest):
    def setUp(self):
        self.op_type = "fusedoperators"
        self.dtype = np.float32
        self.axis = -1

        self.init_axis()
        self.init_dtype()
        self.init_input()
        self.init_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.outputs = {'Out': self.out}

    def init_input(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }

    def init_dtype(self):
        pass

    def init_axis(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.005)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))


class TestFusedOperatorsOp_scalar(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_scalar2(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_Vector(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.random((32, )).astype(self.dtype)
        self.y = np.random.random((32, )).astype(self.dtype)

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_broadcast_0(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(2).astype(self.dtype)

    def init_axis(self):
        self.axis = 0

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y.reshape(2, 1, 1)) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_broadcast_1(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(3).astype(self.dtype)

    def init_axis(self):
        self.axis = 1

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y.reshape(1, 3, 1)) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_broadcast_2(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(4).astype(self.dtype)

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y.reshape(1, 1, 4)) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_broadcast_3(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)

    def init_axis(self):
        self.axis = 1

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y.reshape(1, 3, 4, 1)) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_broadcast_4(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(2, 1).astype(self.dtype)

    def init_axis(self):
        self.axis = 0

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y.reshape(2, 1, 1, 1)) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_rowwise_add_0(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)

    def init_axis(self):
        self.axis = 1

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y.reshape(1, 3, 4)) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_rowwise_add_1(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(2, 1).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)

    def init_axis(self):
        self.axis = 1

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y.reshape(1, 1)) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


class TestFusedOperatorsOp_channelwise_add(TestElementwiseAddOp):
    def init_input(self):
        self.x = np.random.rand(3, 20, 20).astype(self.dtype)
        self.y = np.random.rand(3, 1, 1).astype(self.dtype)

    def init_axis(self):
        self.axis = -1

    def init_output(self):
        self.scale = 0.1
        self.out = (self.x + self.y) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["scale," + str(self.scale), "add"]
        }


# add + scale
#   TestElementwiseAddOp_f_add_scale
#   TestFusedOperatorsOp_scalar_f_add_scale
#   TestFusedOperatorsOp_scalar2_f_add_scale
#   TestFusedOperatorsOp_Vector_f_add_scale
#   TestFusedOperatorsOp_broadcast_0_f_add_scale
#   TestFusedOperatorsOp_broadcast_1_f_add_scale
#   TestFusedOperatorsOp_broadcast_2_f_add_scale
#   TestFusedOperatorsOp_broadcast_3_f_add_scale
#   TestFusedOperatorsOp_broadcast_4_f_add_scale
#   TestFusedOperatorsOp_rowwise_add_0_f_add_scale
#   TestFusedOperatorsOp_rowwise_add_1_f_add_scale
#   TestFusedOperatorsOp_channelwise_add_f_add_scale


class TestFusedOperatorsOp_f_add_scale(TestElementwiseAddOp):
    def init_output(self):
        self.scale = 0.1
        self.out = self.x + self.y * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_scalar_f_add_scale(TestFusedOperatorsOp_scalar):
    def init_output(self):
        self.scale = 0.1
        self.out = self.x + self.y * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_scalar2_f_add_scale(TestFusedOperatorsOp_scalar2):
    def init_output(self):
        self.scale = 0.1
        self.out = self.x + self.y * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_Vector_f_add_scale(TestFusedOperatorsOp_Vector):
    def init_output(self):
        self.scale = 0.1
        self.out = self.x + self.y * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_broadcast_0_f_add_scale(
        TestFusedOperatorsOp_broadcast_0):
    def init_axis(self):
        self.axis = 0

    def init_output(self):
        self.scale = 0.1
        self.out = self.x + self.y.reshape(2, 1, 1) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_broadcast_1_f_add_scale(
        TestFusedOperatorsOp_broadcast_1):
    def init_axis(self):
        self.axis = 1

    def init_output(self):
        self.scale = 0.1
        self.out = self.x + self.y.reshape(1, 3, 1) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_broadcast_2_f_add_scale(
        TestFusedOperatorsOp_broadcast_2):
    def init_output(self):
        self.scale = 0.1
        self.out = self.x + self.y.reshape(1, 1, 4) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_broadcast_3_f_add_scale(
        TestFusedOperatorsOp_broadcast_3):
    def init_axis(self):
        self.axis = 1

    def init_output(self):
        self.scale = 0.1
        self.out = self.x + self.y.reshape(1, 3, 4, 1) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_broadcast_4_f_add_scale(
        TestFusedOperatorsOp_broadcast_4):
    def init_axis(self):
        self.axis = 0

    def init_output(self):
        self.scale = 0.2
        self.out = self.x + self.y.reshape(2, 1, 1, 1) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_rowwise_add_0_f_add_scale(
        TestFusedOperatorsOp_rowwise_add_0):
    def init_axis(self):
        self.axis = 1

    def init_output(self):
        self.scale = 0.1
        self.out = self.x + self.y.reshape(1, 3, 4) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_rowwise_add_1_f_add_scale(
        TestFusedOperatorsOp_rowwise_add_1):
    def init_axis(self):
        self.axis = 1

    def init_output(self):
        self.scale = 0.2
        self.out = self.x + self.y.reshape(1, 1) * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


class TestFusedOperatorsOp_channelwise_add_f_add_scale(
        TestFusedOperatorsOp_channelwise_add):
    def init_axis(self):
        self.axis = -1

    def init_output(self):
        self.scale = 0.2
        self.out = self.x + self.y * self.scale
        self.attrs = {
            'axis': self.axis,
            'functor_list': ["add", "scale," + str(self.scale)]
        }


if __name__ == '__main__':
    unittest.main()
