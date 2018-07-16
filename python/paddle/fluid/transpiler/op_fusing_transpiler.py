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

from paddle.fluid import framework as framework
import os
import numpy as np
from .. import core
from ..framework import Program
from ..executor import global_scope
import collections
import copy


class OpFusionTranspiler(object):
    '''
    Convert the fluid program to optimized fused program.

    There are several optimizations:

      - fuse convolution and batch normalization
      - fuse batch normalization and relu (MKLDNN only)

    '''

    def transpile(self, program):
        '''
        '''
        if not isinstance(program, Program):
            raise TypeError("program should be as Program type")

        var_op = collections.defaultdict(list)
        op_role_vars_ops = dict()

        for idx in range(len(program.block(0).ops)):
            op = program.block(0).ops[idx]
            assert isinstance(op, framework.Operator)
            # Get the relationship between output and op
            for out_name in op.output_names:
                for out_var in op.output(out_name):
                    if var_op.has_key(out_var):
                        if var_op[out_var][-1] == op:
                            raise TypeError("program has duplicate op.")
                        else:
                            var_op[out_var].append(op)
                    else:
                        var_op[out_var] = [op]

            # Get all the momemtum op
            op_role = op.attr('op_role')

            # Collect opt ops
            if op_role == 2L:
                op_role_vars = tuple(op.attr('op_role_var'))
                if op_role_vars_ops.has_key(op_role_vars):
                    op_role_vars_ops[op_role_vars].append((op, idx))
                else:
                    op_role_vars_ops[op_role_vars] = [(op, idx)]

        delete_op_idx = []
        for op_role_vars_op in op_role_vars_ops:
            assert len(op_role_vars_ops[op_role_vars_op]) == 3
            assert op_role_vars_ops[op_role_vars_op][0][0].type == "scale"
            assert op_role_vars_ops[op_role_vars_op][1][
                0].type == "elementwise_add"
            assert op_role_vars_ops[op_role_vars_op][2][0].type == "momentum"
            decay_factor = op_role_vars_ops[op_role_vars_op][0][0].attr('scale')
            op_role_vars_ops[op_role_vars_op][2][0].set_attr('decay',
                                                             decay_factor)
            # it doesn't need to rename input here.
            idx1 = op_role_vars_ops[op_role_vars_op][0][1]
            idx2 = op_role_vars_ops[op_role_vars_op][1][1]
            delete_op_idx.append(idx1)
            delete_op_idx.append(idx2)

        delete_num = 0
        delete_op_idx = np.sort(delete_op_idx)
        for idx in range(len(delete_op_idx)):
            op_idx = delete_op_idx[idx] - delete_num
            program.block(0).remove_op(op_idx)
            delete_num += 1

    def fuse_elementwise_add_and_relu(self, program):

        if not isinstance(program, Program):
            raise TypeError("program should be as Program type")

        elementwise_add_ops = collections.defaultdict(list)
        relu_ops = collections.defaultdict(list)

        delete_op_idx = []
        idx = 0
        while idx < len(program.block(0).ops):
            op = program.block(0).ops[idx]
            assert isinstance(op, framework.Operator)

            # Collect elementwiseadd
            if op.type == "elementwise_add":
                if elementwise_add_ops.has_key(op.output_arg_names[0]):
                    raise TypeError("program has duplicate op.")
                else:
                    elementwise_add_ops[op.output_arg_names[
                        0]] = [op, op.input_arg_names, idx]
            if op.type == "relu":
                if relu_ops.has_key(op.input_arg_names[0]):
                    raise TypeError("program has duplicate op.")
                else:
                    relu_ops[op.input_arg_names[
                        0]] = [op, op.output_arg_names, idx]
                    if elementwise_add_ops.has_key(op.input_arg_names[0]):
                        assert elementwise_add_ops[op.input_arg_names[0]][
                            2] == idx - 1

                        ele_op_idx = elementwise_add_ops[op.input_arg_names[0]][
                            2]
                        ele_op = elementwise_add_ops[op.input_arg_names[0]][0]
                        input = elementwise_add_ops[op.input_arg_names[0]][1]
                        output = op.output_arg_names[0]

                        x_var = program.block(0).var(input[0])
                        y_var = program.block(0).var(input[1])
                        out_var = program.block(0).var(output)

                        axis = ele_op.attr("axis")
                        op_role = op.attr('op_role')
                        op_role_var = op.attr('op_role_var')

                        program.block(0).remove_op(ele_op_idx)
                        # program.block(0).remove_op(idx)
                        delete_op_idx.append(idx)

                        program.block(0).insert_op(
                            ele_op_idx,
                            type="fused_elementwise_add_relu",
                            inputs={"X": x_var,
                                    "Y": y_var},
                            outputs={"Out": out_var},
                            attrs={
                                "axis": int(axis),
                                "op_role": op_role,
                                "op_role_var": op_role_var,
                            })
                        # continue
            idx += 1

        delete_num = 0
        delete_op_idx = np.sort(delete_op_idx)
        for idx in range(len(delete_op_idx)):
            op_idx = delete_op_idx[idx] - delete_num
            program.block(0).remove_op(op_idx)
            delete_num += 1

        # Remove Grad
        elementwise_add_grad_ops = collections.defaultdict(list)
        relu_grad_ops = collections.defaultdict(list)

        delete_op_idx = []
        idx = len(program.block(0).ops) - 1
        while idx >= 0:
            op = program.block(0).ops[idx]
            assert isinstance(op, framework.Operator)

            # Collect elementwiseadd
            if op.type == "relu_grad":
                if relu_grad_ops.has_key(op.output_arg_names[0]):
                    raise TypeError("program has duplicate op.")
                else:
                    relu_grad_ops[op.output_arg_names[
                        0]] = [op.input_arg_names, idx]
                    if elementwise_add_grad_ops.has_key(op.output_arg_names[0]):

                        ele_grad_op_idx = elementwise_add_grad_ops[
                            op.output_arg_names[0]][-1]
                        ele_grad_op = program.block(0).ops[ele_grad_op_idx]
                        ele_grad_op.input_arg_names[1] = op.input_arg_names[1]
                        ele_grad_op.type = "fuse_elementwise_add_relu_grad"
                        program.block(0).remove_op(idx)

            if op.type == "elementwise_add_grad":
                if elementwise_add_grad_ops.has_key(op.input_arg_names[1]):
                    raise TypeError("program has duplicate op.")
                else:
                    elementwise_add_grad_ops[op.input_arg_names[
                        1]] = [op.input_arg_names, op.output_arg_names, idx]

                    if relu_grad_ops.has_key(op.input_arg_names[1]):

                        relu_grad_op_idx = relu_grad_ops[op.input_arg_names[1]][
                            -1]
                        input = relu_grad_ops[op.input_arg_names[1]][0][1]
                        op.input_arg_names[1] = input
                        op.type = "fuse_elementwise_add_relu_grad"
                        program.block(0).remove_op(relu_grad_op_idx)

            idx -= 1
