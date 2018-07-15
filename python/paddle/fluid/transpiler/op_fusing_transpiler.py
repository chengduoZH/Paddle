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

        # get var-op relation

        var_op = collections.defaultdict(list)
        momentum_ops = collections.defaultdict(list)
        op_role_ops = collections.defaultdict(list)
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
            if op_role_ops.has_key(op_role):
                op_role_ops[op_role].append(op)
            else:
                op_role_ops[op_role] = [op]

            if op_role == 2L:
                op_role_vars = tuple(op.attr('op_role_var'))
                if op_role_vars_ops.has_key(op_role_vars):
                    op_role_vars_ops[op_role_vars].append((op, idx))
                else:
                    op_role_vars_ops[op_role_vars] = [(op, idx)]

            if op.type == 'momentum':
                if momentum_ops.has_key('momentum'):
                    momentum_ops['momentum'].append(op)
                else:
                    momentum_ops['momentum'] = [op]

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

            # scale_op = op_role_vars_ops[op_role_vars_op][0][0]
            elementwise_add_op = op_role_vars_ops[op_role_vars_op][1][0]
            # momentum_op = op_role_vars_ops[op_role_vars_op][2][0]
            # for p in momentum_op.input_names():
            #     p_arg_names = momentum_op.inputput(p)
            #     if var_name in p_arg_names:
            #         op_desc.set_input(p, [
            #             new_name if x == var_name else x
            #             for x in p_arg_names
            #         ])

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
