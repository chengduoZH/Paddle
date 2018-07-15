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
        for op in program.block(0).ops:
            assert isinstance(op, framework.Operator)
            # for in_name in op.input_names():
            #     for in_var in op.input(in_name):
            #         if not var_op.has_key(in_var):
            #             var_op[in_var] = op
            #         else:
            #             assert var_op[in_var] == op

            for out_name in op.output_names:
                for out_var in op.output(out_name):
                    if var_op.has_key(out_var):
                        if var_op[out_var][-1] == op:
                            assert var_op[out_var][-1] == op
                        else:
                            var_op[out_var].append(op)
                    else:
                        var_op[out_var] = [op]

        for op in reversed(program.block[0].ops):
            print op
