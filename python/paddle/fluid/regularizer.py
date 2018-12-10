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
import contextlib
import numpy as np
from . import framework
from . import core

__all__ = [
    'L1Decay',
    'L2Decay',
    'L1DecayRegularizer',
    'L2DecayRegularizer',
    'WeightDecay',
]


def append_regularization_ops(parameters_and_grads, regularization=None):
    """Create and add backward regularization Operators

    Creates and adds backward regularization operators in the BlockDesc.
    This will add gradients of the regularizer function to the gradients
    of the parameters and return these modified gradients. This is the
    same as implementing weight decay in optimizers for regularization.

    Args:
        parameters_and_grads: A list of (parameters, gradients) pairs
                              that need to be regularized.
        regularization: A global regularizer. If the parameter is not
                        set. It will be applied with regularizer.

    Returns:
        list[(Variable, Variable)]: list of (parameters, gradients) \
        pair with the regularized gradient

    Raises:
        Exception: Unknown regularization type
    """
    params_and_grads = []
    for param, grad in parameters_and_grads:
        # If no gradient then we don't need to do anything
        if grad is None:
            params_and_grads.append((param, grad))
            continue
        with param.block.program._optimized_guard(
            [param, grad]), framework.name_scope('regularization'):
            regularization_term = None
            if param.regularizer is not None:
                # Add variable for regularization term in grad block
                regularization_term = param.regularizer(param, grad, grad.block)
            elif regularization is not None:
                regularization_term = regularization(param, grad, grad.block)

            # If no regularization specified, then we don't need to do anything
            if regularization_term is None:
                params_and_grads.append((param, grad))
                continue

            new_grad = grad
            if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
                # FIXME(zcd): If the grad is SELECTED_ROWS, after regularization,
                # the grad's type and name will be changed. But the gradient's name
                # is used in ParallelExecutor Reduce mode, so I add a flag for
                # the new_grad here.
                new_grad = grad.block.create_var(
                    name=grad.name + core.kNewGradSuffix(),
                    dtype=param.dtype,
                    shape=param.shape,
                    lod_level=param.lod_level,
                    type=core.VarDesc.VarType.LOD_TENSOR)

            grad.block.append_op(
                type='sum',
                inputs={"X": [grad, regularization_term]},
                outputs={"Out": new_grad})

            params_and_grads.append((param, new_grad))

    return params_and_grads


@contextlib.contextmanager
def append_weight_decay(param_and_grads, weight_decay=None):
    """Add decay for the weight.

    Appends weight decay operators in the BlockDesc.
    This will update the optimized parameters by using the
    parameters before optimization.

    Args:
        param_and_grads: A list of (parameters, gradients) pairs
            that need to be updated.
        weight_decay(WeightDecayBase): A WeightDecay Objection, such as
            fluid.weight_decay.WeightDecay.

    Raises:
        Exception: Unknown weight decay type.
    """

    if weight_decay is not None:
        weight_decay(param_and_grads)
    yield
    if weight_decay is not None:
        weight_decay.apply()


class WeightDecayRegularizer(object):
    """Base class for weight decay regularizers

    Defines the common interface of weight-decay regularizers.
    Weight-decay regularizers are added only during the backward
    pass for faster regularization. They add operations to the network
    that correspond to gradient of the regularization function.
    Users should not use this class directly, but need to use one
    of its implementations
    """

    def __init__(self):
        pass

    def __call__(self, param, grad, block):
        """Add corresponding weight decay operations to the network
        """
        raise NotImplementedError()

    def __str__(self):
        """Debug string
        """
        raise NotImplementedError()


class L2DecayRegularizer(WeightDecayRegularizer):
    """Implements the L2 Weight Decay Regularization

    Small values of L2 can help prevent over fitting the training data.

    .. math::

        L2WeightDecay = reg\_coeff * parameter

    Args:
        regularization_coeff(float): regularization coeff

    Examples:
        .. code-block:: python

            optimizer = fluid.optimizer.Adagrad(
                learning_rate=1e-4,
                regularization=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.1))
            optimizer.minimize(avg_cost)
    """

    def __init__(self, regularization_coeff=0.0):
        assert regularization_coeff is not None
        super(L2DecayRegularizer, self).__init__()
        self._regularization_coeff = regularization_coeff

    def __call__(self, param, grad, block):
        """Add L2 weight decay ops to network

        Adds L2 weight decay ops.
        L2WeightDecay = reg_coeff * parameter

        Args:
            param: parameter variable for which regularization is applied
            block: block in which variable is to be created

        Returns:
            new variable for weight decay
        """
        assert isinstance(param, framework.Parameter)
        assert isinstance(block, framework.Block)

        decay = block.create_var(
            dtype=param.dtype, shape=param.shape, lod_level=param.lod_level)

        # Append Op to calculate decay
        block.append_op(
            type='scale',
            inputs={"X": param},
            outputs={"Out": decay},
            attrs={"scale": self._regularization_coeff})

        return decay

    def __str__(self):
        return "L2Decay, regularization_coeff=%f" % self._regularization_coeff


class L1DecayRegularizer(WeightDecayRegularizer):
    """Implements the L1 Weight Decay Regularization

    L1 regularization encourages sparsity.

    .. math::

        L1WeightDecay = reg\_coeff * sign(parameter)

    Args:
        regularization_coeff(float): regularization coeff

    Examples:
        .. code-block:: python

            optimizer = fluid.optimizer.Adagrad(
                learning_rate=1e-4,
                regularization=fluid.regularizer.L1DecayRegularizer(
                    regularization_coeff=0.1))
            optimizer.minimize(avg_cost)
    """

    def __init__(self, regularization_coeff=0.0):
        assert regularization_coeff is not None
        super(L1DecayRegularizer, self).__init__()
        self._regularization_coeff = regularization_coeff

    def __call__(self, param, grad, block):
        """Add L1 weight decay ops to network

        Adds L1 weight decay ops.
        L1WeightDecay = reg_coeff * sign(parameter)

        Args:
            param: parameter variable for which regularization is applied
            block: block in which variable is to be created

        Returns:
            new variable for weight decay
        """
        assert isinstance(param, framework.Parameter)
        assert isinstance(block, framework.Block)

        decay = block.create_var(
            dtype=param.dtype, shape=param.shape, lod_level=param.lod_level)

        # Append sign op
        block.append_op(
            type='sign', inputs={"X": param}, outputs={"Out": decay})

        # Append scale op to the output of sign op
        block.append_op(
            type='scale',
            inputs={"X": decay},
            outputs={"Out": decay},
            attrs={"scale": self._regularization_coeff})

        return decay

    def __str__(self):
        return "L1Decay, regularization_coeff=%f" % self._regularization_coeff


class WeightDecay(object):
    """
    WeightDecay is used to update the optimized parameters by using the
    parameters before optimization.

    Args:
        coeff (float|Variable): The weight decay coefficient, it can be float
            or Variable.
        attempt_decay_param_fun (function|None): If it is not None,
            only variables that makes attempt_decay_param_fun(variable)==True
            will be updated. It only works when we want to specify variables.
            Default: None.

    Examples:
        .. code-block:: python
            def name_has_fc(var):
                res = "fc" in var.name
                return res

            learning_rate = 0.1
            optimizer = fluid.optimizer.Adagrad(
                learning_rate=learning_rate,
                weight_decay=fluid.weight_decay.WeightDecay(
                    coeff=learning_rate, attempt_decay_param_fun=name_has_fc))
    """

    def __init__(self, coeff=0.0, attempt_decay_param_fun=None):

        if not isinstance(coeff, float) and \
                not isinstance(coeff, framework.Variable):
            raise TypeError("coeff should be float or Variable.")

        self.scaled_params_ = []
        self.params_name_ = set()
        self.attempt_decay_param_fun_ = attempt_decay_param_fun
        self.coeff_ = coeff

    def __call__(self, params_and_grads):
        from . import layers
        if isinstance(self.coeff_, float) and np.isclose(self.coeff_, 0.0):
            return

        for param, grad in params_and_grads:
            # If no gradient then we don't need to do anything
            if grad is None:
                continue
            if self.attempt_decay_param_fun_ is not None \
                    and not self.attempt_decay_param_fun_(param.name):
                continue

            if isinstance(self.coeff_, float):
                assert param.dtype is not core.VarDesc.VarType.FP32, \
                    "the type of coeff(float) and parameter(%s) is not consistent."%(self.coeff_.dtype)
            else:
                assert self.coeff_.dtype == param.dtype, \
                    "the type of coeff(%s) and parameter(%s) is not consistent."%(self.coeff_.dtype, param.dtype)

            with param.block.program._optimized_guard(
                [param, grad]), framework.name_scope('weight decay'):
                assert param.name not in self.params_name_
                self.scaled_params_.append((param, grad, param * self.coeff_))
                self.params_name_.append(param.name)

    def apply(self):
        """
        Update the optimized parameters.
        """
        from . import layers
        if isinstance(self.coeff_, float) and np.isclose(self.coeff_, 0.0):
            return
        for p_grad_sgrad in self.scaled_params_:
            param, grad, scaled_param = p_grad_sgrad
            with param.block.program._optimized_guard(
                [param, grad]), framework.name_scope('weight decay'):
                layers.elementwise_sub(x=param, y=scaled_param, out=param)

    def __str__(self):
        return " ".join(["Weight Decay, params:", ",".join(self.params_name_)])


# We short the class name, since users will use the regulaizer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# hidden = fluid.layers.fc(...,
#                          param_attr=fluid.regularizer.Xavier())
#
# It is no need to add a `Regularizer` as the class suffix
L1Decay = L1DecayRegularizer
L2Decay = L2DecayRegularizer
