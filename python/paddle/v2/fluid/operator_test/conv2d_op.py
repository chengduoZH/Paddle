import paddle.v2.fluid.core as core
import numpy as np
from paddle.v2.fluid.op import Operator


def create_op(scope, op_type, inputs, outputs, attrs):
    kwargs = dict()

    def __create_var__(name, var_name):
        scope.var(var_name).get_tensor()
        kwargs[name].append(var_name)

    for in_name, in_dup in Operator.get_op_inputs(op_type):
        if in_name in inputs:
            kwargs[in_name] = []
            if in_dup:
                sub_in = inputs[in_name]
                for sub_in_name, _ in sub_in:
                    __create_var__(in_name, sub_in_name)
            else:
                __create_var__(in_name, in_name)

    for out_name, out_dup in Operator.get_op_outputs(op_type):
        if out_name in outputs:
            kwargs[out_name] = []
            if out_dup:
                sub_out = outputs[out_name]
                for sub_out_name, _ in sub_out:
                    __create_var__(out_name, sub_out_name)
            else:
                __create_var__(out_name, out_name)

    for attr_name in Operator.get_op_attr_names(op_type):
        if attr_name in attrs:
            kwargs[attr_name] = attrs[attr_name]

    return Operator(op_type, **kwargs)


def set_input(scope, op, inputs, place):
    def __set_input__(var_name, var):
        if isinstance(var, tuple) or isinstance(var, np.ndarray):
            tensor = scope.find_var(var_name).get_tensor()
            if isinstance(var, tuple):
                tensor.set_lod(var[1])
                var = var[0]
            tensor.set_dims(var.shape)
            tensor.set(var, place)
        elif isinstance(var, float):
            scope.find_var(var_name).set_float(var)
        elif isinstance(var, int):
            scope.find_var(var_name).set_int(var)

    for in_name, in_dup in Operator.get_op_inputs(op.type()):
        if in_name in inputs:
            if in_dup:
                sub_in = inputs[in_name]
                for sub_in_name, sub_in_val in sub_in:
                    __set_input__(sub_in_name, sub_in_val)
            else:
                __set_input__(in_name, inputs[in_name])


class conv2d:
    def __init__(self,
                 input,
                 filter,
                 stride,
                 padding,
                 dilation,
                 groups,
                 place=core.CPUPlace()):
        # (TODO) checkout input and filter type("float32" or "float64")

        self.scope = core.Scope()
        self.op_inputs = {'Input': input, 'Filter': filter}
        self.op_attrs = {
            'strides': stride,
            'paddings': padding,
            'groups': groups,
            'dilations': dilation
        }
        self.op_outputs = {'Output': []}

        self.ctx = core.DeviceContext.create(place)
        self.op = create_op(self.scope, "conv2d", self.op_inputs,
                            self.op_outputs, self.op_attrs)

    def run(self):
        set_input(self.scope, self.op, self.op_inputs, core.CPUPlace())
        self.op.run(self.scope, self.ctx)
        output = np.array(self.scope.find_var("Output").get_tensor())
        return output


if __name__ == '__main__':

    pad = [1, 1]
    stride = [1, 1]
    input_size = [2, 3, 5, 5]  # NCHW
    groups = 3
    dilation = [1, 1]
    assert np.mod(input_size[1], groups) == 0
    f_c = input_size[1] / groups
    filter_size = [6, f_c, 3, 3]

    input = np.random.random(input_size).astype("float32")
    filter = np.random.random(filter_size).astype("float32")

    conv = conv2d(
        input=input,
        filter=filter,
        stride=stride,
        padding=pad,
        dilation=dilation,
        groups=groups)
    print conv.run()
