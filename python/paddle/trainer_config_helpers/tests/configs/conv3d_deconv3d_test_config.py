from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='data1', size=12096, height=48, width=42, depth=6)

conv3d = img_conv3d_layer(
    input=data,
    filter_size=3,
    num_filters=16,
    name="conv3ddd",
    num_channels=3,
    act=LinearActivation(),
    groups=1,
    stride=1,
    padding=1,
    bias_attr=True,
    shared_biases=True,
    filter_size_y=3,
    stride_y=1,
    padding_y=1,
    filter_size_z=3,
    stride_z=1,
    padding_z=1,
    trans=False,
    layer_type="conv3d")

deconv3d = img_conv3d_layer(
    input=data,
    filter_size=3,
    num_filters=16,
    name="deconv3ddd",
    num_channels=3,
    act=LinearActivation(),
    groups=1,
    stride=1,
    padding=1,
    bias_attr=True,
    shared_biases=True,
    filter_size_y=3,
    stride_y=1,
    padding_y=1,
    filter_size_z=3,
    stride_z=1,
    padding_z=1,
    trans=True,
    layer_type="deconv3d")
