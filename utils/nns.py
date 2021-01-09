import math
import torch
import torchvision
import torch.nn as nn


def create_mlp(dim_input, dim_hiddens, act_layer=nn.ReLU,
               act_args={"inplace": True}, norm=True):
    """create MLP network
    :param dim_input: dimension of the input layer
    :param dim_hiddens: a list of dimensions of the hidden layers
    :param act_layer: activation layer class
    :param act_args: dict args used to init act_layer (None or {} if no need)
    :param norm: whether to normalize (use BatchNorm1d)
    :return: a MLP network (include len(dim_hiddens) layers)
    """
    modules = []
    dim_in = dim_input
    act_args = act_args or {}
    # hidden layers
    for dim_h in dim_hiddens:
        modules.append(nn.Linear(dim_in, dim_h))
        if norm:
            modules.append(nn.BatchNorm1d(dim_h))
        modules.append(act_layer(**act_args))

        dim_in = dim_h

    return nn.Sequential(*modules)


def create_cnn2d(channel_in, channel_hiddens, kernel_size, stride, padding,
                 act_layer=nn.ReLU, act_args={"inplace": True}, norm=True):
    """create 2d CNN
    :param channel_in: number of input channels
    :param channel_hiddens: number of the input channels of the hidden layers
    :param kernel_size: param kernel_size for all the conv2d layers
    :param stride: param stride for all the Conv2d layers
    :param padding: param padding for all the Conv2d layers
    :param act_layer: activation layer class
    :param act_args: dict args used to init act_layer (None or {} if no need)
    :param norm: whether to normalize (use BatchNorm1d)
    :return: a 2d CNN network (include len(dim_hiddens) layers)
    """
    modules = []
    ch_in = channel_in
    act_args = act_args or {}
    # hidden layers
    for ch_h in channel_hiddens:
        modules.append(nn.Conv2d(ch_in, ch_h, kernel_size, stride, padding))
        if norm:
            modules.append(nn.BatchNorm2d(ch_h))
        modules.append(act_layer(**act_args))

        ch_in = ch_h

    return nn.Sequential(*modules)


def create_transpose_cnn2d(channel_in, channel_hiddens, kernel_size, stride,
                           padding, output_padding, act_layer=nn.ReLU,
                           act_args={"inplace": True}, norm=True):
    """create 2d transposed CNN
    :param channel_in: number of input channels
    :param channel_hiddens: number of the input channels of the hidden layers
    :param kernel_size: param kernel_size for all the conv2d layers
    :param stride: param stride for all the ConvTranspose2d layers
    :param padding: param padding for all the ConvTranspose2d layers
    :param output_padding: param out_padding for all the ConvTranspose2d layers
    :param act_layer: activation layer class
    :param act_args: dict args used to init act_layer (None or {} if no need)
    :param norm: whether to normalize (use BatchNorm1d)
    :return: a 2d CNN network (include len(dim_hiddens) layers)
    """
    modules = []
    ch_in = channel_in
    act_args = act_args or {}
    # hidden layers
    for ch_h in channel_hiddens:
        modules.append(nn.ConvTranspose2d(
            ch_in, ch_h, kernel_size, stride, padding, output_padding))
        if norm:
            modules.append(nn.BatchNorm2d(ch_h))
        modules.append(act_layer(**act_args))

        ch_in = ch_h

    return nn.Sequential(*modules)


def cal_cnn2d_shape(h_in, w_in, kernel_size, n_layers=1,
                    stride=1, padding=0, dilation=1):
    """calculate the output shape of cnns with input shape h_in x w_in
    :param n_layers: number of cnn2d layers with the same param
    """
    h_out, w_out = h_in, w_in
    for _ in range(n_layers):
        h_out = math.floor(
            (h_out + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
        w_out = math.floor(
            (w_out + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)

    return h_out, w_out
