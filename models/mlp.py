from abc import ABC

import torch
import torch.nn as nn


class MLP(nn.Module, ABC):
    def __init__(self, input_dim, output_dim, hidden_dims, activation='tanh', last_activation='identity', biases=None):
        super(MLP, self).__init__()
        sizes_list = hidden_dims.copy()
        self.activation = getattr(torch, activation)
        self.last_activation = getattr(torch, last_activation)
        sizes_list.insert(0, input_dim)
        biases = [True] * len(sizes_list) if biases is None else biases.copy()

        layers = []
        if 1 < len(sizes_list):
            for i in range(len(sizes_list) - 1):
                layers.append(nn.Linear(sizes_list[i], sizes_list[i + 1], bias=biases[i]))
        self.last_layer = nn.Linear(sizes_list[-1], output_dim)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.last_layer(x)
        x = self.last_activation(x)
        return x

    def init(self, init_fn, last_init_fn):
        for layer in self.layers:
            init_fn(layer)
        last_init_fn(self.last_layer)
