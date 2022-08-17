import dataclasses
from typing import Callable
import torch.nn.functional as F
import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv
from ..graph import build_dgl_graph, CubedSphereGraphOperation


def relu_activation():
    return nn.ReLU()


class ResnetBlock(nn.Module):
    def __init__(
        self,
        n_filters: int,
        convolution_factory: Callable[[int], nn.Module],
        activation_factory: Callable[[], nn.Module] = relu_activation,
    ):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            convolution_factory(n_filters),
            nn.InstanceNorm2d(n_filters),
            activation_factory(),
            convolution_factory(n_filters),
            nn.InstanceNorm2d(n_filters),
        )

    def forward(self, inputs):
        g = self.conv_block(inputs)
        # skip-connection
        g = torch.concat([g, inputs], dim=-1)
        return g


class ConvBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self, inputs):
        pass


class CycleGenerator(nn.Module):
    def __init__(self, config, n_features_in: int, n_features_out: int):
        super(CycleGenerator, self).__init__()
        self.conv1 = CubedSphereGraphOperation(
            SAGEConv(n_features_in, config.n_hidden, config.aggregator)
        )
        self.config = config

    def forward(self, inputs):
        for _ in range(self.config.num_blocks):
            h1 = self.conv1(inputs)
            out = self.config.activation(h1)
        return out


def define_generator():
    pass


def define_discriminator():
    pass


def define_composite_model():
    pass
