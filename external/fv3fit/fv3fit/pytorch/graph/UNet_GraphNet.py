import torch
import torch.nn as nn
import dataclasses
from dgl.nn.pytorch import SAGEConv
from typing import Callable
from .graph_builder import build_dgl_graph


@dataclasses.dataclass
class UNetGraphNetworkConfig:
    """
    Attributes:
        depth: depth of U-net architecture maximum
        min_filters: mimumum number of hidden channels after first convolution
        aggregator: type of aggregator, one of "mean", "gcn", "pool", or "lstm"
        pooling_size: size of the pooling kernel
        pooling_stride: pooling layer stride
        activation: activation function
    """

    depth: int = 5
    min_filters: int = 16
    aggregator: str = "mean"
    pooling_size: int = 2
    pooling_stride: int = 2
    activation: Callable = nn.ReLU()


class CubedSphereGraphOperation(nn.Module):
    """
    A wrapper class which applies graph operations to cubed sphere data.
    """

    def __init__(self, graph_op: nn.Module):
        super().__init__()
        self.graph_op = graph_op

    def forward(self, inputs):
        """
        Args:
            inputs: tensor of shape (batch_size, n_tiles, n_x, n_y, n_features)
        """
        if len(inputs.shape) != 5:
            raise ValueError(
                "inputs must be of shape (batch_size, n_tiles, n_x, n_y, n_features), "
                f"got {inputs.shape}"
            )
        graph = build_dgl_graph(nx_tile=inputs.shape[2])
        reshaped = inputs.reshape(
            inputs.shape[0],
            inputs.shape[1] * inputs.shape[2] * inputs.shape[3],
            inputs.shape[4],
        ).transpose(0, 1)
        convolved = self.graph_op(graph, reshaped).transpose(1, 0)
        return convolved.reshape(
            inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3], -1
        )


class DoubleConv(nn.Module):
    """
    A class which applies 2 graph convolution layers, each followed by an activation.
    """

    def __init__(self, in_channels, hidden_channels, activation, aggregator):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            CubedSphereGraphOperation(
                SAGEConv(in_channels, hidden_channels, aggregator)
            ),
            activation,
            CubedSphereGraphOperation(
                SAGEConv(hidden_channels, hidden_channels, aggregator)
            ),
            activation,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    """
    A class for the down path of the U-net which
    reduce the resolution by applying pooling layer
    """

    def __init__(self, config, in_channels, out_channels):
        """
        Args:
            in_channels: size of input channels
            out_channels: size of output channels
        """
        super(Down, self).__init__()
        self.conv = DoubleConv(
            in_channels, out_channels, config.activation, config.aggregator
        )
        self.pool = nn.MaxPool2d(
            kernel_size=config.pooling_size, stride=config.pooling_stride
        )

    def forward(self, x):
        before_pooling = self.conv(x)
        x = self.pool(before_pooling)
        return x, before_pooling


class Up(nn.Module):
    """
    A class for the processes on each level of up path of the U-Net
    """

    def __init__(self, config, in_channels):
        """
        Args:
            in_channels: size of input channels
        """
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=config.pooling_size,
            stride=config.pooling_stride,
        )
        self.conv = DoubleConv(
            in_channels, in_channels // 2, config.activation, config.aggregator
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=-1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    A graph based U-net architucture
    """

    def __init__(
        self, config, down_factory, up_factory, depth: int, in_channels: int,
    ):
        """
        Args:
            down_factory: double-convolution followed
                by a pooling layer on the down-path side
            up_factory: Upsampling followed by
                double-convolution on the up-path side
            depth: depth of the UNet
            in_channels: number of input channels
        """
        super(UNet, self).__init__()

        lower_channels = 2 * config.min_filters

        self._down = down_factory(in_channels=in_channels, out_channels=lower_channels)
        self._up = up_factory(in_channels=lower_channels)
        if depth == 1:
            self._lower = DoubleConv(
                lower_channels, lower_channels, config.activation, config.aggregator
            )
        elif depth <= 0:
            raise ValueError(f"depth must be at least 1, got {depth}")
        else:
            self._lower = UNet(
                config,
                down_factory(
                    in_channels=lower_channels, out_channels=lower_channels // 2
                ),
                up_factory(in_channels=lower_channels // 2),
                depth=depth - 1,
                in_channels=lower_channels,
            )

    def forward(self, inputs):
        x, before_pooling = self._down(inputs)
        x = self._lower(x)
        x = self._up(x, before_pooling)
        return x


class GraphUNet(nn.Module):
    def __init__(self, config, in_channels: int, out_dim: int):
        """
        Args:
            in_channels: number of input channels
        """

        super(GraphUNet, self).__init__()

        def down(in_channels: int, out_channels: int):
            return Down(config, in_channels=in_channels, out_channels=out_channels)

        def up(in_channels: int):
            return Up(config, in_channels)

        self._first_conv = nn.Sequential(
            CubedSphereGraphOperation(
                SAGEConv(in_channels, config.min_filters, config.aggregator)
            ),
            config.activation,
        )

        self._last_conv = CubedSphereGraphOperation(
            SAGEConv(config.min_filters, out_dim, config.aggregator)
        )

        self._unet = UNet(
            config,
            down_factory=down,
            up_factory=up,
            depth=config.depth,
            in_channels=config.min_filters,
        )

    def forward(self, inputs):
        """
        Args:
            inputs: tensor of shape (batch_size, n_tiles, n_x, n_y, n_features)
        Returns:
            tensor of shape (batch_size, n_tiles, n_x, n_y, n_features_out)
        """
        x = self._first_conv(inputs)
        x = self._unet(x)
        outputs = self._last_conv(x)
        return outputs
