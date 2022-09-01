import torch
import torch.nn as nn
import dataclasses
from dgl.nn.pytorch import SAGEConv
from .graph_builder import build_dgl_graph
from fv3fit.pytorch.activation import ActivationConfig


@dataclasses.dataclass
class GraphUNetConfig:
    """
    Attributes:
        depth: depth of U-net architecture maximum
        min_filters: mimumum number of hidden channels after first convolution
        aggregator: type of aggregator, one of "mean", "gcn", "pool", or "lstm"
        pooling_size: size of the pooling kernel
        pooling_stride: pooling layer stride
        activation: activation function
    """

    depth: int = 1
    min_filters: int = 4
    aggregator: str = "mean"
    pooling_size: int = 2
    pooling_stride: int = 2
    activation: ActivationConfig = dataclasses.field(
        default_factory=lambda: ActivationConfig()
    )

    def build(self, in_channels: int, out_channels: int, nx: int,) -> "GraphUNet":
        return GraphUNet(self, in_channels=in_channels, out_channels=out_channels)


class CubedSphereGraphOperation(nn.Module):
    """
    A wrapper class which applies graph operations to cubed sphere data.
    """

    def __init__(self, graph_op: nn.Module):
        super().__init__()
        self.graph_op = graph_op

    def forward(self, inputs: torch.Tensor):
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

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        activation: nn.modules,
        aggregator: str,
    ):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            CubedSphereGraphOperation(
                SAGEConv(in_channels, hidden_channels, aggregator)
            ),
            activation.instance,
            CubedSphereGraphOperation(
                SAGEConv(hidden_channels, hidden_channels, aggregator)
            ),
            activation.instance,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x


class Down(nn.Module):
    """
    A class for the down path of the U-net which
    reduce the resolution by applying pooling layer
    """

    def __init__(self, config):
        super(Down, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=config.pooling_size, stride=config.pooling_stride
        )

    def forward(self, x: torch.Tensor):
        batch_size, n_tiles, n_x, n_y, n_features = x.size()
        x = x.permute(
            0, 1, 4, 2, 3
        )  # change dimensions to (batch_size, n_tiles, n_features, n_x, n_y)
        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4),)
        x = self.pool(x)
        x = x.reshape(batch_size, n_tiles, n_features, n_x // 2, n_y // 2,)

        return x.permute(0, 1, 3, 4, 2)


class Up(nn.Module):
    """
    A class for the processes on each level of up path of the U-Net
    """

    def __init__(self, config: GraphUNetConfig, in_channels: int):
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

    def forward(self, x1: torch.Tensor):
        batch_size, n_tiles, n_x, n_y, n_features = x1.size()
        x1 = x1.permute(
            0, 1, 4, 2, 3
        )  # change dimensions to (batch_size, n_tiles, n_features, n_x, n_y)
        x1 = x1.reshape(
            x1.size(0) * x1.size(1), x1.size(2), x1.size(3), x1.size(4)
        )  # change the shape to (batch_size*n_tiles, n_features, n_x, n_y )
        x1 = self.up(x1)
        x1 = x1.reshape(batch_size, n_tiles, n_features // 2, n_x * 2, n_y * 2,)
        x1 = x1.permute(
            0, 1, 3, 4, 2
        )  # change dimensions to (batch_size, n_tiles, n_x, n_y, n_features)
        return x1


class UNet(nn.Module):
    """
    A graph based U-net architucture
    """

    def __init__(
        self,
        config: GraphUNetConfig,
        down_factory,
        up_factory,
        depth: int,
        in_channels: int,
    ):
        """
        Args:
            config: Model configuration
            down_factory: double-convolution followed
            by a pooling layer on the down-path side
            up_factory: Upsampling followed by
            double-convolution on the up-path side
            depth: depth of the UNet
            in_channels: number of input channels
        """
        super(UNet, self).__init__()

        lower_channels = 2 * in_channels

        self._down = down_factory()

        self.conv1 = DoubleConv(
            in_channels, lower_channels, config.activation, config.aggregator
        )

        self.conv2 = DoubleConv(
            lower_channels * 2, lower_channels, config.activation, config.aggregator
        )

        if depth == 1:
            self._lower = DoubleConv(
                lower_channels, lower_channels * 2, config.activation, config.aggregator
            )
        elif depth <= 0:
            raise ValueError(f"depth must be at least 1, got {depth}")
        else:
            self._lower = UNet(
                config,
                down_factory,
                up_factory,
                depth=depth - 1,
                in_channels=lower_channels,
            )
        self._up = up_factory(in_channels=lower_channels * 2)
        self.depth = depth

    def forward(self, inputs: torch.Tensor):
        before_pooling = self.conv1(inputs)
        x = self._down(before_pooling)
        x = self._lower(x)
        x = self._up(x)
        x = torch.cat([before_pooling, x], dim=-1)
        x = self.conv2(x)
        return x


class GraphUNet(nn.Module):
    def __init__(self, config: GraphUNetConfig, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: number of input channels
        """

        super(GraphUNet, self).__init__()

        def down():
            return Down(config)

        def up(in_channels: int):
            return Up(config, in_channels)

        self._first_conv = nn.Sequential(
            CubedSphereGraphOperation(
                SAGEConv(in_channels, config.min_filters, config.aggregator)
            ),
            config.activation.instance,
        )

        self._last_conv = CubedSphereGraphOperation(
            SAGEConv(config.min_filters * 2, out_channels, config.aggregator)
        )

        self._unet = UNet(
            config,
            down_factory=down,
            up_factory=up,
            depth=config.depth,
            in_channels=config.min_filters,
        )

    def forward(self, inputs: torch.Tensor):
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
