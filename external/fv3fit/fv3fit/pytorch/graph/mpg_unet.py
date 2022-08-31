import torch
import torch.nn as nn
import dataclasses
from typing import Callable
from .graph_builder import build_dgl_graph_with_edge
from dgl.nn.pytorch import NNConv


@dataclasses.dataclass
class MPGraphUNetConfig:
    """
    Attributes:
        depth: depth of U-net architecture maximum
        min_filters: mimumum number of hidden channels after first convolution
        num_step_message_passing : number of message passing steps.
        aggregator: type of aggregator, one of "mean", "sum", or "max"
        pooling_size: size of the pooling kernel
        pooling_stride: pooling layer stride
        activation: activation function
    """

    num_step_message_passing: int
    depth: int = 1
    min_filters: int = 4
    aggregator: str = "mean"
    edge_hidden_features: int = 4
    pooling_size: int = 2
    pooling_stride: int = 2
    activation: Callable = nn.ReLU()

    def build(
        self, in_channels: int, out_channels: int, nx: int,
    ):
        return MPGraphUNet(
            self, in_channels=in_channels, out_channels=out_channels, nx=nx
        )


class MPNNGNN(nn.Module):
    def __init__(
        self,
        node_in_channels,
        node_hidden_channels,
        edge_hidden_channels,
        num_step_message_passing,
        activation,
        aggregator,
        nx,
    ):
        """
        MPNN is introduced in `Neural Message Passing for Quantum Chemistry
        <https://arxiv.org/abs/1704.01212>`__.

        Args:
        ----------
        node_in_channels : int
            Size for the input node channels.
        node_hidden_channels : int
            Size for the hidden node representations.
        edge_hidden_channels : int
            Size for the hidden edge representations.
        num_step_message_passing : int
            Number of message passing steps.
        activation: activation function
        aggregator: aggregator type, one of 'sum', 'mean', 'max'
        nx: number of horizontal grid points on each tile of the cubed sphere
        """
        super(MPNNGNN, self).__init__()
        self.graph, self.edge_relation = build_dgl_graph_with_edge(nx_tile=nx)
        self.project_node_features = nn.Sequential(
            nn.Linear(node_in_channels, node_hidden_channels),
            activation,
            nn.Linear(node_hidden_channels, node_hidden_channels),
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(self.edge_relation.size(1), edge_hidden_channels),
            activation,
            nn.Linear(
                edge_hidden_channels, node_hidden_channels * node_hidden_channels
            ),
        )
        self.gnn_layer = NNConv(
            in_feats=node_hidden_channels,
            out_feats=node_hidden_channels,
            edge_func=edge_network,
            aggregator_type=aggregator,
        )
        self.gru = nn.GRU(node_hidden_channels, node_hidden_channels)
        self.relu = activation
        self.out_features = node_hidden_channels

    def forward(self, in_node_features):
        """Performs message passing and updates node representations.

        Args:
        g : DGL graph
        in_node_features : Input node features.

        Returns:
        out_node_features : Output node representations.
        """

        out_node_features = torch.zeros(
            in_node_features.size(0),
            in_node_features.size(1),
            in_node_features.size(2),
            in_node_features.size(3),
            self.out_features,
        )  # initialize the updated node features (n_batch,tile,x,y,features)

        for batch in range(
            in_node_features.size(0)
        ):  # for loop over the n_batch since dgl NNConv
            # only accepts data in (nodes, features) format
            node_features = in_node_features[batch].squeeze()
            in_size = node_features.size()
            node_features = node_features.reshape(
                node_features.shape[0]
                * node_features.shape[1]
                * node_features.shape[2],
                node_features.shape[3],
            )  # reshape (tile, x, y, features) to (tile * x * y, features)

            node_features = self.project_node_features(
                node_features
            )  # (nodes, node_out_channels)
            hidden_features = node_features.unsqueeze(
                0
            )  # (1, nodes , node_out_channels)

            for _ in range(self.num_step_message_passing):
                node_features = self.relu(
                    self.gnn_layer(self.graph, node_features, self.edge_relation)
                )
                node_features, hidden_features = self.gru(
                    node_features.unsqueeze(0), hidden_features
                )
                node_features = node_features.squeeze(0)
            out_node_features[batch] = node_features.reshape(
                in_size[0], in_size[1], in_size[2], node_features.size(1)
            )  # reshape (tile * x * y, features) to (tile, x, y, features)
        return out_node_features


class Down(nn.Module):
    """
    A class for the down path of the U-net which
    reduce the resolution by applying pooling layer
    """

    def __init__(self, pooling_size, pooling_stride):
        super(Down, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_stride)

    def forward(self, x):
        input_size = x.size()
        x = x.permute(
            0, 1, 4, 2, 3
        )  # change dimensions to (batch_size, n_tiles, n_features, n_x, n_y)
        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4),)
        x = self.pool(x)
        x = x.reshape(
            input_size[0],
            input_size[1],
            input_size[4],
            input_size[2] // 2,
            input_size[3] // 2,
        )

        return x.permute(0, 1, 3, 4, 2)


class Up(nn.Module):
    """
    A class for the processes on each level of up path of the U-Net
    """

    def __init__(self, pooling_size, pooling_stride, in_channels):
        """
        Args:
            in_channels: size of input channels
        """
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=pooling_size,
            stride=pooling_stride,
        )

    def forward(self, x1):
        input_size = x1.size()
        x1 = x1.permute(
            0, 1, 4, 2, 3
        )  # change dimensions to (batch_size, n_tiles, n_features, n_x, n_y)
        x1 = x1.reshape(
            x1.size(0) * x1.size(1), x1.size(2), x1.size(3), x1.size(4)
        )  # change the shape to (batch_size*n_tiles, n_features, n_x, n_y )
        x1 = self.up(x1)
        x1 = x1.reshape(
            input_size[0],
            input_size[1],
            input_size[4] // 2,  # channel
            input_size[2] * 2,  # x
            input_size[3] * 2,  # y
        )
        x1 = x1.permute(
            0, 1, 3, 4, 2
        )  # change dimensions to (batch_size, n_tiles, n_x, n_y, n_features)
        return x1


class UNet(nn.Module):
    """
    A graph based U-net architucture
    """

    def __init__(
        self, config, down_factory, up_factory, depth: int, in_channels: int, nx: int
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

        node_lower_channels = 2 * in_channels
        self._down = down_factory()

        self.conv1 = MPNNGNN(
            in_channels,
            node_lower_channels,
            config.edge_hidden_features,
            config.num_step_message_passing,
            config.activation,
            config.aggregator,
            nx,
        )

        self.conv2 = MPNNGNN(
            2 * node_lower_channels,
            node_lower_channels,
            config.edge_hidden_features,
            config.num_step_message_passing,
            config.activation,
            config.aggregator,
            nx,
        )

        if depth == 1:
            self._lower = MPNNGNN(
                node_lower_channels,
                node_lower_channels * 2,
                config.edge_hidden_features,
                config.num_step_message_passing,
                config.activation,
                config.aggregator,
                nx=nx // 2,
            )

        elif depth <= 0:
            raise ValueError(f"depth must be at least 1, got {depth}")
        else:
            self._lower = UNet(
                config,
                down_factory,
                up_factory,
                depth=depth - 1,
                in_channels=node_lower_channels,
                nx=nx // 2,
            )
        self._up = up_factory(in_channels=node_lower_channels * 2)

    def forward(self, inputs):
        before_pooling = self.conv1(inputs)
        x = self._down(before_pooling)
        x = self._lower(x)
        x = self._up(x)
        x = torch.cat([before_pooling, x], dim=-1)
        x = self.conv2(x)
        return x


class MPGraphUNet(nn.Module):
    def __init__(self, config, in_channels: int, out_channels: int, nx: int):
        """
        Args:
            in_channels: number of input channels
        """

        super(MPGraphUNet, self).__init__()

        def down():
            return Down(config.pooling_size, config.pooling_stride)

        def up(in_channels: int):
            return Up(config.pooling_size, config.pooling_stride, in_channels)

        self._first_layer = MPNNGNN(
            in_channels,
            config.min_filters,
            config.edge_hidden_features,
            config.num_step_message_passing,
            config.activation,
            config.aggregator,
            nx,
        )

        self._last_layer = nn.Sequential(
            nn.Linear(config.min_filters * 2, config.min_filters),
            nn.ReLU(),
            nn.Linear(config.min_filters, out_channels),
        )

        self._unet = UNet(
            config,
            down_factory=down,
            up_factory=up,
            depth=config.depth,
            in_channels=config.min_filters,
            nx=nx,
        )

    def forward(self, inputs):
        """
        Args:
            inputs: tensor of shape (batch_size, n_tiles, n_x, n_y, n_features)
        Returns:
            tensor of shape (batch_size, n_tiles, n_x, n_y, n_features_out)
        """
        x = self._first_layer(inputs)
        x = self._unet(x)
        outputs = self._last_layer(x)
        return outputs
