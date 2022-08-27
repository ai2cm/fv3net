import torch
import torch.nn as nn
import dataclasses
from typing import Callable
from .graph_builder_edge_feat import build_dgl_graph_with_edge
from dgl.nn.pytorch import NNConv


@dataclasses.dataclass
class MPGUNetGraphNetworkConfig:
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

    depth: int = 1
    min_filters: int = 4
    num_step_message_passing: int = 1
    aggregator: str = "mean"
    edge_hidden_feats: int = 4
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

    def forward(self, inputs, nx):
        """
        Args:
            inputs: tensor of shape (batch_size, n_tiles, n_x, n_y, n_features)
        """
        graph, edge_relation = build_dgl_graph_with_edge(nx_tile=nx)
        convolved = self.graph_op(graph, inputs, edge_relation)
        return convolved


class MPNNGNN(nn.Module):
    """
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node representations.
    edge_in_feats : int
        Size for the input edge features.
    edge_hidden_feats : int
        Size for the hidden edge representations.
    num_step_message_passing : int
        Number of message passing steps.
    """

    def __init__(
        self,
        node_in_channels,
        node_hidden_channels,
        edge_hidden_channels,
        num_step_message_passing,
        activation,
        aggregator,
    ):
        super(MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_channels, node_hidden_channels),
            activation,
            nn.Linear(node_hidden_channels, node_hidden_channels),
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(2, edge_hidden_channels),
            activation,
            nn.Linear(
                edge_hidden_channels, node_hidden_channels * node_hidden_channels
            ),
        )
        self.gnn_layer = CubedSphereGraphOperation(
            NNConv(
                in_feats=node_hidden_channels,
                out_feats=node_hidden_channels,
                edge_func=edge_network,
                aggregator_type=aggregator,
            )
        )
        self.gru = nn.GRU(node_hidden_channels, node_hidden_channels)
        self.relu = activation
        self.out_feats = node_hidden_channels

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, in_node_feats):
        """Performs message passing and updates node representations.

        Args:
        ----------
        g : DGL graph
        node_feats : float32 tensor of shape (grid, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (edge, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        node_feats : float32 tensor of shape (grid, node_out_feats)
            Output node representations.
        """
        out_node_feats = torch.zeros(
            in_node_feats.size(0),
            in_node_feats.size(1),
            in_node_feats.size(2),
            in_node_feats.size(3),
            self.out_feats,
        )
        for batch in range(in_node_feats.size(0)):
            node_feats = in_node_feats[batch].squeeze()
            in_size = node_feats.size()
            node_feats = node_feats.reshape(
                node_feats.shape[0] * node_feats.shape[1] * node_feats.shape[2],
                node_feats.shape[3],
            )

            node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
            hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

            for _ in range(self.num_step_message_passing):
                node_feats = self.relu(self.gnn_layer(node_feats, in_size[2]))
                node_feats, hidden_feats = self.gru(
                    node_feats.unsqueeze(0), hidden_feats
                )
                node_feats = node_feats.squeeze(0)
            out_node_feats[batch] = node_feats.reshape(
                in_size[0], in_size[1], in_size[2], node_feats.size(1)
            )
        return out_node_feats


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

    def __init__(self, config, down_factory, up_factory, depth: int, in_channels: int):
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
            config.edge_hidden_feats,
            config.num_step_message_passing,
            config.activation,
            config.aggregator,
        )

        self.conv2 = MPNNGNN(
            2 * node_lower_channels,
            node_lower_channels,
            config.edge_hidden_feats,
            config.num_step_message_passing,
            config.activation,
            config.aggregator,
        )

        if depth == 1:
            self._lower = MPNNGNN(
                node_lower_channels,
                node_lower_channels * 2,
                config.edge_hidden_feats,
                config.num_step_message_passing,
                config.activation,
                config.aggregator,
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
    def __init__(self, config, in_channels: int, out_dim: int):
        """
        Args:
            in_channels: number of input channels
        """

        super(MPGraphUNet, self).__init__()

        def down():
            return Down(config)

        def up(in_channels: int):
            return Up(config, in_channels)

        self._first_layer = MPNNGNN(
            in_channels,
            config.min_filters,
            config.edge_hidden_feats,
            config.num_step_message_passing,
            config.activation,
            config.aggregator,
        )

        self._last_layer = MPNNGNN(
            config.min_filters * 2,
            config.min_filters,
            config.edge_hidden_feats,
            config.num_step_message_passing,
            config.activation,
            config.aggregator,
        )

        self._last_layer = nn.Sequential(
            nn.Linear(config.min_filters * 2, config.min_filters),
            nn.ReLU(),
            nn.Linear(config.min_filters, out_dim),
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
        x = self._first_layer(inputs)
        x = self._unet(x)
        outputs = self._last_layer(x)
        return outputs
