import dataclasses
from typing import Callable
import torch.nn.functional as F
import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv
from .graph_builder import build_dgl_graph


@dataclasses.dataclass
class GraphNetworkConfig:
    """
    Attributes:
        n_hidden: maximum number of hidden channels
        num_blocks: number of nested network (e.g, 2 means N(N(theta)))
        aggregator: type of aggregator, one of "mean", "gcn", "pool", or "lstm"
        activation: activation function
    """

    n_hidden: int = 256
    num_blocks: int = 1
    aggregator: str = "mean"
    activation: Callable = F.relu


class CubedSphereGraphOperation(nn.Module):
    """
    A wrapper class which applies graph operations to cubed sphere data.
    """

    def __init__(self, graph_op: nn.Module):
        """
        Initializes the wrapped graph operation.

        Args:
            graph_op: graph operation to apply, whose first argument should be
                a DGL graph and second argument should be a tensor of shape
                (n_tiles * n_x * n_y, batch_size, n_features)
        """
        super().__init__()
        self.graph_op = graph_op

    def forward(self, inputs):
        """
        Args:
            inputs: tensor of shape (batch_size, n_tiles, n_x, n_y, n_features)

        Returns:
            tensor of shape (batch_size, n_tiles, n_x, n_y, n_features_out)
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


class GraphNetwork(nn.Module):
    def __init__(self, config, n_features_in: int, n_features_out: int):
        super(GraphNetwork, self).__init__()
        self.conv1 = CubedSphereGraphOperation(
            SAGEConv(n_features_in, config.n_hidden, config.aggregator)
        )
        self.conv2 = CubedSphereGraphOperation(
            SAGEConv(config.n_hidden, int(config.n_hidden / 2), config.aggregator)
        )
        self.conv3 = CubedSphereGraphOperation(
            SAGEConv(
                int(config.n_hidden / 2), int(config.n_hidden / 4), config.aggregator
            )
        )
        self.conv4 = CubedSphereGraphOperation(
            SAGEConv(
                int(config.n_hidden / 4), int(config.n_hidden / 4), config.aggregator
            )
        )
        self.conv5 = CubedSphereGraphOperation(
            SAGEConv(
                int(config.n_hidden / 2), int(config.n_hidden / 2), config.aggregator
            )
        )
        self.conv6 = CubedSphereGraphOperation(
            SAGEConv(config.n_hidden, n_features_out, config.aggregator)
        )
        self.config = config

    def forward(self, inputs):
        for _ in range(self.config.num_blocks):
            h1 = self.conv1(inputs)
            h1 = self.config.activation(h1)
            h2 = self.conv2(h1)
            h2 = self.config.activation(h2)
            h3 = self.conv3(h2)
            h3 = self.config.activation(h3)
            h4 = self.conv4(h3)
            h4 = self.config.activation(h4)
            h5 = torch.cat((self.config.activation(self.conv4(h4)), h3), dim=-1)
            h6 = torch.cat((self.config.activation(self.conv5(h5)), h2), dim=-1)
            out = self.conv6(h6)
            inputs = out
        return out
