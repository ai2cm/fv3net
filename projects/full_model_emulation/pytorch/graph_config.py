import dataclasses
from typing import Callable, Any

import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


@dataclasses.dataclass
class GraphNetworkConfig:
    """
    Attributes:
        n_hidden: number of neurons in each hidden layer
        n_layers: number of hidden layers
        dropout: dropout value
        activation: activation function
        out_feats: number of output features
        in_feats: numbe rof input features
    """

    activation: Callable
    in_feats: int = 2
    out_feats: int = 2
    n_hidden: int = 256
    n_layers: int = 4
    dropout: float = 0.0


class graphnetwork(nn.Module):
    def __init__(self, config, g):
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(config.in_feats, config.n_hidden, activation=config.activation)
        )
        # hidden layers
        for i in range(config.n_layers - 1):
            self.layers.append(
                GraphConv(
                    config.n_hidden, config.n_hidden, activation=config.activation
                )
            )
        # output layer
        self.layers.append(GraphConv(config.n_hidden, config.out_feats))
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
