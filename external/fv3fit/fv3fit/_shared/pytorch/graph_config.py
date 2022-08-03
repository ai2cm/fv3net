import dataclasses
from typing import Callable
import torch.nn.functional as F
import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv


@dataclasses.dataclass
class GraphNetworkConfig:
    """
    Attributes:
        in_feats: number of input features
        out_feats: number of output features
        n_hidden: number of hidden channels
        num_step: number of U-net blocks
        aggregat: type of aggregator
        activation: activation function
    """

    in_feats: int = 2
    out_feats: int = 2
    n_hidden: int = 256
    num_step: int = 1
    aggregat: str = "mean"
    activation: Callable = F.relu


class graphnetwork(nn.Module):
    def __init__(self, config, g):
        super(graphnetwork, self).__init__()
        self.conv1 = SAGEConv(config.in_feats, config.n_hidden, config.aggregat)
        self.conv2 = SAGEConv(
            config.n_hidden, int(config.n_hidden / 2), config.aggregat
        )
        self.conv3 = SAGEConv(
            int(config.n_hidden / 2), int(config.n_hidden / 4), config.aggregat
        )
        self.conv4 = SAGEConv(
            int(config.n_hidden / 4), int(config.n_hidden / 4), config.aggregat
        )
        self.conv5 = SAGEConv(
            int(config.n_hidden / 2), int(config.n_hidden / 2), config.aggregat
        )
        self.conv6 = SAGEConv(config.n_hidden, config.out_feats, config.aggregat)
        self.g = g
        self.config = config

    def forward(self, in_feat):
        in_feat = in_feat.transpose(0, 1)
        for _ in range(self.config.num_step):
            h1 = self.conv1(self.g, in_feat)
            h1 = self.config.activation(h1)
            h2 = self.conv2(self.g, h1)
            h2 = self.config.activation(h2)
            h3 = self.conv3(self.g, h2)
            h3 = self.config.activation(h3)
            h4 = self.conv4(self.g, h3)
            h4 = self.config.activation(h4)
            h5 = torch.cat((self.config.activation(self.conv4(self.g, h4)), h3), dim=2)
            h6 = torch.cat((self.config.activation(self.conv5(self.g, h5)), h2), dim=2)
            out = self.conv6(self.g, h6)
            in_feat = out
        return out.transpose(0, 1)
