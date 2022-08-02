import dataclasses
from typing import Callable, Any
import torch.nn.functional as F
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, SAGEConv


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

    
    in_feats: int = 2
    out_feats: int = 2
    n_hidden: int = 256
    n_layers: int = 4
    num_step: int = 4
    dropout: float = 0.0
    aggregat: str='mean'
    activation: Callable = torch.nn.ReLU


class graphnetwork(nn.Module):
    def __init__(self,config, g):
        super(self,config).__init__()
        self.conv1 = SAGEConv(config.in_feats, config.n_hidden,config.aggregat)
        self.conv2 = SAGEConv(config.n_hidden, int(config.h_feats/2), config.aggregat)
        self.conv3 = SAGEConv(int(config.n_hidden/2), int(config.n_hidden/4), config.aggregat)
        self.conv4 = SAGEConv(int(config.n_hidden/4), int(config.n_hidden/4), config.aggregat)
        self.conv5 = SAGEConv(int(config.n_hidden/2), int(config.n_hidden/2), config.aggregat)
        self.conv6 = SAGEConv(config.h_feats, config.out_feat,config.aggregat)
        self.g=g
        self.num_step=config.num_step
        
    def forward(self, in_feat):

        for _ in range(self.num_step):
            h = self.conv1(self.g, in_feat)
            h = F.relu(h)
            h = self.conv2(self.g, h)
            h = F.relu(h)
            h = self.conv3(self.g, h)
            h = F.relu(h)
            tuple = (self.conv4(self.g, h),h)
            h = torch.cat(tuple,dim=1)
            h = F.relu(h)
            tuple = (self.conv5(self.g, h),h)
            h = torch.cat(tuple,dim=1)
            h = F.relu(h)
            h = self.conv6(self.g, h)
            in_feat=h
        return h
