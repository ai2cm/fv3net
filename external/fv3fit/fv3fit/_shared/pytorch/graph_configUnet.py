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
    aggregat: str='mean'
    activation: Callable = F.relu


class graphnetwork(nn.Module):
    def __init__(self,config,g):
        super(graphnetwork,self).__init__()
        self.conv1 = SAGEConv(config.in_feats, config.n_hidden,config.aggregat)
        self.conv2 = SAGEConv(config.n_hidden, int(config.n_hidden/2), config.aggregat)
        self.conv3 = SAGEConv(int(config.n_hidden/2), int(config.n_hidden/4), config.aggregat)
        self.conv4 = SAGEConv(int(config.n_hidden/4), int(config.n_hidden/4), config.aggregat)
        self.conv5 = SAGEConv(int(config.n_hidden/2), int(config.n_hidden/2), config.aggregat)
        self.conv6 = SAGEConv(config.n_hidden, config.out_feats,config.aggregat)
        self.g=g
        self.config=config

    def forward(self,in_feat):
        in_feat = in_feat.transpose(0, 1)
        for _ in range(self.config.num_step):
            h = self.conv1(self.g, in_feat)
            h = self.config.activation(h)
            h = self.conv2(self.g, h)
            h = self.config.activation(h)
            h = self.conv3(self.g, h)
            h = self.config.activation(h)
            h = torch.cat((self.config.activation(self.conv4(self.g, h)),h),dim=2)
            h = torch.cat((self.config.activation(self.conv5(self.g, h)),h),dim=2)
            h = self.conv6(self.g, h)
            in_feat=h  
        return h.transpose(0, 1)
