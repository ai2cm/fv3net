import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class UnetGraphSAGE(nn.Module):
    def __init__(self, g, in_feats, h_feats, out_feat, num_step, aggregator):
        super(UnetGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator)
        self.conv2 = SAGEConv(h_feats, int(h_feats / 2), aggregator)
        self.conv3 = SAGEConv(int(h_feats / 2), int(h_feats / 4), aggregator)
        self.conv4 = SAGEConv(int(h_feats / 4), int(h_feats / 4), aggregator)
        self.conv5 = SAGEConv(int(h_feats / 2), int(h_feats / 2), aggregator)
        self.conv6 = SAGEConv(h_feats, out_feat, aggregator)
        self.g = g
        self.num_step = num_step

    def forward(self, in_feat, exteraVar1):
        for _ in range(self.num_step):
            # h = self.conv1(self.g, in_feat)
            # h = F.relu(h)
            # h = self.conv2(self.g, h)
            # h = F.relu(h)
            # h = self.conv3(self.g, h)
            # h = F.relu(h)
            # # tuple = (self.conv4(self.g, h),h)
            # # h = torch.cat(tuple,dim=1)
            # # h = F.relu(h)
            # h = torch.cat((F.relu(self.conv4(self.g, h)),h),dim=1)
            # # tuple = (self.conv5(self.g, h),h)
            # # h = torch.cat(tuple,dim=1)
            # # h = F.relu(h)
            # h = torch.cat((F.relu(self.conv5(self.g, h)),h),dim=1)
            # h = self.conv6(self.g, h)
            # in_feat=torch.cat((h, torch.squeeze(exteraVar1)), 1).float()
            h1 = F.relu(self.conv1(self.g, in_feat))
            h2 = F.relu(self.conv2(self.g, h1))
            h3 = F.relu(self.conv3(self.g, h2))
            h4 = F.relu(self.conv4(self.g, h3))
            h5 = torch.cat((F.relu(self.conv4(self.g, h4)), h3), dim=1)
            h6 = torch.cat((F.relu(self.conv5(self.g, h5)), h2), dim=1)
            out = self.conv6(self.g, h6)
            in_feat = torch.cat((out, torch.squeeze(exteraVar1)), 1).float()

        return out
