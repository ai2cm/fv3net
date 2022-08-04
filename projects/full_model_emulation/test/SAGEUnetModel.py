import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


# class UnetGraphSAGE(nn.Module):
#     def __init__(self, g, in_feats, h_feats, out_feat, num_step, aggregator):
#         super(UnetGraphSAGE, self).__init__()
#         self.conv1 = SAGEConv(in_feats, h_feats, aggregator)
#         self.conv2 = SAGEConv(h_feats, int(h_feats / 2), aggregator)
#         self.conv3 = SAGEConv(int(h_feats / 2), int(h_feats / 4), aggregator)
#         self.conv4 = SAGEConv(int(h_feats / 4), int(h_feats / 4), aggregator)
#         self.conv5 = SAGEConv(int(h_feats / 2), int(h_feats / 2), aggregator)
#         self.conv6 = SAGEConv(h_feats, out_feat, aggregator)
#         self.g = g
#         self.num_step = num_step

#     def forward(self, in_feat, exteraVar1):
#         for _ in range(self.num_step):
#             # # h = self.conv1(self.g, in_feat)
#             # # h = F.relu(h)
#             # # h = self.conv2(self.g, h)
#             # # h = F.relu(h)
#             # # h = self.conv3(self.g, h)
#             # # h = F.relu(h)
#             # # # tuple = (self.conv4(self.g, h),h)
#             # # # h = torch.cat(tuple,dim=1)
#             # # # h = F.relu(h)
#             # # h = torch.cat((F.relu(self.conv4(self.g, h)),h),dim=1)
#             # # # tuple = (self.conv5(self.g, h),h)
#             # # # h = torch.cat(tuple,dim=1)
#             # # # h = F.relu(h)
#             # # h = torch.cat((F.relu(self.conv5(self.g, h)),h),dim=1)
#             # # h = self.conv6(self.g, h)
#             # # in_feat=torch.cat((h, torch.squeeze(exteraVar1)), 1).float()
#             # h1 = F.relu(self.conv1(self.g, in_feat))
#             # h2 = F.relu(self.conv2(self.g, h1))
#             # h3 = F.relu(self.conv3(self.g, h2))
#             # h4 = F.relu(self.conv4(self.g, h3))
#             # h5 = torch.cat((F.relu(self.conv4(self.g, h4)), h3), dim=1)
#             # h6 = torch.cat((F.relu(self.conv5(self.g, h5)), h2), dim=1)
#             # out = self.conv6(self.g, h6)
#             # in_feat = torch.cat((out, torch.squeeze(exteraVar1)), 1).float()

#             h1 = F.relu(self.conv1(self.g, in_feat))
#             h2 = F.relu(self.conv2(self.g, h1))
#             h3 = F.relu(self.conv3(self.g, h2))
#             h4 = F.relu(self.conv4(self.g, h3))

#             h4 = F.relu(self.conv4(self.g, h4))
#             h4 = F.relu(self.conv4(self.g, h4))
#             h4 = F.relu(self.conv4(self.g, h4))
#             h4 = F.relu(self.conv4(self.g, h4))
#             h4 = F.relu(self.conv4(self.g, h4))
#             h4 = F.relu(self.conv4(self.g, h4))

#             h5 = torch.cat((F.relu(self.conv4(self.g, h4)), h3), dim=1)
#             h6 = torch.cat((F.relu(self.conv5(self.g, h5)), h2), dim=1)
#             out = self.conv6(self.g, h6)
#             in_feat = torch.cat((out, torch.squeeze(exteraVar1)), 1).float()

#         return out


class UnetGraphSAGE(nn.Module):
    def __init__(self, g, in_feats, h_feats, out_feat, num_step, aggregat):
        super(UnetGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, int(h_feats / 16), aggregat)
        self.conv2 = SAGEConv(int(h_feats / 16), int(h_feats / 16), aggregat)

        self.conv3 = SAGEConv(int(h_feats / 16), int(h_feats / 8), aggregat)
        self.conv4 = SAGEConv(int(h_feats / 8), int(h_feats / 4), aggregat)
        self.conv5 = SAGEConv(int(h_feats / 4), int(h_feats / 2), aggregat)
        self.conv6 = SAGEConv(int(h_feats / 2), int(h_feats), aggregat)
        self.conv7 = SAGEConv(int(h_feats), int(h_feats), aggregat)
        self.conv8 = SAGEConv(int(h_feats), int(h_feats / 2), aggregat)
        self.conv9 = SAGEConv(int(h_feats / 2), int(h_feats / 4), aggregat)
        self.conv10 = SAGEConv(int(h_feats / 4), int(h_feats / 8), aggregat)
        self.conv11 = SAGEConv(int(h_feats / 8), int(h_feats / 16), aggregat)
        self.conv12 = SAGEConv(int(h_feats / 16), out_feat, aggregat)
        self.g = g
        self.num_step = num_step

    def forward(self, in_feat, exteraVar1):

        for _ in range(self.num_step):
            h1 = F.relu(self.conv1(self.g, in_feat))
            h2 = F.relu(self.conv2(self.g, h1))
            h3 = F.relu(self.conv3(self.g, h2))
            h4 = F.relu(self.conv4(self.g, h3))
            h5 = F.relu(self.conv5(self.g, h4))
            h6 = F.relu(self.conv6(self.g, h5))
            h7 = F.relu(self.conv7(self.g, h6))
            h8 = torch.cat((F.relu(self.conv8(self.g, h7)), h5), dim=1)
            h8 = F.relu(self.conv8(self.g, h8))
            h9 = torch.cat((F.relu(self.conv9(self.g, h8)), h4), dim=1)
            h9 = F.relu(self.conv9(self.g, h9))
            h10 = torch.cat((F.relu(self.conv10(self.g, h9)), h3), dim=1)
            h10 = F.relu(self.conv10(self.g, h10))
            h11 = torch.cat((F.relu(self.conv11(self.g, h10)), h2), dim=1)
            h11 = F.relu(self.conv11(self.g, h11))
            out = self.conv12(self.g, h11)
            in_feat = torch.cat((out, torch.squeeze(exteraVar1)), 1).float()
        return out