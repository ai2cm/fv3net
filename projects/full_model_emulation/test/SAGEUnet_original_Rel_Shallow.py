import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv


class UnetGraphSAGE(nn.Module):
    def __init__(
        self,
        input_res,
        pooling_size,
        g1,
        g2,
        g3,

        in_feats,
        h_feats,
        out_feat,
        num_rels1,
        num_rels2,
        num_rels3,
        reg,
        num_bases,
    ):
        super(UnetGraphSAGE, self).__init__()
        self.conv1 = RelGraphConv(
            in_feats, int(h_feats / 4), num_rels1, regularizer=reg, num_bases=num_bases
        )
        self.conv2 = RelGraphConv(
            int(h_feats / 4),
            int(h_feats / 4),
            num_rels1,
            regularizer=reg,
            num_bases=num_bases,
        )
        self.conv3 = RelGraphConv(
            int(h_feats / 4),
            int(h_feats / 2),
            num_rels2,
            regularizer=reg,
            num_bases=num_bases,
        )
        self.conv33 = RelGraphConv(
            int(h_feats / 2),
            int(h_feats / 2),
            num_rels2,
            regularizer=reg,
            num_bases=num_bases,
        )

        self.conv4 = RelGraphConv(
            int(h_feats / 2),
            int(h_feats),
            num_rels3,
            regularizer=reg,
            num_bases=num_bases,
        )
        self.conv44 = RelGraphConv(
            int(h_feats),
            int(h_feats),
            num_rels3,
            regularizer=reg,
            num_bases=num_bases,
        )

        self.conv444 = RelGraphConv(
            int(h_feats),
            int(h_feats / 2),
            num_rels3,
            regularizer=reg,
            num_bases=num_bases,
        )

        self.conv5 = RelGraphConv(
            int(h_feats),
            int(h_feats / 2),
            num_rels2,
            regularizer=reg,
            num_bases=num_bases,
        )
        self.conv55 = RelGraphConv(
            int(h_feats / 2),
            int(h_feats / 2),
            num_rels2,
            regularizer=reg,
            num_bases=num_bases,
        )
        self.conv555 = RelGraphConv(
            int(h_feats / 2),
            int(h_feats / 4),
            num_rels2,
            regularizer=reg,
            num_bases=num_bases,
        )

        self.conv6 = RelGraphConv(
            int(h_feats / 2),
            int(h_feats / 4),
            num_rels1,
            regularizer=reg,
            num_bases=num_bases,
        )
        self.conv66 = RelGraphConv(
            int(h_feats / 4),
            int(h_feats / 4),
            num_rels1,
            regularizer=reg,
            num_bases=num_bases,
        )

        self.conv7 = RelGraphConv(
            int(h_feats / 4), out_feat, num_rels1, regularizer=reg, num_bases=num_bases
        )
        self.Maxpool = nn.MaxPool2d(
            (pooling_size, pooling_size), stride=(pooling_size, pooling_size)
        )
        self.Meanpool = nn.AvgPool2d(
            (pooling_size, pooling_size), stride=(pooling_size, pooling_size)
        )

        self.upsample1 = nn.ConvTranspose2d(
            int(h_feats / 2), int(h_feats / 2), 2, stride=2, padding=0
        )
        self.upsample2 = nn.ConvTranspose2d(
            int(h_feats / 4), int(h_feats / 4), 2, stride=2, padding=0
        )

        self.g1 = g1
        self.g2 = g2
        self.g3 = g3

        self.input_res = input_res
        self.pooling_size = pooling_size

    def forward(self, in_feat, etype1, etype2, etype3):

        h1 = F.relu(self.conv1(self.g1, in_feat, etype1))

        h22 = F.relu(self.conv2(self.g1, h1, etype1))
        h2 = h22.view(6, self.input_res, self.input_res, -1)
        h2 = torch.permute(h2, (3, 0, 1, 2))
        h2 = self.Meanpool(h2).view(
            -1,
            int(
                6
                * self.input_res
                / self.pooling_size
                * self.input_res
                / self.pooling_size
            ),
        )
        h2 = torch.transpose(h2, 0, 1)

        h3 = F.relu(self.conv3(self.g2, h2, etype2))
        h33 = F.relu(self.conv33(self.g2, h3, etype2))
        h3 = h33.view(
            6,
            int(self.input_res / self.pooling_size),
            int(self.input_res / self.pooling_size),
            -1,
        )
        h3 = torch.permute(h3, (3, 0, 1, 2))
        h3 = self.Meanpool(h3).view(
            -1,
            int(
                6
                * self.input_res
                / (self.pooling_size) ** 2
                * self.input_res
                / (self.pooling_size) ** 2
            ),
        )
        h3 = torch.transpose(h3, 0, 1)

        h6 = F.relu(self.conv4(self.g3, h3, etype3))
        h6 = F.relu(self.conv44(self.g3, h6, etype3))
        h6 = F.relu(self.conv444(self.g3, h6, etype3)).view(
            6,
            int(self.input_res / (self.pooling_size) ** 2),
            int(self.input_res / (self.pooling_size) ** 2),
            -1,
        )
        h6 = torch.permute(h6, (0, 3, 1, 2))
        h6 = self.upsample1(h6)
        h6 = torch.permute(h6, (1, 0, 2, 3)).reshape(
            -1,
            int(
                6
                * self.input_res
                / (self.pooling_size)
                * self.input_res
                / (self.pooling_size)
            ),
        )
        h6 = torch.transpose(h6, 0, 1)
        h6 = torch.cat((h6, h33), dim=1)

        h6 = F.relu(self.conv5(self.g2, h6, etype2))
        h6 = F.relu(self.conv55(self.g2, h6, etype2))
        h6 = F.relu(self.conv555(self.g2, h6, etype2)).view(
            6,
            int(self.input_res / (self.pooling_size)),
            int(self.input_res / (self.pooling_size)),
            -1,
        )
        h6 = torch.permute(h6, (0, 3, 1, 2))
        h6 = self.upsample2(h6)
        h6 = torch.permute(h6, (1, 0, 2, 3)).reshape(
            -1,
            int(
                6
                * self.input_res
                * self.input_res
            ),
        )

        h6 = torch.transpose(h6, 0, 1)
        h6 = torch.cat((h6, h22), dim=1)

        h6 = F.relu(self.conv6(self.g1, h6, etype1))
        h6 = F.relu(self.conv66(self.g1, h6, etype1))
        out = self.conv7(self.g1, h6, etype1)
        return out
