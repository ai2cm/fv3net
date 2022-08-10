import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)

class UnetGraphSAGE(nn.Module):
    def __init__(
        self,
        input_res,
        pooling_size,
        g1,
        g2,
        g3,
        g4,
        g5,
        in_feats,
        h_feats,
        out_feat,
        num_rels,
        num_bases,
    ):
        super(UnetGraphSAGE, self).__init__()
        self.conv1=RGCNLayer(in_feats,int(h_feats / 16), num_rels, num_bases,
                         activation=F.relu, is_input_layer=True)

        self.conv2=RGCNLayer(int(h_feats / 16), int(h_feats / 16), num_rels, num_bases,
                         activation=F.relu)

        self.conv3=RGCNLayer(int(h_feats / 16), int(h_feats / 8), num_rels, num_bases,
                         activation=F.relu)
        
        self.conv33=RGCNLayer(int(h_feats / 8), int(h_feats / 8), num_rels, num_bases,
                         activation=F.relu)
        
        self.conv4=RGCNLayer(int(h_feats / 8), int(h_feats / 4), num_rels, num_bases,
                         activation=F.relu)

        self.conv44=RGCNLayer(int(h_feats / 4), int(h_feats / 4), num_rels, num_bases,
                         activation=F.relu)
        
        self.conv5=RGCNLayer(int(h_feats / 4), int(h_feats / 2), num_rels, num_bases,
                         activation=F.relu)

        self.conv55=RGCNLayer(int(h_feats / 2), int(h_feats / 2), num_rels, num_bases,
                         activation=F.relu)

        self.conv6=RGCNLayer(int(h_feats / 2), int(h_feats), num_rels, num_bases,
                         activation=F.relu)
        
        self.conv66 = RGCNLayer(int(h_feats), int(h_feats), num_rels, num_bases,
                         activation=F.relu)

        self.conv7 = RGCNLayer(int(h_feats), int(h_feats / 2), num_rels, num_bases,
                         activation=F.relu)

        self.conv77 = RGCNLayer(int(h_feats / 2), int(h_feats / 2), num_rels, num_bases,
                         activation=F.relu)

        self.conv8 = RGCNLayer(int(h_feats / 2), int(h_feats / 4), num_rels, num_bases,
                         activation=F.relu)

        self.conv88 = RGCNLayer(int(h_feats / 4), int(h_feats / 4), num_rels, num_bases,
                         activation=F.relu)

        self.conv9 = RGCNLayer(int(h_feats / 4), int(h_feats / 8), num_rels, num_bases,
                         activation=F.relu)

        self.conv99 = RGCNLayer(int(h_feats / 8), int(h_feats / 8), num_rels, num_bases,
                         activation=F.relu)

        self.conv10 = RGCNLayer(int(h_feats / 8), int(h_feats / 16), num_rels, num_bases,
                         activation=F.relu)

        self.conv101 = RGCNLayer(int(h_feats / 16), int(h_feats / 16), num_rels, num_bases,
                         activation=F.relu)

        self.conv11 = RGCNLayer(int(h_feats / 16), out_feat, num_rels, num_bases,
                         activation=F.relu)
    
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
        self.upsample3 = nn.ConvTranspose2d(
            int(h_feats / 8), int(h_feats / 8), 2, stride=2, padding=0
        )
        self.upsample4 = nn.ConvTranspose2d(
            int(h_feats / 16), int(h_feats / 16), 2, stride=2, padding=0
        )

        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.g4 = g4
        self.g5 = g5

        self.input_res = input_res
        self.pooling_size = pooling_size

    def forward(self, in_feat):

        h1 = F.relu(self.conv1(self.g1, in_feat))

        h22 = F.relu(self.conv2(self.g1, h1))
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
        # g2=self.get_graph(24)

        h3 = F.relu(self.conv3(self.g2, h2))
        h33 = F.relu(self.conv33(self.g2, h3))
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
        # g3=self.get_graph(self.input_res/(self.pooling_size)**2)

        h4 = F.relu(self.conv4(self.g3, h3))
        h44 = F.relu(self.conv44(self.g3, h4))
        h4 = h44.view(
            6,
            int(self.input_res / (self.pooling_size) ** 2),
            int(self.input_res / (self.pooling_size) ** 2),
            -1,
        )
        h4 = torch.permute(h4, (3, 0, 1, 2))
        h4 = self.Meanpool(h4).view(
            -1,
            int(
                6
                * self.input_res
                / (self.pooling_size) ** 3
                * self.input_res
                / (self.pooling_size) ** 3
            ),
        )
        h4 = torch.transpose(h4, 0, 1)
        # g4=self.get_graph(self.input_res/(self.pooling_size)**3)

        h5 = F.relu(self.conv5(self.g4, h4))
        h55 = F.relu(self.conv55(self.g4, h5))
        h5 = h55.view(
            6,
            int(self.input_res / (self.pooling_size) ** 3),
            int(self.input_res / (self.pooling_size) ** 3),
            -1,
        )
        h5 = torch.permute(h5, (3, 0, 1, 2))
        h5 = self.Meanpool(h5).view(
            -1,
            int(
                6
                * self.input_res
                / (self.pooling_size) ** 4
                * self.input_res
                / (self.pooling_size) ** 4
            ),
        )
        h5 = torch.transpose(h5, 0, 1)

        h6 = F.relu(self.conv6(self.g5, h5))
        h6 = F.relu(self.conv66(self.g5, h6))
        h6 = F.relu(self.conv7(self.g5, h6)).view(
            6,
            int(self.input_res / (self.pooling_size) ** 4),
            int(self.input_res / (self.pooling_size) ** 4),
            -1,
        )
        h6 = torch.permute(h6, (0, 3, 1, 2))
        h6 = self.upsample1(h6)
        h6 = torch.permute(h6, (1, 0, 2, 3)).reshape(
            -1,
            int(
                6
                * self.input_res
                / (self.pooling_size) ** 3
                * self.input_res
                / (self.pooling_size) ** 3
            ),
        )
        h6 = torch.transpose(h6, 0, 1)
        h6 = torch.cat((h6, h55), dim=1)

        h6 = F.relu(self.conv7(self.g4, h6))
        h6 = F.relu(self.conv77(self.g4, h6))
        h6 = F.relu(self.conv8(self.g4, h6)).view(
            6,
            int(self.input_res / (self.pooling_size) ** 3),
            int(self.input_res / (self.pooling_size) ** 3),
            -1,
        )
        h6 = torch.permute(h6, (0, 3, 1, 2))
        h6 = self.upsample2(h6)
        h6 = torch.permute(h6, (1, 0, 2, 3)).reshape(
            -1,
            int(
                6
                * self.input_res
                / (self.pooling_size) ** 2
                * self.input_res
                / (self.pooling_size) ** 2
            ),
        )
        h6 = torch.transpose(h6, 0, 1)
        h6 = torch.cat((h6, h44), dim=1)

        h6 = F.relu(self.conv8(self.g3, h6))
        h6 = F.relu(self.conv88(self.g3, h6))
        h6 = F.relu(self.conv9(self.g3, h6)).view(
            6,
            int(self.input_res / (self.pooling_size) ** 2),
            int(self.input_res / (self.pooling_size) ** 2),
            -1,
        )
        h6 = torch.permute(h6, (0, 3, 1, 2))
        h6 = self.upsample3(h6)
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

        h6 = F.relu(self.conv9(self.g2, h6))
        h6 = F.relu(self.conv99(self.g2, h6))
        h6 = F.relu(self.conv10(self.g2, h6)).view(
            6,
            int(self.input_res / (self.pooling_size)),
            int(self.input_res / (self.pooling_size)),
            -1,
        )
        h6 = torch.permute(h6, (0, 3, 1, 2))
        h6 = self.upsample4(h6)
        h6 = torch.permute(h6, (1, 0, 2, 3)).reshape(
            -1, int(6 * self.input_res * self.input_res)
        )
        h6 = torch.transpose(h6, 0, 1)
        h6 = torch.cat((h6, h22), dim=1)

        h6 = F.relu(self.conv10(self.g1, h6))
        h6 = F.relu(self.conv101(self.g1, h6))
        out = self.conv11(self.g1, h6)
        return out
