import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from dgl.nn.pytorch import NNConv

class MPNNGNN(nn.Module):
    """MPNN.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_in_feats : int
        Size for the input edge features. Default to 128.
    edge_hidden_feats : int
        Size for the hidden edge representations.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    """
    def __init__(self, node_in_feats, edge_in_feats, node_hidden_feats,
                 edge_hidden_feats, node_out_feats, num_step_message_passing):
        super(MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hidden_feats),
            nn.ReLU(),
            nn.Linear(node_hidden_feats, node_hidden_feats)
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_hidden_feats * node_hidden_feats)
        )
        self.gnn_layer = NNConv(
            in_feats=node_hidden_feats,
            out_feats=node_hidden_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(node_hidden_feats, node_hidden_feats)

        self.decoder = nn.Sequential(nn.Linear( node_hidden_feats , node_hidden_feats),
                              nn.ReLU(),
                              nn.Linear( node_hidden_feats, node_out_feats)
                              )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()


    def forward(self, g , node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        node_feats : float32 tensor of shape (V, node_out_feats)
            Output node representations.
        """
        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            # print(node_feats.shape)
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return self.decoder(node_feats)


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
        num_step,
        aggregat,
        edge_in_feats,
        num_step_message_passing,
    ):
        super(UnetGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, int(h_feats / 16), aggregat)
        self.conv2 = SAGEConv(int(h_feats / 16), int(h_feats / 16), aggregat)
        self.conv3 = SAGEConv(int(h_feats / 16), int(h_feats / 8), aggregat)
        self.conv33 = SAGEConv(int(h_feats / 8), int(h_feats / 8), aggregat)

        self.conv4 = SAGEConv(int(h_feats / 8), int(h_feats / 4), aggregat)
        self.conv44 = SAGEConv(int(h_feats / 4), int(h_feats / 4), aggregat)

        self.MP1=MPNNGNN(int(h_feats / 4), edge_in_feats, int(h_feats / 4), int(h_feats / 4), int(h_feats / 2) ,num_step_message_passing)

        self.conv5 = SAGEConv(int(h_feats / 4), int(h_feats / 2), aggregat)
        self.conv55 = SAGEConv(int(h_feats / 2), int(h_feats / 2), aggregat)

        self.MP2=MPNNGNN(int(h_feats / 2), edge_in_feats, int(h_feats / 2), int(h_feats / 2), int(h_feats) ,num_step_message_passing)

        self.conv6 = SAGEConv(int(h_feats / 2), int(h_feats), aggregat)
        self.conv66 = SAGEConv(int(h_feats), int(h_feats), aggregat)

        self.MP3=MPNNGNN(int(h_feats), edge_in_feats, int(h_feats), int(h_feats), int(h_feats / 2) ,num_step_message_passing)

        self.conv7 = SAGEConv(int(h_feats), int(h_feats / 2), aggregat)
        self.conv77 = SAGEConv(int(h_feats / 2), int(h_feats / 2), aggregat)

        self.MP4=MPNNGNN(int(h_feats / 2), edge_in_feats, int(h_feats / 2), int(h_feats / 2),  int(h_feats / 4) ,num_step_message_passing)

        self.conv8 = SAGEConv(int(h_feats / 2), int(h_feats / 4), aggregat)
        self.conv88 = SAGEConv(int(h_feats / 4), int(h_feats / 4), aggregat)

        self.conv9 = SAGEConv(int(h_feats / 4), int(h_feats / 8), aggregat)
        self.conv99 = SAGEConv(int(h_feats / 8), int(h_feats / 8), aggregat)

        self.conv10 = SAGEConv(int(h_feats / 8), int(h_feats / 16), aggregat)
        self.conv101 = SAGEConv(int(h_feats / 16), int(h_feats / 16), aggregat)

        self.conv11 = SAGEConv(int(h_feats / 16), out_feat, aggregat)
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

        self.num_step = num_step
        self.input_res = input_res
        self.pooling_size = pooling_size


    def forward(self, in_feat,edge3,edge4,edge5):

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


        h5=self.MP1(self.g4, h4, edge4)
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

        h6=self.MP2(self.g5, h5,edge5)
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

        h6=self.MP3(self.g4, h6,edge4)
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

        h6=self.MP4(self.g3, h6,edge3)
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

        