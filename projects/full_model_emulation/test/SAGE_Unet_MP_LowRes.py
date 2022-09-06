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

    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        node_hidden_feats,
        edge_hidden_feats,
        node_out_feats,
        num_step_message_passing,
    ):
        super(MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hidden_feats),
            nn.ReLU(),
            nn.Linear(node_hidden_feats, node_hidden_feats),
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_hidden_feats * node_hidden_feats),
        )
        self.gnn_layer = NNConv(
            in_feats=node_hidden_feats,
            out_feats=node_hidden_feats,
            edge_func=edge_network,
            aggregator_type="mean",
        )
        self.gru = nn.GRU(node_hidden_feats, node_hidden_feats)

        self.decoder = nn.Sequential(
            nn.Linear(node_hidden_feats, node_hidden_feats),
            nn.ReLU(),
            nn.Linear(node_hidden_feats, node_out_feats),
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats,outputLayer):
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
        node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            # print(node_feats.shape)
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
            if outputLayer==1:
                out=self.decoder(node_feats)
            else:
                out=node_feats
        return out


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
        edge_in_feats,
        edge_h_feats,
        num_step_message_passing,
    ):
        super(UnetGraphSAGE, self).__init__()

        self.MP1 = MPNNGNN(
            int(in_feats),
            edge_in_feats,
            32,
            edge_h_feats,
            32,
            num_step_message_passing,
        )


        self.MP2 = MPNNGNN(
            32,
            edge_in_feats,
            64,
            edge_h_feats,
            64,
            num_step_message_passing,
        )

        self.MP3 = MPNNGNN(
            64,
            edge_in_feats,
            128,
            edge_h_feats,
            128,
            num_step_message_passing,
        )


        self.MP4 = MPNNGNN(
            192,
            edge_in_feats,
            98,
            edge_h_feats,
            98,
            num_step_message_passing,
        )

        self.MP5 = MPNNGNN(
            130,
            edge_in_feats,
            60,
            edge_h_feats,
            60,
            num_step_message_passing,
        )

        self.MP6 = MPNNGNN(
            67,
            edge_in_feats,
            32,
            edge_h_feats,
            out_feat,
            num_step_message_passing,
        )
       
        self.Maxpool = nn.MaxPool2d(
            (pooling_size, pooling_size), stride=(pooling_size, pooling_size)
        )
        self.Meanpool = nn.AvgPool2d(
            (pooling_size, pooling_size), stride=(pooling_size, pooling_size)
        )

        self.upsample1 = nn.ConvTranspose2d(
            128,128, 2, stride=2, padding=0
        )
        self.upsample2 = nn.ConvTranspose2d(
            98, 98, 2, stride=2, padding=0
        )
        self.upsample3 = nn.ConvTranspose2d(
            60, 60, 2, stride=2, padding=0
        )
        self.upsample4 = nn.ConvTranspose2d(
            60, 60, 2, stride=2, padding=0
        )

        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.g4 = g4
        self.g5 = g5

        self.input_res = input_res
        self.pooling_size = pooling_size

    def forward(self, in_feat, edge1,edge2,edge3, edge4, edge5):
        
        h1 = in_feat.view(
            6,
            int(self.input_res),
            int(self.input_res),
            -1,
        )
        h1 = torch.permute(h1, (3, 0, 1, 2))

        h1 = self.Meanpool(h1)
        
        h2 = self.Meanpool(h1).view(
            -1,
            int(
                6
                * self.input_res
                / self.pooling_size**2
                * self.input_res
                / self.pooling_size**2
            ),
        )

        h2 = torch.transpose(h2, 0, 1)
        h2 = self.MP1(self.g3, h2, edge3,0)

        h3 = h2.view(
            6,
            int(self.input_res / self.pooling_size**2),
            int(self.input_res / self.pooling_size**2),
            -1,
        )

        h3 = torch.permute(h3, (3, 0, 1, 2))
        h3 = self.Meanpool(h3).view(
            -1,
            int(
                6
                * self.input_res
                / (self.pooling_size) ** 3
                * self.input_res
                / (self.pooling_size) ** 3
            ),
        )
        
        h3 = torch.transpose(h3, 0, 1)
        h3 = self.MP2(self.g4, h3, edge4,0)
        h33 = h3.view(
            6,
            int(self.input_res / (self.pooling_size) ** 2),
            int(self.input_res / (self.pooling_size) ** 2),
            -1,
        )


        h4 = torch.permute(h33, (3, 0, 1, 2))
        h4 = self.Meanpool(h4).view(
            -1,
            int(
                6
                * self.input_res
                / (self.pooling_size) ** 4
                * self.input_res
                / (self.pooling_size) ** 4
            ),
        )
        h4 = torch.transpose(h4, 0, 1)
        h4 = self.MP3(self.g5, h4, edge5,0)
        h44 = h4.view(
            6,
            int(self.input_res / (self.pooling_size) **4),
            int(self.input_res / (self.pooling_size) ** 4),
            -1,
        )


        h6 = torch.permute(h44, (0, 3, 1, 2))
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
        h6 = torch.cat((h6, h3), dim=1)
        h6 = self.MP4(self.g4, h6, edge4,0).view(
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
        h6 = torch.cat((h6, h2), dim=1)

        h6 = self.MP5(self.g3, h6, edge3,0).view(
            6,
            int(self.input_res / (self.pooling_size) ** 2),
            int(self.input_res / (self.pooling_size) ** 2),
            -1,
        )
        h6 = torch.permute(h6, (0, 3, 1, 2))
        h6 = self.upsample3(h6)

        h6 = self.upsample4(h6)         
        h6 = torch.permute(h6, (1, 0, 2, 3)).reshape(
            -1, int(6 * self.input_res * self.input_res)
        )
        h6 = torch.transpose(h6, 0, 1)
        h6 = torch.cat((h6, in_feat), dim=1)
        out = self.MP6(self.g1, h6, edge1,1)
        return out
