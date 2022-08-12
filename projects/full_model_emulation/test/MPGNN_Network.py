import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from dgl.nn.pytorch import NNConv


class MPNNGNN(nn.Module):
    """MPNN.

    This class performs message passing in MPNN and returns the updated node representations.

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
        g,
        node_in_feats,
        edge_in_feats,
        node_hidden_feats,
        edge_hidden_feats,
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
            aggregator_type="sum",
        )
        self.gru = nn.GRU(node_hidden_feats, node_hidden_feats)


        self.g = g

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, node_feats, edge_feats):
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
            node_feats = F.relu(self.gnn_layer(self.g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats


class MPGNN(nn.Module):
    def __init__(
        self,
        g,
        node_in_feats,
        edge_in_feats,
        node_hidden_feats,
        node_out_feats,
        edge_hidden_feats,
        num_step_message_passing,
        input_res,
        num_layers,
        pooling_size,
    ):
        super(MPGNN, self).__init__()
        self.num_layers=num_layers
        self.pooling_size=pooling_size
        self.input_res=input_res

        self.encoder_Res = nn.Sequential(
                nn.Linear(input_res, int(input_res/2)),
                nn.Linear(int(input_res/2), int(input_res/4)),
            )

        processor_layer=self.build_processor_model()
        self.inputlayer = processor_layer(g,
                                    node_in_feats,
                                    edge_in_feats,
                                    node_hidden_feats,
                                    edge_hidden_feats,
                                    num_step_message_passing)

        self.processor = nn.ModuleList([processor_layer(g,
                                    node_hidden_feats,
                                    edge_in_feats,
                                    node_hidden_feats,
                                    edge_hidden_feats,
                                    num_step_message_passing) for i in range(num_layers)])

        # self.processor[0]=processor_layer(g,
        #                             node_in_feats,
        #                             edge_in_feats,
        #                             node_hidden_feats,
        #                             edge_hidden_feats,
        #                             num_step_message_passing,
        #                             input_res)

        # for i in range(1,self.num_layers-1):
        #     self.processor[i]=processor_layer(g,
        #                             node_hidden_feats,
        #                             edge_in_feats,
        #                             node_hidden_feats,
        #                             edge_hidden_feats,
        #                             num_step_message_passing,
        #                             input_res)

        # decoder: only for node embeddings
        self.decoder = nn.Sequential(nn.Linear( node_hidden_feats ,int(node_hidden_feats/2)),
                                nn.ReLU(),
                                nn.Linear( int(node_hidden_feats/2), node_out_feats)
                                )

        self.decoder_Res = nn.Sequential(
            nn.Linear(int(input_res/4), int(input_res/2)),
            nn.Linear(int(input_res/2), int(input_res)),
        )

    def build_processor_model(self):
        return MPNNGNN




    def forward(self, node_feats, edge_feats):
        
        node_feats = node_feats.view(
            6,
            int(self.input_res),
            int(self.input_res),
            -1,
        )
        node_feats = torch.permute(node_feats, (3, 0, 1, 2))
        node_feats=self.encoder_Res(node_feats)
        node_feats=node_feats.transpose(2,3)
        node_feats=self.encoder_Res(node_feats)
        node_feats=node_feats.transpose(2,3).reshape(
            -1,
            int(
                6
                * self.input_res
                / self.pooling_size**2
                * self.input_res
                / self.pooling_size**2
            ),
        )

        node_feats = torch.transpose(node_feats, 0, 1)
        node_feats=self.inputlayer(node_feats, edge_feats)
        for i in range(self.num_layers):
            node_feats = self.processor[i](node_feats, edge_feats)


        node_feats = node_feats.view(
            6,
            int(self.input_res/self.pooling_size**2),
            int(self.input_res/self.pooling_size**2),
            -1,
        )
        node_feats = torch.permute(node_feats, (3, 0, 1, 2))
        node_feats=self.decoder_Res(node_feats)
        node_feats=node_feats.transpose(2,3)
        node_feats=self.decoder_Res(node_feats)
        node_feats=node_feats.transpose(2,3).reshape(
            -1,
            int(
                6
                * self.input_res
                * self.input_res
            ),
        )
        node_feats=node_feats.transpose(0,1)
        out=self.decoder(node_feats)
        return out





