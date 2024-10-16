import torch
import torch.nn as nn
from nets.rgcnDGL import RGCN
from nets.gat import GAT
from nets.mpnn_dgl import MPNN
from nets.graphormer import Graphormer
import dgl

class SELECT_GNN(nn.Module):
    def __init__(self, num_features, num_edge_feats, n_classes, num_hidden, gnn_layers, dropout,
                 activation, final_activation, gnn_type, num_heads, num_rels, num_bases, g, residual,
                 aggregator_type, attn_drop, concat=True, bias=True, norm=None, alpha=0.12):
        super(SELECT_GNN, self).__init__()

        self.activation = activation
        self.gnn_type = gnn_type
        if final_activation == 'relu':
            self.final_activation = torch.nn.ReLU()
        elif final_activation == 'tanh':
            self.final_activation = torch.nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

        self.attn_drop = attn_drop
        self.num_rels = num_rels
        self.residual = residual
        self.aggregator = aggregator_type
        self.num_bases = num_bases
        self.n_classes = n_classes
        self.num_hidden = num_hidden
        self.gnn_layers = gnn_layers
        self.num_features = num_features
        self.num_edge_feats = num_edge_feats
        self.dropout = dropout
        self.bias = bias
        self.norm = norm
        self.g = g
        self.num_heads = num_heads
        self.concat = concat
        self.alpha = alpha
        # self.classifier = nn.Linear(42, self.n_classes)

        if self.gnn_type == 'rgcn':
            # print("GNN being used is RGCN")
            self.gnn_object = self.rgcn()
        elif self.gnn_type == 'gat':
            # print("GNN being used is GAT")
            self.gnn_object = self.gat()
        elif self.gnn_type == 'mpnn':
            # print("GNN being used is MPNN")
            self.gnn_object = self.mpnn()
        elif self.gnn_type == 'graphormer':
            # print("GNN being used is Graphormer")
            self.gnn_object = self.graphormer()
        elif self.gnn_type == 'gat':
            # print("GNN being used is Gat")
            self.gnn_object = self.gat()

    def rgcn(self):
        return RGCN(self.g, self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_rels,
                    self.activation, self.final_activation, self.dropout, self.num_bases)

    def gat(self):
        return GAT(self.g, self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_heads,
                   self.activation, self.final_activation,  self.dropout, self.attn_drop, self.alpha, self.residual)

    def mpnn(self):
        return MPNN(self.num_features, self.n_classes, self.num_hidden, self.num_edge_feats, self.final_activation,
                    self.aggregator, self.bias, self.residual, self.norm, self.activation)

    def graphormer(self):
        print("Going to Graphormer")
        return Graphormer(self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_edge_feats, self.activation, self.num_heads, self.final_activation, self.dropout)
        # return Graphormer()

    def gat(self):
        return GAT(self.g, self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_heads, self.final_activation, self.activation, self.dropout, self.attn_drop, self.alpha, self.residual)


    def forward(self, data, g, efeat):
        if self.gnn_type == 'mpnn' and efeat is not None:
            x = self.gnn_object(g, data, efeat)
        else:
            # x = self.gnn_object(data, g, efeat)
            x = self.gnn_object(g, data, efeat)
        logits = x
        base_index = 0
        batch_number = 0
        unbatched = dgl.unbatch(self.g)
        output = torch.Tensor(size=(len(unbatched), 2))
        for g in unbatched:
            num_nodes = g.number_of_nodes()
            #output[batch_number, :] = logits[0, base_index, :]  # Output is just the room's node
            output[batch_number, :] = logits[batch_number, :]
            base_index += num_nodes
            batch_number += 1
        print(batch_number)
        return output




# output.shape=torch.Size([31, 2])
# logits.shape=torch.Size([1, 5793, 42])
# batch_number=0
# base_index=0