# Implementation of the message passing graph neural network according to the article in:
# https://arxiv.org/pdf/1704.01212.pdf

import numpy as np
import torch as th
import dgl
from dgl.nn import NNConv, GraphormerLayer
from dgl.nn import nn

class Graphormer(th.nn.Module):
    def __init__(self, gnn_layers, num_feats, n_classes, hidden, num_edge_feats, activation, num_heads, final_activation, dropout):
        super(Graphormer, self).__init__()
        self._gnn_layers = gnn_layers
        self._num_feats = num_feats
        self._n_classes = n_classes
        self._num_hiden_features = hidden
        self.activation = activation
        self._num_edge_feats = num_edge_feats
        self._final_activation = final_activation
        self._num_heads = num_heads
        self.dropout = dropout

        print(f'GRAPHORMER {num_feats=}')
        print(f'GRAPHORMER {n_classes=}')
        print(f'GRAPHORMER {hidden=}')
        print(f'GRAPHORMER {num_heads=}')
        self.build_model()

          

    def build_model(self):
        self.linear_layers = []
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for i in range(self._gnn_layers - 2):
            print('i: ', i)
            h2h = self.build_hidden_layer(i)
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def build_input_layer(self):
        print('Building an INPUT  layer of {}x{}'.format(self._num_feats, self._num_hiden_features[0]))
        self.linear_layers.append(th.nn.Linear(self._num_feats, self._num_hiden_features[0]))
        return GraphormerLayer(self._num_feats, self._num_feats, self._num_heads, dropout=self.dropout, activation=self.activation)
        # return GraphormerLayer(self._num_feats, self._num_hiden_features[0], self._num_heads, self.dropout, activation=self.activation)

    def build_hidden_layer(self, i):
        print('Building an HIDDEN  layer of {}x{}'.format(self._num_hiden_features[i], self._num_hiden_features[i+1]))
        self.linear_layers.append(th.nn.Linear(self._num_hiden_features[i], self._num_hiden_features[i+1]))
        return GraphormerLayer(self._num_hiden_features[i], self._num_hiden_features[i], self._num_heads, dropout=self.dropout, activation=self.activation)

    def build_output_layer(self):
        print('Building an OUTPUT  layer of {}x{}'.format(self._num_hiden_features[-1], self._n_classes))
        self.linear_layers.append(th.nn.Linear(self._num_hiden_features[-1], self._n_classes))
        return GraphormerLayer(self._num_hiden_features[-1], self._num_hiden_features[-1], self._num_heads, dropout=self.dropout, activation=self._final_activation)


    @staticmethod
    def edge_function(f_in, f_out):
        a = int(f_in*0.666 + f_out*0.334)
        b = int(f_in*0.334 + f_out*0.666)
        return th.nn.Sequential(
            th.nn.Linear(f_in, a),
            th.nn.ReLU(),
            th.nn.Linear(a, b),
            th.nn.ReLU(),
            th.nn.Linear(b, f_out)
        )


    def set_g(self, g):
        self.g = g
        for l in range(self._gnn_layers):
            self.layers[l].g = g

    def forward(self, graph, feat, efeat):
        self.set_g(graph)

        # g: DGL Graph
        # features: Node features (tensor)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)  # Add batch dimension

        x = feat

        # print(f'input: {x.shape}')
        for idx, layer in enumerate(self.layers):
            # print(f'layer:{idx} input: {x.shape}')
            x = layer(x)
            # print(f'layer:{idx} temp: {x.shape}')
            x = self.linear_layers[idx](x)
            # print(f'layer:{idx} output: {x.shape}')

        # if self._final_activation is not None:
        #     logits = self._final_activation(x)
        # else:
        #     logits = x

        return x
