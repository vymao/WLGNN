import argparse
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from DIGCNConv import DIGCNConv
from torch_scatter import scatter_add
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from torch.nn import Linear
from train_eval import run

class InceptionBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock, self).__init__()
        self.ln = Linear(in_dim, out_dim)
        self.conv1 = DIGCNConv(in_dim, out_dim)
        self.conv2 = DIGCNConv(in_dim, out_dim)
    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, x, edge_index, edge_weight, edge_index2, edge_weight2):
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        x2 = self.conv2(x, edge_index2, edge_weight2)
        return x0, x1, x2

class Sparse_Three_Sum(torch.nn.Module):
    def __init__(self, train_dataset, feature_dim, hidden_dim, num_classes, dropout, k=0.6):
        super(Sparse_Three_Sum, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if k <= 1:  # Transform percentile to number.
            sampled_train = train_dataset[:1000]
            num_nodes = sorted([g.num_nodes for g in sampled_train])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.ib1 = InceptionBlock(feature_dim, hidden_dim)
        self.ib2 = InceptionBlock(hidden_dim, hidden_dim)
        self.ib3 = InceptionBlock(hidden_dim, hidden_dim)
        self.final = InceptionBlock(hidden_dim, 1)

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_dim + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, num_classes)

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index2, edge_weight2 = data.edge_index2, data.edge_weight2
        
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self.dropout, training=self.training)

        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self.dropout, training=self.training)

        x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = x0+x1+x2

         # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)                           
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x

class Sparse_Three_Concat(torch.nn.Module):
    def __init__(self, train_dataset, feature_dim, hidden_dim, num_classes, dropout, k=0.6):
        super(Sparse_Three_Concat, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if k <= 1:  # Transform percentile to number.
            sampled_train = train_dataset[:1000]
            num_nodes = sorted([g.num_nodes for g in sampled_train])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.ib1 = InceptionBlock(feature_dims, hidden_dim)
        self.ib2 = InceptionBlock(hidden_dim, hidden_dim)
        self.ib3 = InceptionBlock(hidden_dim, hidden_dim)
        
        self.ln1 = Linear(hidden_dim * 3, hidden_dim)
        self.ln2 = Linear(hidden_dim * 3, hidden_dim)
        self.ln3 = Linear(hidden_dim * 3, hidden_dim)

        self.final = InceptionBlock(hidden_dim, 1)

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_dim + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, num_classes)

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.ln3.reset_parameters()


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index2, edge_weight2 = data.edge_index2, data.edge_weight2
        
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropoutt, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = torch.cat((x0,x1,x2),1)
        x = self.ln1(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)

        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = torch.cat((x0,x1,x2),1)
        x = self.ln2(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
       
        x = torch.cat((x0,x1,x2),1)
        x = self.ln3(x)

        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)                           
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x


