import argparse
import torch
import os
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F
from DIGCNConv import DIGCNConv
from torch_scatter import scatter_add
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx

import scipy.sparse as ssp
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Embedding

from torch_sparse import coalesce
from torch_scatter import scatter_min
import torch_geometric.transforms as T
from torch_geometric.nn import global_sort_pool, global_add_pool
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, to_networkx,
                                   to_scipy_sparse_matrix, to_undirected)



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

        self.class_embedding = Embedding(10, feature_dim)
        self.z_embedding = Embedding(1000, feature_dim)
        self.word2idx = {
            "disease": 1,
            "function": 2,
            "drug": 3,
            "sideeffect": 4,
            "protein": 5
        }
 
        input_dim = 4 * feature_dim + 53
        self.ib1 = InceptionBlock(input_dim, hidden_dim)
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
        x, edge_index, edge_weight, node_weights = data.x, data.edge_index, data.edge_weight, data.node_weights
        edge_index2, edge_weight2 = data.edge_index2, data.edge_weight2

        src_class = class_embedding(node_weights[:,0])
        dst_class = class_emebedding(node_weights[:, 1])
        src_z = z_embedding(node_weights[:, 2])
        dst_z = z_embedding(node_weights[:, 3])

        #src, end = node_class.t()
        #src, end = embedding(src), embedding(end)
        x = torch.cat([x, src_class, dist_class, src_z, dst_z], 1)
        
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


