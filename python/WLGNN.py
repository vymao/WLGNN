import argparse
import math
import time
import random
import os, sys
import os.path as osp
from itertools import chain
from shutil import copy
import copy as cp
from tqdm import tqdm
import pdb

import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Embedding
from torch.utils.data import DataLoader

from torch_sparse import coalesce
from torch_scatter import scatter_min
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, global_sort_pool, global_add_pool
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, to_networkx, 
                                   to_scipy_sparse_matrix, to_undirected)

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class WLGNN_model(torch.nn.Module):
    def __init__(self, args, train_dataset, dataset, hidden_channels, num_layers, max_z, GNN=GCNConv, k=0.6, 
                 use_feature=False, node_embedding=None):
        super(WLGNN_model, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.args = args

        if k <= 1:  # Transform percentile to number.
            if args.dynamic_train:
                sampled_train = train_dataset[:1000]
            else:
                sampled_train = train_dataset
            num_nodes = sorted([g.num_nodes for g in sampled_train])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.w_embedding = Embedding(self.max_z, hidden_channels)
        self.z1_embedding = Embedding(self.max_z, hidden_channels)
        self.z2_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels * 3
        if self.use_feature:
            initial_channels += dataset.num_features * 2
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for i in range(0, num_layers-1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, data):
        x = data.x if self.args.use_feature else None
        z1, z2, w, edge_index, batch, x, edge_weight, node_id = data.z1, data.z2, data.w, data.edge_index, data.batch, x, data.edge_weight, None
        if (edge_index.size()[0] != 2): 
            print(data)
            print(data.x)
            print(data.y)
            print(data.w)
            print(data.z1)
            print(data.z2)
            print(data.node_id)
        z1_emb = self.z1_embedding(z1)
        z2_emb = self.z2_embedding(z2)
        w_emb = self.w_embedding(w)

        z_emb = torch.cat([w_emb, z1_emb], 1)
        z_emb = torch.cat([z_emb, z2_emb], 1)

        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

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

