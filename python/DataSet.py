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
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Embedding
from torch.utils.data import DataLoader

from torch_sparse import coalesce
from torch_scatter import scatter_min
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, to_networkx, 
                                   to_scipy_sparse_matrix, to_undirected)
import glob
from torch_geometric.data.makedirs import makedirs

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)

from enclosing_subgraph import *

class WLDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops,percent=100, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        super(WLDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'WLGNN_{}_data'.format(self.split)
        else:
            name = 'WLGNN_{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes, op="mean")

        self.data.edge_weight = torch.round(self.data.edge_weight)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        
        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, "pos", self.num_hops, 
            self.node_label, self.ratio_per_hop, self.max_nodes_per_hop)
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, "neg", self.num_hops, 
            self.node_label, self.ratio_per_hop, self.max_nodes_per_hop)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


class WLDynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        processed_dir = osp.join(root, "processed")
        #print(self.datalist)
        self.split = split

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)
        self.datalist = ['data_{}_{}.pt'.format(i, self.split) for i in range(len(self.links))]
        
        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        super(WLDynamicDataset, self).__init__(root)
    
    @property
    def processed_file_names(self):
        return self.datalist
 
    def __len__(self):
        return len(self.datalist)
    def _process(self):
        makedirs(self.processed_dir)
        if self.split == 'train' and len(glob.glob(osp.join(self.processed_dir, '*train.pt'))) > 0:
            return
        if self.split == 'test' and len(glob.glob(osp.join(self.processed_dir, '*test.pt'))) > 0:
            return
        if self.split == 'valid' and len(glob.glob(osp.join(self.processed_dir, '*valid.pt'))) > 0:
            return

        self.process()

    def process(self):
        for idx in tqdm(range(len(self.links))):
            src, dst = self.links[idx]

            if self.labels[idx]: status = "pos"
            else: status = "neg"

            tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, status, self.ratio_per_hop, 
                                 self.max_nodes_per_hop, node_features=self.data.x)
            data = construct_pyg_graph(*tmp, self.node_label)

            torch.save(data, osp.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, self.split)))
            #self.datalist.append('data_{}_{}.pt'.format(idx, self.split))
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, self.split)))
        return data  


def get_pos_neg_edges(split, split_edge, edge_index=None, num_nodes=None, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])
    
    elif 'head' in split_edge['train']: 
        source = split_edge[split]['head']
        target = split_edge[split]['tail']
        pos_edges = torch.stack([source, target])

        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        pos_edge = pos_edges[:, perm]


        if split == 'train':
            neg_edge = negative_sampling(
                pos_edges, num_nodes=num_nodes,
                num_neg_samples=source.size(0))
            #target_neg_head = torch.randint(0, num_nodes, [target.size(0), 1],
                                       #dtype=torch.long)
            #target_neg_tail = torch.randint(0, num_nodes, [target.size(0), 1],
                                       #dtype=torch.long)

            np.random.seed(123)
            num_neg = neg_edge.size(1)
            perm = np.random.permutation(num_neg)
            perm = perm[:int(percent / 100 * num_neg)]
            neg_edge = neg_edge[:, perm]
        else:
            target_neg_head = split_edge[split]['head_neg']
            target_neg_tail = split_edge[split]['tail_neg']

            source, target, target_neg_head, target_neg_tail = source[perm], target[perm], target_neg_head[perm, :], target_neg_tail[perm, :]
            neg_edge = torch.stack([target_neg_head.view(-1), target_neg_tail.view(-1)])
        #neg_per_target = target_neg.size(1)
        #neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                #target_neg.view(-1)])
    else: 
        print('Unrecognized dataset')
        sys.exit()


    return pos_edge, neg_edge


