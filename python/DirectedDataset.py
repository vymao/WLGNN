import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
import torch
import sys
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected

from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add
import scipy
from torch_geometric.data import InMemoryDataset, Dataset
from get_adj import get_undirected_adj,get_pr_directed_adj,get_appr_directed_adj,get_second_directed_adj
from DataSet import *
from enclosing_subgraph import *

class Directed_Dataset(Dataset):
    r"""
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"cora_ml"`,
            :obj:`"citeseer"`, :obj:`"pubmed"`), :obj:`"amazon_computer", :obj:`"amazon_photo", :obj:`"cora_full"`) .
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, data, split_edge, num_nodes, alpha, num_hops, percent, input_dim = 16, 
                    split='train', ratio_per_hop=1.0, max_nodes_per_hop=None, adj_type=None, transform=None, pre_transform=None):
        self.data = data
        self.split_edge = split_edge
        self.alpha = alpha
        self.adj_type = adj_type
        self.num_nodes = num_nodes
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = splitc
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop

        max_z = 100
        self.embedding = Embedding(self.max_z, input_dim)
        word2idx = {
            "disease": 1, 
            "function": 2, 
            "drug": 3, 
            "sideeffect": 4, 
            "protein": 5
        }

        node_type= list(range(num_nodes))
        for edge in range(len(data['relation'])): 
            node_type[data['head'][edge]] = word2idx[data['head_type'][edge]]
            node_type[data['tail'][edge]] = word2idx[data['tail_type'][edge]]

        node_type = torch.LongTensor(node_type)
        self.x = embedding(node_type)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, 
                                               self.num_nodes, 
                                               self.percent)

        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)
        self.datalist = ['data_{}_{}.pt'.format(i, self.split) for i in range(len(self.links))]

        self.A = ssp.csr_matrix(
        (data['relation'] + 1, (self.data['head'], self.data['tail'])), 
        shape=(self.num_nodes, self.num_nodes))

        super(Citation, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return self.datalist

    # def download(self):
    #     return

    def process(self):
        for idx in tqdm(range(len(self.links))):
            src, dst = self.links[idx]
            y = self.A[src, dst]

            if self.labels[idx]: status = "pos"
            else: status = "neg"

            tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, status, self.ratio_per_hop, 
                                 self.max_nodes_per_hop, node_features=self.x, directed=True)
            L_node_features, L_edges,  L_num_nodes, L_node_ids = construct_pyg_graph(*tmp, self.node_label, directed=True)
    
            data = citation_datasets(L_node_features, L_edges,  L_num_nodes, L_node_ids, y, self.alpha, self.adj_type)        
            data = data if self.pre_transform is None else self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, self.split)))
            #torch.save(self.collate([data]), self.processed_paths[0])

    def _process(self):
        makedirs(self.processed_dir)
        if self.split == 'train' and len(glob.glob(osp.join(self.processed_dir, '*train.pt'))) > 0:
            return
        if self.split == 'test' and len(glob.glob(osp.join(self.processed_dir, '*test.pt'))) > 0:
            return
        if self.split == 'valid' and len(glob.glob(osp.join(self.processed_dir, '*valid.pt'))) > 0:
            return

        self.process()

    def __repr__(self):
        return '{}()'.format(self.name)

def citation_datasets(features, edges,  num_nodes, node_ids, labels, alpha=0.1, adj_type=None):
    # path = os.path.join(save_path, dataset)
    indices = edges.t()

    """
    g = load_npz_dataset(node_features, edges,  num_nodes, node_ids)
    adj, features, labels = g['A'], g['X'], g['z']

    coo = adj.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    features = torch.from_numpy(features.todense()).float()
    labels = torch.from_numpy(labels).long()
    """

    if adj_type == 'un':
        print("Processing to undirected adj")
        indices = to_undirected(indices)
        edge_index, edge_weight = get_undirected_adj(indices, features.shape[0], features.dtype)
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor([labels]))
    elif adj_type == 'pr':
        print("Processing pagerank adj matrix")
        edge_index, edge_weight = get_pr_directed_adj(alpha, indices, features.shape[0],features.dtype)
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor([labels]))
    elif adj_type == 'appr':
        print("Processing approximate personalized pagerank adj matrix")
        edge_index, edge_weight = get_appr_directed_adj(alpha, indices, features.shape[0],features.dtype)
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor([labels]))
    elif adj_type == 'ib':
        print("Processing first and second-order adj matrix")
        edge_index, edge_weight = get_appr_directed_adj(alpha, indices, features.shape[0],features.dtype) 
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor([labels]))
        edge_index, edge_weight = get_second_directed_adj(indices, features.shape[0],features.dtype)
        data.edge_index2 = edge_index
        data.edge_weight2 = edge_weight
    elif adj_type == 'or':
        print("Processing to original directed adj")
        data = Data(x=features, edge_index=indices, edge_weight=None, y=labels)
    else:
        print("Unsupported adj type.")
        sys.exit()
    
    return data

def load_npz_dataset(node_features, edges,  num_nodes, node_ids):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        return graph
