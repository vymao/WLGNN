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

from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops, negative_sampling
from torch_scatter import scatter_add
import scipy
from torch_geometric.data import InMemoryDataset, Dataset
from get_adj import get_undirected_adj,get_pr_directed_adj,get_appr_directed_adj,get_second_directed_adj
from enclosing_subgraph_directed import *

from torch_sparse import coalesce
from torch_scatter import scatter_min

import glob
from torch_geometric.data.makedirs import makedirs

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

    def __init__(self, root, data, split_edge, A, num_nodes, node_dict, alpha, num_hops, percent, input_dim = 16, 
                    split='train', ratio_per_hop=1.0, max_nodes_per_hop=None, adj_type=None, transform=None, pre_transform=None, 
                    skip_data_processing=False, class_embed=None, z_embed=None, **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.alpha = alpha
        self.adj_type = adj_type
        self.num_nodes = num_nodes
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.A = A
        self.dim = input_dim
        self.class_embed = class_embed
        self.z_embed = z_embed
        processed_dir = osp.join(root, "processed")

        print(f"Processing {split} dataset...")
        if not skip_data_processing:
            if split != 'train': 
                current_data= split_edge[split]
                total = 0
                skip = {}
                for key in node_dict.keys(): 
                    skip[key] = total
                    total += node_dict[key]

                new_head, new_tail = [], []
                for edge in tqdm(range(len(current_data['relation']))): 
                    new_head.append(current_data['head'][edge] + skip[current_data['head_type'][edge]])
                    new_tail.append(current_data['tail'][edge] + skip[current_data['tail_type'][edge]])
                
                current_data['head'] = torch.LongTensor(new_head)
                current_data['tail'] = torch.LongTensor(new_tail)
                
                new_head, new_tail = [], [] 
                for edge in tqdm(range(len(current_data['relation']))):
                    h = current_data['head_neg'][edge] + skip[current_data['head_type'][edge]]
                    t = current_data['tail_neg'][edge] + skip[current_data['tail_type'][edge]]
                    new_head.append(h.tolist())
                    new_tail.append(t.tolist())
 
                current_data['head_neg'] = torch.LongTensor(new_head)
                current_data['tail_neg'] = torch.LongTensor(new_tail)

                current_data['relation'] += 1
            
            self.x = data['node_class']
                #self.data['relation'] = self.data['relation'] + 1

            pos_edge, pos_label, neg_edge, neg_label = get_pos_neg_edges(split, self.split_edge, 
                                                   num_nodes = self.num_nodes, 
                                                   percent = self.percent)

            self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
            self.relations = torch.cat([pos_label, neg_label]).tolist()
            self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)
            self.datalist = ['data_{}_{}.pt'.format(i, self.split) for i in range(len(self.links))]
        else: 
            self.datalist = [os.path.basename(file) for file in glob.glob(osp.join(processed_dir, 
                                                                              '*{}.pt'.format(self.split)))]
            #print()
            #print(max(data['relation'].tolist())).

            #self.A = ssp.csr_matrix(
            #(self.data['relation'].tolist(), (self.data['head'], self.data['tail'])), 
            #shape=(self.num_nodes, self.num_nodes))


        print("Done")
        super(Directed_Dataset, self).__init__(root)

    @property
    def processed_file_names(self):
        return self.datalist

    # def download(self):
    #     return
    def __len__(self):
        return len(self.datalist)

    def process(self):
        for idx in tqdm(range(len(self.links))):
            if os.path.exists(osp.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, self.split))): continue 
            src, dst = self.links[idx]
            #print()
            #print((src, dst))
            #print(f'Num_src_out_edge: {self.split_edge[self.split]["head"][self.split_edge[self.split]["head"] == src].size()}')
            #print(f'Num_dst_out_edge: {self.split_edge[self.split]["head"][self.split_edge[self.split]["head"] == dst].size()}')
            label = self.relations[idx]
            y = self.labels[idx]

            if self.labels[idx]: status = "pos"
            else: status = "neg"

            tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, status, self.ratio_per_hop, 
                                 self.max_nodes_per_hop, node_features=self.x, directed=True)
            L_node_features, L_node_weights, L_edges,  L_num_nodes, L_node_ids = construct_pyg_graph(*tmp, src, dst, label)
            
            src_class = self.class_embed(L_node_weights[:,0])
            dst_class = self.class_embed(L_node_weights[:, 1])
            src_z = self.z_embed(L_node_weights[:, 2])
            dst_z = self.z_embed(L_node_weights[:, 3])

            L_node_features = torch.cat([L_node_features, src_class, dst_class, src_z, dst_z], 1)
    
            data = citation_datasets(L_node_features, L_node_weights, L_edges,  L_num_nodes, L_node_ids,  
                y, self.dim, self.alpha, self.adj_type)        
            data = data if self.pre_transform is None else self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, self.split)))
            #torch.save(self.collate([data]), self.processed_paths[0])

    def _process(self):
        makedirs(self.processed_dir)
        if self.split == 'train' and len(glob.glob(osp.join(self.processed_dir, '*train.pt'))) > 0:
            return
        if self.split == 'test' and len(glob.glob(osp.join(self.processed_dir, '*test.pt'))) > 0:
            return
        #if self.split == 'valid' and len(glob.glob(osp.join(self.processed_dir, '*valid.pt'))) > 0:
        #    return

        self.process()

    #def __repr__(self):
    #    return '{}()'.format(self.name)
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}_{}.pt'.format(idx, self.split)))
        return data  
def citation_datasets(features, node_weights, edges,  num_nodes, node_ids, labels, dim, alpha=0.1, adj_type=None):
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
        #print("Processing first and second-order adj matrix")
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

def get_pos_neg_edges(split, split_edge, edge_index=None, num_nodes=None, percent=100):
    source = split_edge[split]['head']
    target = split_edge[split]['tail']
    relation = split_edge[split]['relation']
    pos_edges = torch.stack([source, target])

    np.random.seed(123)
    num_source = source.size(0)
    perm = np.random.permutation(num_source)
    perm = perm[:int(percent / 100 * num_source)]
    pos_edge = pos_edges[:, perm]
    pos_rel = relation[perm]

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
        neg_relation = [0] * neg_edge.size(1)
    else:
        target_neg_head = split_edge[split]['head_neg']
        target_neg_tail = split_edge[split]['tail_neg']

        source, target, target_neg_head, target_neg_tail = source[perm], target[perm], target_neg_head[perm, :], target_neg_tail[perm, :]
        neg_edge = torch.stack([target_neg_head.view(-1), target_neg_tail.view(-1)])
    #neg_per_target = target_neg.size(1)
    #neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                            #target_neg.view(-1)])

    neg_rel= torch.LongTensor([0] * neg_edge.size(1))                        
    return pos_edge, pos_rel, neg_edge, neg_rel
