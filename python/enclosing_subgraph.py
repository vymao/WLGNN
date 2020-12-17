import sys
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from torch_sparse import spspmm
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import pdb

import networkx as nx


def neighbors(fringe, A, row=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    res = set()
    for node in fringe:
        if row:
            _, nei, _ = ssp.find(A[node, :])
        else:
            nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)

    return res


def k_hop_subgraph(src, dst, num_hops, A, status, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features=None, directed=False, use_orig_graph=False):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    if directed or use_orig_graph: 
        subgraph[0, 1] = 0
        subgraph[1, 0] = 0        
    else: 
        subgraph[0, 1] = 1
        subgraph[1, 0] = 1

    if node_features is not None:
        node_features = node_features[nodes]

    if status == "pos": y = A[src, dst]
    else: y = 0
    
    #print(f'k-hop num_nodes: {len(nodes)}')

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double-radius node labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl', use_orig_A=False, directed=False, use_orig_graph=False):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    #u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    
    node_ids = torch.LongTensor(node_ids)
    #u, v = torch.LongTensor(u), torch.LongTensor(v)

    #r = torch.LongTensor(r)
    #edge_index = torch.stack([u, v], 0)
    #edge_weight = r.to(torch.float)

    y = torch.tensor([y])
    if use_orig_graph:
        u, v, r = ssp.find(adj)
        num_nodes = adj.shape[0]

        u, v = torch.LongTensor(u), torch.LongTensor(v)
        r = torch.LongTensor(r)
        edge_index = torch.stack([u, v], 0)
        edge_weight = r.to(torch.float)

        if node_label == 'drnl':
            z = drnl_node_labeling(adj, 0, 1)
        elif node_label == 'hop':
            z = torch.tensor(dists)
        data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                    node_id=node_ids, num_nodes=num_nodes)
        return data
    elif not directed: 
        if node_label == 'drnl':
            z = drnl_node_labeling(adj, 0, 1)
        elif node_label == 'hop':
            z = torch.tensor(dists)

        if use_orig_A: o_data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z, 
                node_id=node_ids, num_nodes=num_nodes)
        else: o_data = None

        L_node_features, L_edges,  L_num_nodes, w, z1, z2, L_node_ids = construct_line_graph_undirected(node_ids, adj, z, node_features)
        edge_weight = torch.ones(len(L_edges))
        #print(L_edges)
        data = Data(L_node_features, L_edges.t(), edge_weight=edge_weight, y=y, w=torch.LongTensor(w), z1=torch.LongTensor(z1), 
            z2=torch.LongTensor(z2), node_id=L_node_ids, num_nodes=len(L_node_ids), o_data=o_data)
        return data
    else: 
        L_node_features, L_edges,  L_num_nodes, L_node_ids, L_node_classes = construct_line_graph_directed(node_ids, adj, node_features)
        
        
        
        return L_node_features, L_edges,  L_num_nodes, L_node_ids, L_node_classes


def construct_line_graph_directed(node_ids, A, node_features):

    u, v, r = ssp.find(A)
    print(f'max_weight: {max(r)}')
    
    #print(f'num_edges_khop: {len(u)}')
    #print(f'num_nodes_khop: {node_ids.size()}')

    node_ids = node_ids.tolist()
    node_features = node_features.tolist()

    G = nx.DiGraph()
    #G.add_nodes_from(node_ids)
    rows, cols = A.nonzero()
    A_edges_forward = list(zip(u, v))  
    A_edges_reverse = list(zip(v, u))

    info = {}
    node_class = {}
    for edge in A_edges_forward: 
        src, end = edge[0], edge[1]
        weight = A[src,end]
        edge_label = [0] * 52 + [node_features[src] != node_features[end]]
        #print(weight)
        edge_label[weight] = 1

        f1, f2 = node_features[src], node_features[end]
        info[(src, end)] = edge_label
        node_class[(src, end)] = [f1, f2]

    for edge in A_edges_reverse: 
        src, end = edge[0], edge[1]
        weight = A[end,src]
        edge_label = [0] * 52 + [node_features[src] != node_features[end]]
        #print(weight)
        edge_label[weight] = 1

        f1, f2 = node_features[src], node_features[end]
        info[(src, end)] = edge_label
        node_class[(src, end)] = [f1, f2]

    G.add_edges_from(A_edges_forward)
    G.add_edges_from(A_edges_reverse)
    

    L = nx.line_graph(G)
    num_nodes = L.number_of_nodes()
       
    L_node_ids = list(L.nodes)
    L_edges = list(L.edges)

    L_node_features = []

    index = {}
    node_ids, f = [], []
    value = 0
    for node in L_node_ids:
        node_ids.append(value) 
        L_node_features.append(info[node])
        f.append(node_class[node])
        index[node] = value
        value += 1

    edge_list = []
    for edge in L_edges: 
        v1, v2 = edge[0], edge[1]
        n1, n2 = index[v1], index[v2]
        edge_list.append([n1, n2])

    return torch.LongTensor(L_node_features), torch.LongTensor(edge_list), num_nodes, torch.LongTensor(node_ids), torch.LongTensor(f)

def construct_line_graph_undirected(node_ids, A, z, node_features): 
    info = {}
    z = z.tolist()
    node_ids = node_ids.tolist()
    node_features = node_features.tolist()

    G = nx.Graph()
    #G.add_nodes_from(node_ids)
    rows, cols = A.nonzero()
    A_edges = list(zip(rows,cols))
    #print(A_edges)

    weights = {}
    for edge in A_edges: 
        src, end = edge[0], edge[1]
        src_z, end_z = z[src], z[end]
        weight = A[src,end]
        weights[(src, end)] = weight
        f1, f2 = node_features[src], node_features[end]
        info[(src, end)] = f1 + f2
        
    A[0, 1] = 0
    A[1, 0] = 0  
    rows, cols = A.nonzero()
    A_edges = list(zip(rows,cols))

    G.add_edges_from(A_edges)
    L = nx.line_graph(G)
    num_nodes = L.number_of_nodes()

    L_node_ids = list(L.nodes)
    L_edges = list(L.edges)
    #L_edges = list(map(list, L_edges))

    L_node_features = []
    w, z1, z2 = [], [], []

    index = {}
    node_ids = []
    value = 0
    for node in L_node_ids:
        node_ids.append(value) 
        L_node_features.append(info[node])
        w.append(weights[node])
        z1.append(z[node[0]])
        z2.append(z[node[1]])
        index[node] = value
        value += 1
    node_ids.append(value + 1)
    L_node_features.append(info[(0, 1)])
    w.append(0)
    z1.append(z[0])
    z2.append(z[1])

    edge_list = []
    for edge in L_edges: 
        v1, v2 = edge[0], edge[1]
        n1, n2 = index[v1], index[v2]
        edge_list.append([n1, n2])
        edge_list.append([n2, n1])

    return torch.LongTensor(L_node_features), torch.LongTensor(edge_list), num_nodes, w, z1, z2, torch.LongTensor(node_ids)

 
def extract_enclosing_subgraphs(link_index, A, x, status, num_hops, node_label='drnl', 
                                ratio_per_hop=1.0, max_nodes_per_hop=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        tmp = k_hop_subgraph(src, dst, num_hops, A, status, ratio_per_hop, 
                             max_nodes_per_hop, node_features=x)
        data = construct_pyg_graph(*tmp, node_label)
        data_list.append(data)

    return data_list
