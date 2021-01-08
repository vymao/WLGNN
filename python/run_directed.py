import wandb
wandb.init(project="cs-222")

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
from sklearn.metrics import mean_squared_error

import scipy.sparse as ssp
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Embedding
from torch.utils.data import DataLoader

from torch_sparse import coalesce
from torch_scatter import scatter_min
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, global_sort_pool, global_add_pool, DataParallel
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader, DataListLoader
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, to_networkx, 
                                   to_scipy_sparse_matrix, to_undirected)

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)

import gc
#from DataSet import *
#from WLGNN import *
from DIGCNConv_ib import *
from DirectedDataset import *

def train():
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader)
    for data in pbar:
        #if not args.multi_gpu: data = data.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        #print(data)
        #print(args)
        #out = model(data).view(-1)
        out = model(data).view(-1)
        if args.multi_gpu: y = torch.cat([d.y.to(torch.float) for d in data]).to(out.device)
        else: y = data.y.to(torch.float)

        loss = BCEWithLogitsLoss()(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)

    return total_loss / len(train_dataset)


@torch.no_grad()
def test():
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(val_loader):
        if not args.multi_gpu: data = data.to(device)
        #x = data.x if args.use_feature else None
        #edge_weight = data.edge_weight if args.use_edge_weight else None
        #node_id = data.node_id if emb else None
        out = model(data).view(-1)
        out = torch.round(out.view(-1)).cpu()
        
        if args.multi_gpu: y = torch.cat([d.y.view(-1).cpu().to(torch.float) for d in data])
        else: y = data.y.view(-1).cpu().to(torch.float)

        if args.pos_edge_test_only: 
            y_status = y != 0
            y = y[y_status]
            out = out[y_status]

        y_pred.append(out)
        y_true.append(y)

    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    for data in tqdm(test_loader):
        if not args.multi_gpu: data = data.to(device)
        #x = data.x if args.use_feature else None
        #edge_weight = data.edge_weight if args.use_edge_weight else None
        #node_id = data.node_id if emb else None
        out = model(data)
        out = torch.round(out).view(-1).cpu()

        if args.multi_gpu: y = torch.cat([d.y.view(-1).cpu().to(torch.float) for d in data])
        else: y = data.y.view(-1).cpu().to(torch.float)

        if args.pos_edge_test_only: 
            y_status = y != 0
            y = y[y_status]
            out = out[y_status]

        y_pred.append(out)
        y_true.append(y)
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true==1]
    neg_test_pred = test_pred[test_true==0]

    results = {}
    results['MRR'] = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    results['AUC'] = evaluate_auc(val_pred, val_true, test_pred, test_true)
    return results

def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    return (valid_auc, test_auc)

def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    return (valid_mrr, test_mrr)

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Creation')

    parser.add_argument('--dataset', type=str, default='ogbl-biokg')
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    # GNN settings
    parser.add_argument('--model', type=str, default='DGCNN')
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    # Subgraph extraction settings
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--node_label', type=str, default='drnl', 
                        help="which specific labeling trick to use")
    parser.add_argument('--use_feature', action='store_true', 
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_edge_weight', action='store_true', 
                        help="whether to consider edge weight in GNN")
    parser.add_argument('--skip_data_processing', action='store_true')
    parser.add_argument('--skip_graph_generation', action='store_true')
    # Training settings
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--train_percent', type=float, default=100)
    parser.add_argument('--val_percent', type=float, default=100)
    parser.add_argument('--test_percent', type=float, default=100)
    parser.add_argument('--dynamic_train', action='store_true', 
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--train_node_embedding', action='store_true', 
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--pretrained_node_embedding', type=str, default=None, 
                        help="load pretrained node embeddings as additional node features")
    parser.add_argument('--neg_edge_percent', type=float, default=100)
  
    # Testing settings
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--data_appendix', type=str, default='', 
                        help="an appendix to the data directory")
    parser.add_argument('--save_appendix', type=str, default='', 
                        help="an appendix to the save directory")
    parser.add_argument('--keep_old', action='store_true', 
                        help="do not overwrite old files in the save directory")
    parser.add_argument('--continue_from', type=int, default=None, 
                        help="from which epoch's checkpoint to continue training")
    parser.add_argument('--only_test', action='store_true', 
                        help="only test without training")
    parser.add_argument('--test_multiple_models', action='store_true', 
                        help="test multiple models together")
    parser.add_argument('--use_heuristic', type=str, default=None, 
                        help="test a link prediction heuristic (CN or AA)")
    parser.add_argument('--multi_gpu', action='store_true', 
                        help="whether to use multi-gpu parallelism")
    parser.add_argument('--pos_edge_test_only', action='store_true', 
                        help='whether to only test against positive edge weights or all edges')

    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)

    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--recache', action="store_true", help="clean up the old adj data", default=True)
    parser.add_argument('--normalize-features', action="store_true", default=True)
    parser.add_argument('--adj_type', type=str, default='ib')
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--node_embed_dim', type=int, default=16)


    return parser.parse_args()


###SCRIPT START######

args = parse_args()

if args.run_name: 
    wandb.run.name = args.run_name
    wandb.run.save()

wandb.config.update(args)

dataset = PygLinkPropPredDataset(name=args.dataset)

data = dataset[0]
split_edge = dataset.get_edge_split()
train_data = split_edge["train"]

if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_rph{}'.format(
        args.num_hops, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)

args.res_dir = os.path.join('results/{}.{}'.format(args.dataset, args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

if args.use_valedges_as_input:
    val_edge_index = split_edge['valid']['edge'].t()
    val_edge_index = to_undirected(val_edge_index)
    data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
    data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)


if args.dataset_dir is not None: 
    path = os.path.join("dataset", "ogbl-biokg")
    path = os.path.join(args.dataset_dir, path) + '.{}'.format(args.data_appendix)
else:  
    path = os.path.join("dataset", "ogbl-biokg") + '.{}'.format(args.data_appendix)
    os.makedirs(path, exist_ok = True)

path = dataset.root + '_wl.{}'.format(args.data_appendix)
use_coalesce = True
print("Converting node indices...")
num_nodes_dict = data['num_nodes_dict']
num_nodes = 0
for key in num_nodes_dict.keys(): 
    num_nodes += num_nodes_dict[key]


total = 0
skip = {}

#print()
if not args.skip_data_processing:
    for key in data['num_nodes_dict'].keys():
        #print(f'{key}: {total}')
        skip[key] = total
        total += data['num_nodes_dict'][key]

    word2idx = {
        "disease": 1,
        "function": 2,
        "drug": 3,
        "sideeffect": 4,
        "protein": 5
    }


    node_type= list(range(num_nodes))
    new_head, new_tail = [], []
    for edge in range(len(train_data['relation'])):
        node_type[train_data['head'][edge] + skip[train_data['head_type'][edge]]] = word2idx[train_data['head_type'][edge]]
        node_type[train_data['tail'][edge] + skip[train_data['tail_type'][edge]]] = word2idx[train_data['tail_type'][edge]]
        new_head.append(train_data['head'][edge] + skip[train_data['head_type'][edge]])
        new_tail.append(train_data['tail'][edge] + skip[train_data['tail_type'][edge]])
    node_type = torch.LongTensor(node_type)
    train_data['head'] = torch.LongTensor(new_head)
    train_data['tail'] = torch.LongTensor(new_tail)
    train_data['node_class'] = node_type

    train_data['relation'] += 1
    #print(len(train_data['head'][train_data['head'] == 30622]))
else: 
    train_data, split_edge = None, None

print("Finished")

if not args.skip_graph_generation: 
    G = nx.MultiDiGraph()
    edges = torch.stack([train_data['head'], train_data['tail']]).t()
    #labeled_attr = [{"class": i} for i in train_data['relation']]
    att = torch.reshape(train_data['relation'], (len(train_data['relation']), 1))
    edges = torch.hstack([edges, att])
    #labeled_edges = []
    #for i in range(len(edges)): 
    #    edge = edges[i]
    #    labeled_edges.append([edge[0], edge[1], labeled_attr[i]])
    G.add_edges_from(edges.tolist())
else: 
    print("Skipping graph generation")
    G = None

class_embedding = Embedding(10, args.node_embed_dim)
z_embedding = Embedding(1000, args.node_embed_dim)


dataset_class = 'Directed_Dataset' 
train_dataset = eval(dataset_class)(
    path, 
    train_data, 
    split_edge, 
    G, 
    num_nodes,
    data['num_nodes_dict'],
    alpha=args.alpha,
    num_hops=args.num_hops, 
    percent=args.train_percent, 
    input_dim=args.node_embed_dim,
    split='train', 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    adj_type = args.adj_type,
    skip_data_processing = args.skip_data_processing,
    class_embed = class_embedding,
    z_embed = z_embedding,
) 

dataset_class = 'Directed_Dataset' 
val_dataset = eval(dataset_class)(
    path, 
    train_data, 
    split_edge, 
    G,
    num_nodes,
    data['num_nodes_dict'],
    alpha=args.alpha,
    num_hops=args.num_hops, 
    percent=args.val_percent, 
    input_dim=args.node_embed_dim,
    split='valid', 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    adj_type = args.adj_type,
    skip_data_processing = args.skip_data_processing,
    class_embed = class_embedding,
    z_embed = z_embedding,
)

dataset_class = 'Directed_Dataset' 
test_dataset = eval(dataset_class)(
    path, 
    train_data, 
    split_edge, 
    G,
    num_nodes,
    data['num_nodes_dict'],
    alpha=args.alpha,
    num_hops=args.num_hops, 
    percent=args.test_percent,
    input_dim=args.node_embed_dim, 
    split='test', 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
    adj_type = args.adj_type,
    skip_data_processing = args.skip_data_processing,
    class_embed = class_embedding,
    z_embed = z_embedding,
)
print('Using', torch.cuda.device_count(), 'GPUs!')

if args.multi_gpu: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_z = 1000  # set a large max_z so that every z has embeddings to look up

if args.multi_gpu: train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, 
                      shuffle=True, num_workers=args.num_workers)
else: train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                      shuffle=True, num_workers=args.num_workers)

if args.multi_gpu: val_loader = DataListLoader(val_dataset, batch_size=args.batch_size, 
                    num_workers=args.num_workers)
else: val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                    num_workers=args.num_workers)
 
if args.multi_gpu: test_loader = DataListLoader(test_dataset, batch_size=args.batch_size, 
                     num_workers=args.num_workers)
else: test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                     num_workers=args.num_workers)
print("Training...")
evaluator = Evaluator(name=args.dataset)

for run in range(args.runs):
    emb = None
    #model = WLGNN_model(args, train_dataset, dataset, hidden_channels=args.hidden_channels, num_layers=args.num_layers, 
                  #max_z=max_z, k=args.sortpool_k, use_feature=args.use_feature, 
                  #node_embedding=emb)

    if args.aggregation == 'sum': model = Sparse_Three_Sum(train_dataset, args.node_embed_dim, args.hidden_channels, 1, args.dropout)
    else: model = Sparse_Three_Concat(train_dataset, args.node_embed_dim, args.hidden_channels, 1, args.dropout)

    wandb.watch(model)

    parameters = list(model.parameters())

    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    print(f'SortPooling k is set to {model.k}')
    log_file = os.path.join(args.res_dir, 'log.txt')
    with open(log_file, 'a') as f:
        print(f'Total number of parameters is {total_params}', file=f)
        if args.model == 'DGCNN':
            print(f'SortPooling k is set to {model.k}', file=f)

    start_epoch = 1
    if args.multi_gpu: model = DataParallel(model)
    model = model.to(device)
    # Training starts
    if args.only_test:
        results = test()
        for key, result in results.items():
            valid_res, test_res = result
            print(key)
            print(f'Run: {run + 1:02d}, '
                  f'Valid: {100 * valid_res:.2f}%, '
                  f'Test: {100 * test_res:.2f}%')
        exit()
    print("Beginning training...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss = train()

        if epoch % args.eval_steps == 0:
            results = test()

            if epoch % args.log_steps == 0:
                model_name = os.path.join(
                    args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
                optimizer_name = os.path.join(
                    args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
                torch.save(model.state_dict(), model_name)
                torch.save(optimizer.state_dict(), optimizer_name)

                valid_res_AUC, test_res_AUC = results['AUC']
                valid_res_MRR, test_res_MRR = results['MRR']

                wandb.log({"Run": run,"Epoch": epoch, "Epoch Normalized BCE Train Loss": loss,
                    "Valid_set MRR": valid_res_MRR, "Test_set MRR": test_res_MRR, "Valid_set AUC": valid_res_AUC, 
                    "Test_set AUC": valid_res_AUC})

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Valid MRR: {valid_res_MRR:.2f}, '
                      f'Test MRR: {test_res_MRR:.2f} '
                      f'Valid AUC: {valid_res_AUC:.2f}, '
                      f'Test AUC: {test_res_AUC:.2f} '
                      )
        gc.collect()
print(f'Results are saved in {args.res_dir}')



