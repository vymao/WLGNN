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
from torch.nn import BCEWithLogitsLoss, MSELoss
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


from DataSet import *
from WLGNN import *

def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results

def train():
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader)
    for data in pbar:
        if not args.multi_gpu: data = data.to(device)
        optimizer.zero_grad()
        #print(data)
        #print(args)
        out = model(data).view(-1)
        y = torch.cat([d.y.to(torch.float) for d in data]).to(out.device)
         
        if args.neg_edge_percent != 100:
            y_neg = y[y == 0]
            out_neg = out[y == 0]
            y_pos = y[y != 0]
            out_pos = out[y != 0]
            
            num_neg = int(args.neg_edge_percent / 100 * len(out_neg))
            out_neg, y_neg  = out_neg[:num_neg], y[:num_neg]
            
            y = torch.cat([y_pos, y_neg])
            out = torch.cat([out_pos, out_neg])
            
            


        loss = MSELoss()(out, y)
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
        out = model(data)
        out = torch.round(out).view(-1).cpu()
        y = torch.cat([d.y.view(-1).cpu().to(torch.float) for d in data])

        if args.pos_edge_test_only: 
            y_status = y != 0
            y = y[y_status]
            out = out[y_status]

        y_pred.append(out)
        y_true.append(y)

    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    #pos_val_pred = val_pred[val_true==1]
    #neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    for data in tqdm(test_loader):
        if not args.multi_gpu: data = data.to(device)
        #x = data.x if args.use_feature else None
        #edge_weight = data.edge_weight if args.use_edge_weight else None
        #node_id = data.node_id if emb else None
        out = model(data)
        out = torch.round(out).view(-1).cpu()
        y = torch.cat([d.y.view(-1).cpu().to(torch.float) for d in data])

        if args.pos_edge_test_only: 
            y_status = y != 0
            y = y[y_status]
            out = out[y_status]

        y_pred.append(out)
        y_true.append(y)
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    #pos_test_pred = test_pred[test_true==1]
    #neg_test_pred = test_pred[test_true==0]

    results = {}
    results['MSE'] = (mean_squared_error(val_true, val_pred), mean_squared_error(test_true, test_pred))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Creation')

    parser.add_argument('--dataset', type=str, default='ogbl-collab')
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
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
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
    return parser.parse_args()


###SCRIPT START######

args = parse_args()
wandb.config.update(args)

dataset = PygLinkPropPredDataset(name=args.dataset)
data = dataset[0]
split_edge = dataset.get_edge_split()

if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_rph{}'.format(
        args.num_hops, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)

args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

if args.use_valedges_as_input:
    val_edge_index = split_edge['valid']['edge'].t()
    val_edge_index = to_undirected(val_edge_index)
    data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
    data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

path = dataset.root + '_wl{}'.format(args.data_appendix)
use_coalesce = True

#datalist = ['data_{}_{}.pt'.format(i, "t") for i in range(]

dataset_class = 'WLDynamicDataset' if args.dynamic_train else 'WLDataset'
train_dataset = eval(dataset_class)(
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.train_percent, 
    split='train', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
) 

dataset_class = 'WLDynamicDataset' if args.dynamic_val else 'WLDataset'
val_dataset = eval(dataset_class)(
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.val_percent, 
    split='valid', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
)

dataset_class = 'WLDynamicDataset' if args.dynamic_test else 'WLDataset'
test_dataset = eval(dataset_class)(
    path, 
    data, 
    split_edge, 
    num_hops=args.num_hops, 
    percent=args.test_percent, 
    split='test', 
    use_coalesce=use_coalesce, 
    node_label=args.node_label, 
    ratio_per_hop=args.ratio_per_hop, 
    max_nodes_per_hop=args.max_nodes_per_hop, 
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

for run in range(args.runs):
    emb = None
    model = WLGNN_model(args, train_dataset, dataset, hidden_channels=args.hidden_channels, num_layers=args.num_layers, 
                  max_z=max_z, k=args.sortpool_k, use_feature=args.use_feature, 
                  node_embedding=emb)

    wandb.watch(model)

    parameters = list(model.parameters())

    optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
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

                for key, result in results.items():
                    valid_res, test_res = result

                    wandb.log({"Run": run,"Epoch": epoch, "Epoch Normalized MSE Train Loss": loss,
                        "Valid_set MSE": valid_res, "Test_set MSE": test_res})

                    print(key)
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Valid MSE: {valid_res:.2f}%, '
                          f'Test MSE: {test_res:.2f}%')

print(f'Results are saved in {args.res_dir}')



