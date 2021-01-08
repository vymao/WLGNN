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

import gc
from DataSet import *
from WLGNN import *

def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results

def train_model():
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader)
    for data in pbar:
        if not args.multi_gpu: data = data.to(device)
        optimizer.zero_grad()
        #print(data)
        #print(args)
        if args.use_orig_graph: 
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            out = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id).view(-1)
        else: out = model(data).view(-1)
        if args.multi_gpu: y = torch.cat([d.y.to(torch.float) for d in data]).to(out.device)
        else: y = data.y.to(torch.float)
         
        if args.neg_edge_percent != 100:
            y_neg = y[y == 0]
            out_neg = out[y == 0]
            y_pos = y[y != 0]
            out_pos = out[y != 0]
            
            num_neg = int(args.neg_edge_percent / 100 * len(out_neg))
            out_neg, y_neg  = out_neg[:num_neg], y_neg[:num_neg]
            
            y = torch.cat([y_pos, y_neg])
            out = torch.cat([out_pos, out_neg])
            

        loss = MSELoss()(out, y)
        loss = torch.sqrt(loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)

    return total_loss / len(train_dataset)

def normalized_RMSE(val_true, val_pred, norm_type):
    mse = mean_squared_error(val_true, val_pred, squared = False)
    if norm_type == "mean": return mse / val_pred.mean().item()
    elif norm_type == "std_dev": return mse / torch.std(val_pred).item()
    else: return mse

@torch.no_grad()
def test_model():
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(val_loader):
        if not args.multi_gpu: data = data.to(device)
        #x = data.x if args.use_feature else None
        #edge_weight = data.edge_weight if args.use_edge_weight else None
        #node_id = data.node_id if emb else None
        if args.use_orig_graph:
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            out = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        else: out = model(data)

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
    #pos_val_pred = val_pred[val_true==1]
    #neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    for data in tqdm(test_loader):
        if not args.multi_gpu: data = data.to(device)
        #x = data.x if args.use_feature else None
        #edge_weight = data.edge_weight if args.use_edge_weight else None
        #node_id = data.node_id if emb else None
        if args.use_orig_graph:
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            out = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        else: out = model(data)

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
    #pos_test_pred = test_pred[test_true==1]
    #neg_test_pred = test_pred[test_true==0]
    if 1 in torch.isnan(val_true): 
        print("Found in validation truth")
    if 1 in torch.isnan(val_pred):
        print("Found in validation prediction")
    if 1 in torch.isnan(test_true): 
        print("Found in test truth")
    if 1 in torch.isnan(test_pred): 
        print("Found in test prediction")

    results = {}
    #results['MSE'] = (mean_squared_error(val_true, val_pred, squared = False), mean_squared_error(test_true, test_pred, squared = False))
    results['MSE'] = (normalized_RMSE(val_true, val_pred, args.normalize), normalized_RMSE(test_true, test_pred, args.normalize))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Creation')

    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--dataset_file', type=str, default=None)
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
    parser.add_argument('--use_orig_graph', action='store_true',
                        help='whether to use the original subgraph instead of the line graph')
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
    parser.add_argument('--normalize', type=str, default="mean")
    return parser.parse_args()


###SCRIPT START######

args = parse_args()

if args.run_name: 
    wandb.run.name = args.run_name
    wandb.run.save()

wandb.config.update(args)

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

if "ogbl" in args.dataset:
    dataset = PygLinkPropPredDataset(name=args.dataset)
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)
        data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
        data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

else:
    if args.dataset_file is None: 
        print("Dataset file required.")
        sys.exit()

    s, d, w = [], [], []
    with open(args.dataset_file, 'r') as f: 
        for index, line in enumerate(f): 
            t1, t2, t3 = line.strip().split(" ")
            s.append(t1)
            d.append(t2)
            w.append(t3)
    
    s, d, w = list(map(int, s)), list(map(int, d)), list(map(int, w))
    s = torch.LongTensor(s) - 1
    d = torch.LongTensor(d) - 1
    w = torch.LongTensor(w)

    edges = torch.stack([s, d])
    split_edge = {}

    ### Setting training edges, validation edges, testing edges 
    np.random.seed(123)
    num_pos = edges.size(1)
    perm = np.random.permutation(num_pos)
    train = int(85/ 100 * num_pos)
    valid = int((num_pos - train) / 2)
    test = num_pos - valid - train

    train_edge = edges[:, perm[:train]]
    train_weight = w[perm[:train]]
    valid_edge = edges[:, perm[train:(train + valid)]]
    valid_weight = w[perm[train:(train + valid)]]
    test_edge = edges[:, perm[(train + valid):]]
    test_weight = w[perm[(train + valid):]]

    
    
    data_train_edge = torch.cat([train_edge, torch.stack([train_edge[1], train_edge[0]])], 1)
    data_train_weight = torch.cat([train_weight, train_weight])

    data = Data(edge_index = data_train_edge, edge_weight = data_train_weight, num_nodes = 1899)

    new_edge_index, _ = add_self_loops(edges)
    #print(valid)
    #print(test)
    neg_edge = negative_sampling(
        new_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=(valid + test))
    np.random.seed(123)
    num_neg = neg_edge.size(1)
    perm = np.random.permutation(num_neg)
    val_neg = neg_edge[:, perm[:valid]]
    test_neg = neg_edge[:, perm[valid:]]
    
    #print(len(test_edge[0]))
    #print(len(test_neg[0]))
    split_edge['train'] = {"edge": train_edge.t(), "weight": train_weight}
    split_edge['valid'] = {"edge": valid_edge.t(), "weight": valid_weight, "edge_neg": val_neg.t()}
    split_edge['test'] = {"edge": test_edge.t(), "weight": test_weight, "edge_neg": test_neg.t()}

if "ogbl" in args.dataset: 
    if args.dataset_dir is not None: 
        path = os.path.join(args.dataset_dir, dataset.root) + '_wl{}'.format(args.data_appendix)
    else: 
        path = dataset.root + '_wl{}'.format(args.data_appendix)
else:
    if args.dataset_dir is not None: 
        path = os.path.join("dataset", "social_network")
        path = os.path.join(args.dataset_dir, path) + '.{}'.format(args.data_appendix)
    else:  
        path = os.path.join("dataset", "social_network") + '.{}'.format(args.data_appendix)
        os.makedirs(path, exist_ok = True)


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
    use_orig_graph=args.use_orig_graph,
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
    use_orig_graph=args.use_orig_graph,
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
    use_orig_graph=args.use_orig_graph,
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

if "social" in args.dataset: dataset = None

print("Training...")
print(train_dataset)
print(train_dataset[:5])
for data in train_loader:
    for key, item in data:
        print(item.requires_grad)

for run in range(args.runs):
    emb = None
    if args.use_orig_graph: model = DGCNN(args, train_dataset, dataset, hidden_channels=args.hidden_channels, num_layers=args.num_layers, 
                      max_z=max_z, k=args.sortpool_k, use_feature=args.use_feature, 
                      node_embedding=emb)
    else: model = WLGNN_model(args, train_dataset, dataset, hidden_channels=args.hidden_channels, num_layers=args.num_layers, 
                      max_z=max_z, k=args.sortpool_k, use_feature=args.use_feature, dataset_name=args.dataset, 
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

    if args.continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'model_checkpoint{}.pth'.format(args.continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'optimizer_checkpoint{}.pth'.format(args.continue_from)))
        )
        start_epoch = args.continue_from + 1
        args.epochs -= args.continue_from

    #if args.multi_gpu: model = DataParallel(model)
    #model = model.to(device)
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
        loss = train_model()

        if epoch % args.eval_steps == 0:
            results = test_model()

            if epoch % args.log_steps == 0:
                model_name = os.path.join(
                    args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
                optimizer_name = os.path.join(
                    args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
                torch.save(model.state_dict(), model_name)
                torch.save(optimizer.state_dict(), optimizer_name)

                for key, result in results.items():
                    valid_res, test_res = result

                    wandb.log({"Run": run,"Epoch": epoch, "Epoch Normalized RMSE Train Loss": loss,
                        "Valid_set RMSE": valid_res, "Test_set RMSE": test_res})

                    print(key)
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Valid MSE: {valid_res:.2f}%, '
                          f'Test MSE: {test_res:.2f}%')
        gc.collect()
print(f'Results are saved in {args.res_dir}')



