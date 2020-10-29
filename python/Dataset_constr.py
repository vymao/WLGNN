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

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)

import Dataset_constr, enclosing_subgraph


def main(): 
	args = parse_args()
	dataset = PygLinkPropPredDataset(name=args.dataset)

    split_edge = dataset.get_edge_split()

	if args.save_appendix == '':
	    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
	if args.data_appendix == '':
	    args.data_appendix = '_h{}_rph{}'.format(
	        args.num_hops, ''.join(str(args.ratio_per_hop).split('.')))
	    if args.max_nodes_per_hop is not None:
	        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)


	path = dataset.root + '_wl{}'.format(args.data_appendix)
	use_coalesce = False

	train_dataset = eval("WLDataset")(
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

	val_dataset = eval("WLDataset")(
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

	test_dataset = eval("WLDataset")(
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

def parse_args();
	parser = argparse.ArgumentParser(description='Dataset Creation')
	parser.add_argument('--dataset', type=str, default='ogbl-collab')
	parser.add_argument('--num_hops', type=int, default=1)
	parser.add_argument('--ratio_per_hop', type=float, default=1.0)
	parser.add_argument('--max_nodes_per_hop', type=int, default=None)
	parser.add_argument('--train_percent', type=float, default=100)
	parser.add_argument('--val_percent', type=float, default=100)
	parser.add_argument('--test_percent', type=float, default=100)
	parser.add_argument('--node_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
	parser.add_argument('--data_appendix', type=str, default='', 
	                    help="an appendix to the data directory")
	parser.add_argument('--save_appendix', type=str, default='', 
	                    help="an appendix to the save directory")

	return parser.parse_args()


if __name__ == "__main__":
    main()