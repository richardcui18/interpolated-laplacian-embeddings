"""
Setup the require datasets and a getter
"""
from torch_geometric.datasets import FacebookPagePage, Planetoid
from sklearn import model_selection
import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer
import json
from torch_sparse import SparseTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import KarateClub, Planetoid, WebKB, AttributedGraphDataset, CitationFull, PolBlogs


def get_dataloaders(dataset):
    """
    Returns a standardized PyG Data object with train/test masks for each dataset.
    """
    if dataset == "random_affinity":
        total_nodes = 1000
        num_nodes_per_community = int(total_nodes / 2)

        sizes = [num_nodes_per_community, num_nodes_per_community]
        probs = [[0.99, 0.3],
                [0.3, 0.99]]

        G = nx.stochastic_block_model(sizes, probs)

        labels = {}
        for i, block_size in enumerate(sizes):
            for node in range(sum(sizes[:i]), sum(sizes[:i+1])):
                labels[node] = i
        
        node_list = sorted(G.nodes())
        node_id_map = {node: i for i, node in enumerate(node_list)}
        mapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in G.edges()]
        edge_index = torch.tensor(mapped_edges, dtype=torch.long).T
        y = torch.tensor([labels[node] for node in node_list], dtype=torch.long)
     
        # Features: random vectors
        num_nodes = len(node_list)
        feature_dim = 1
        x = torch.rand((num_nodes, feature_dim), dtype=torch.float)

        # Train/test split
        num_train = int(num_nodes * 0.7)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        return data, train_mask, test_mask, len(torch.unique(y))

    elif dataset == "random_core":
        total_nodes = 1000
        num_cores = int(total_nodes * 0.5)
        num_periphery = total_nodes - num_cores

        sizes = [num_cores, num_periphery]
        probs = [[0.9, 0.5], 
                [0.5, 0.1]] 

        G = nx.stochastic_block_model(sizes, probs)

        labels = {}
        for i, block_size in enumerate(sizes):
            for node in range(sum(sizes[:i]), sum(sizes[:i+1])):
                labels[node] = i

        node_list = sorted(G.nodes())
        node_id_map = {node: i for i, node in enumerate(node_list)}
        mapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in G.edges()]
        edge_index = torch.tensor(mapped_edges, dtype=torch.long).T
        
        y = torch.tensor([labels[node] for node in node_list], dtype=torch.long)

        # Features: random vectors
        num_nodes = len(node_list)
        feature_dim = 1
        x = torch.rand((num_nodes, feature_dim), dtype=torch.float)

        # Train/test split
        num_train = int(num_nodes * 0.7)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        return data, train_mask, test_mask, len(torch.unique(y))
    

    elif dataset == "karate":
        pyg_data = KarateClub()[0]

        edge_index = pyg_data.edge_index
        y = pyg_data.y

        # Features: random
        num_nodes = pyg_data.num_nodes
        feature_dim = 1
        x = torch.rand((num_nodes, feature_dim), dtype=torch.float)

        # Train/test split
        num_train = int(num_nodes * 0.7)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        return data, train_mask, test_mask, len(torch.unique(y))
    
    elif dataset == "cora":
        dataset = Planetoid(root='data/Cora', name='Cora', split='random', num_val=0, num_test=2000)
        data = dataset[0]

        return data, data.train_mask, data.test_mask, dataset.num_classes
    
    elif dataset == 'pubmed':
        dataset = Planetoid(root='data/Planetoid', name='PubMed', split='random', num_val=0, num_test=2000)
        data = dataset[0]

        return data, data.train_mask, data.test_mask, dataset.num_classes
    
    elif dataset == 'citeseer':
        dataset = Planetoid(root='data/CiteSeer', name='CiteSeer', split='random', num_val=0, num_test=2000)
        data = dataset[0]

        return data, data.train_mask, data.test_mask, dataset.num_classes

    elif dataset == 'lastfm_asia':
        edges_df = pd.read_csv("data/LastFMAsia/raw/lasftm_asia/lastfm_asia_edges.csv")
        edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

        num_nodes = edges_df.values.max() + 1
        
        # Load node features
        with open("data/LastFMAsia/raw/lasftm_asia/lastfm_asia_features.json", 'r') as f:
            features_dict = json.load(f)

        node_ids = sorted(map(int, features_dict.keys()))
        features_list = [features_dict[str(node)] for node in node_ids]
        
        mlb = MultiLabelBinarizer()
        x = torch.tensor(mlb.fit_transform(features_list), dtype=torch.float)

        targets_df = pd.read_csv("data/LastFMAsia/raw/lasftm_asia/lastfm_asia_target.csv")
        y = torch.tensor(targets_df.sort_values('id')['target'].values, dtype=torch.long)
        
        # Train/test masks
        num_nodes = len(y)
        num_train = int(0.7 * num_nodes)

        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask
        
        return data, train_mask, test_mask, len(torch.unique(y))
    
    elif dataset == "facebook_ego":
        edges_df = pd.read_csv(
            "data/Facebook/raw/facebook_combined.txt", 
            sep=" ", header=None, names=["source", "target"]
        )
        edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

        num_nodes = edges_df.values.max() + 1

        # Features: random
        feature_dim = 1
        x = torch.rand((num_nodes, feature_dim), dtype=torch.float)

        degrees = torch.bincount(edge_index.view(-1), minlength=num_nodes)

        # Assign labels based on top 20%
        cutoff = int(0.2 * num_nodes)
        topk_indices = torch.topk(degrees, cutoff).indices
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[topk_indices] = 1

        # Train/test split
        num_train = int(0.7 * num_nodes)
        perm = torch.randperm(num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        return data, train_mask, test_mask, len(torch.unique(y))

    elif dataset == "twitter_congress":
        path = "data/congress_network/congress.edgelist"
        G = nx.read_edgelist(
            path,
            create_using=nx.DiGraph(),
            nodetype=int,
        )

        # Convert to undirected & drop weights
        G = G.to_undirected()
        edge_index = torch.tensor(list(G.edges)).t().contiguous()

        num_nodes = max(G.nodes) + 1

        # Random features
        feature_dim = 1
        x = torch.rand((num_nodes, feature_dim), dtype=torch.float)

        degrees = torch.tensor([deg for _, deg in G.degree(range(num_nodes))])

        # Assign labels based on top 20%
        cutoff = int(0.2 * num_nodes)
        topk_indices = torch.topk(degrees, cutoff).indices
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[topk_indices] = 1

        # Train/test split
        num_train = int(0.7 * num_nodes)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        return data, train_mask, test_mask, len(torch.unique(y))

    elif dataset == "arxiv_grqc":
        path = "data/ca-GrQc/ca-GrQc.txt"

        edges_df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            header=None,
            names=["source", "target"]
        )
        edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

        num_nodes = edges_df.values.max() + 1

        # Random features
        x = torch.rand((num_nodes, 1), dtype=torch.float)

        # Degree-based labels
        degrees = torch.bincount(edge_index.view(-1), minlength=num_nodes)
        cutoff = int(0.2 * num_nodes)
        topk_indices = torch.topk(degrees, cutoff).indices
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[topk_indices] = 1

        # Train/test split
        num_train = int(0.7 * num_nodes)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        return data, train_mask, test_mask, len(torch.unique(y))

    elif dataset == "cornell":
        dataset = WebKB(root='data/WebKB', name='Cornell')
        pyg_data = dataset[0]

        edge_index = pyg_data.edge_index
        y = pyg_data.y
        x = pyg_data.x

        num_nodes = pyg_data.num_nodes

        # Train/test split
        num_train = int(num_nodes * 0.7)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        return data, train_mask, test_mask, len(torch.unique(y))

    elif dataset == "texas":
        dataset = WebKB(root='data/WebKB', name='Texas')
        pyg_data = dataset[0]

        edge_index = pyg_data.edge_index
        y = pyg_data.y
        x = pyg_data.x

        num_nodes = pyg_data.num_nodes

        # Train/test split
        num_train = int(num_nodes * 0.7)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        return data, train_mask, test_mask, len(torch.unique(y))

    elif dataset == "wisconsin":
        dataset = WebKB(root='data/WebKB', name='Wisconsin')
        pyg_data = dataset[0]

        edge_index = pyg_data.edge_index
        y = pyg_data.y
        x = pyg_data.x

        num_nodes = pyg_data.num_nodes

        # Train/test split
        num_train = int(num_nodes * 0.7)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        return data, train_mask, test_mask, len(torch.unique(y))
    
    elif dataset == "wiki":
        dataset = AttributedGraphDataset(root="data/AttributedGraph", name="Wiki")
        pyg_data = dataset[0]
        edge_index, y, x = pyg_data.edge_index, pyg_data.y, pyg_data.x
        num_nodes = pyg_data.num_nodes

        # Train/test split (70/30 random)
        num_train = int(num_nodes * 0.7)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask, data.test_mask = train_mask, test_mask

        return data, train_mask, test_mask, len(torch.unique(y))

    elif dataset == "coraml":
        dataset = CitationFull(root="data/CitationFull", name="Cora_ML")
        pyg_data = dataset[0]
        edge_index, y, x = pyg_data.edge_index, pyg_data.y, pyg_data.x
        num_nodes = pyg_data.num_nodes

        # Train/test split (70/30 random)
        num_train = int(num_nodes * 0.7)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask, data.test_mask = train_mask, test_mask

        return data, train_mask, test_mask, len(torch.unique(y))

    elif dataset == "polblogs":
        dataset = PolBlogs(root="data/PolBlogs")
        pyg_data = dataset[0]

        edge_index = pyg_data.edge_index
        y = pyg_data.y
        num_nodes = pyg_data.num_nodes

        # Features: random (since none are provided)
        feature_dim = 1
        x = torch.rand((num_nodes, feature_dim), dtype=torch.float)

        # Train/test split (70/30 random)
        num_train = int(num_nodes * 0.7)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask, data.test_mask = train_mask, test_mask

        return data, train_mask, test_mask, len(torch.unique(y))


