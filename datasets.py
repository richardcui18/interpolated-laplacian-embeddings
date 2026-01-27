"""
Setup the require datasets and a getter
"""
from torch_geometric.datasets import FacebookPagePage, Planetoid, KarateClub, Planetoid, WebKB, PolBlogs, DeezerEurope
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
from torch_geometric.datasets import Twitch as PyGTwitch


def get_dataloaders(dataset):
    """
    Returns a standardized PyG Data object with train/test masks for each dataset.
    """
    if dataset == "random_affinity":
        total_nodes = 200
        num_nodes_per_community = int(total_nodes / 2)

        sizes = [num_nodes_per_community, num_nodes_per_community]
        probs = [[0.99, 0.3],
                [0.3, 0.99]]

        G = nx.stochastic_block_model(sizes, probs, seed = 1)

        labels = {}
        for i, block_size in enumerate(sizes):
            for node in range(sum(sizes[:i]), sum(sizes[:i+1])):
                labels[node] = i
        
        node_list = sorted(G.nodes())
        node_id_map = {node: i for i, node in enumerate(node_list)}
        mapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in G.edges()]
        edge_index = torch.tensor(mapped_edges, dtype=torch.long).T
        y = torch.tensor([labels[node] for node in node_list], dtype=torch.long)
     
        # Features: all ones vectors
        num_nodes = len(node_list)
        feature_dim = 1
        x = torch.ones((num_nodes, feature_dim), dtype=torch.float)

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
        total_nodes = 200
        num_cores = int(total_nodes * 0.5)
        num_periphery = total_nodes - num_cores

        sizes = [num_cores, num_periphery]
        probs = [[0.9, 0.5], 
                [0.5, 0.1]] 

        G = nx.stochastic_block_model(sizes, probs, seed = 1)

        labels = {}
        for i, block_size in enumerate(sizes):
            for node in range(sum(sizes[:i]), sum(sizes[:i+1])):
                labels[node] = i

        node_list = sorted(G.nodes())
        node_id_map = {node: i for i, node in enumerate(node_list)}
        mapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in G.edges()]
        edge_index = torch.tensor(mapped_edges, dtype=torch.long).T
        
        y = torch.tensor([labels[node] for node in node_list], dtype=torch.long)

        # Features: all ones vectors
        num_nodes = len(node_list)
        feature_dim = 1
        x = torch.ones((num_nodes, feature_dim), dtype=torch.float)

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

        # Features
        num_nodes = pyg_data.num_nodes
        feature_dim = 1
        x = torch.ones((num_nodes, feature_dim), dtype=torch.float)

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
    
    elif dataset == "facebook_ego":
        edges_df = pd.read_csv(
            "data/Facebook/raw/facebook_combined.txt", 
            sep=" ", header=None, names=["source", "target"]
        )
        edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

        num_nodes = edges_df.values.max() + 1

        # Features
        feature_dim = 1
        x = torch.ones((num_nodes, feature_dim), dtype=torch.float)

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

        # Features
        feature_dim = 1
        x = torch.ones((num_nodes, feature_dim), dtype=torch.float)

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

    elif dataset == "polblogs":
        dataset = PolBlogs(root="data/PolBlogs")
        pyg_data = dataset[0]

        edge_index = pyg_data.edge_index
        y = pyg_data.y
        num_nodes = pyg_data.num_nodes

        # Features
        feature_dim = 1
        x = torch.ones((num_nodes, feature_dim), dtype=torch.float)

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

    elif dataset == "deezer_europe":
        data_dir = "/home/users/zc186/interpolated-laplacian-embeddings/data/deezer_europe"

        # edges
        edges_df = pd.read_csv(f"{data_dir}/deezer_europe_edges.csv")
        edges_df = edges_df.astype({"node_1": int, "node_2": int})
        edge_index = torch.tensor(edges_df[["node_1", "node_2"]].values.T, dtype=torch.long).contiguous()

        # targets
        targets_df = pd.read_csv(f"{data_dir}/deezer_europe_target.csv")
        targets_df = targets_df.astype({"id": int, "target": int})
        max_id_targets = int(targets_df["id"].max())
        y_full = torch.zeros(max_id_targets + 1, dtype=torch.long)
        y_full[targets_df["id"].values] = torch.tensor(targets_df["target"].values, dtype=torch.long)

        # features
        # with open(f"{data_dir}/deezer_europe_features.json", "r") as fh:
        #     features_json = json.load(fh)

        # max_edge_node = int(max(edges_df["node_1"].max(), edges_df["node_2"].max()))
        # max_node_id = max(max_edge_node, int(y_full.size(0) - 1))
        # num_nodes = max_node_id + 1

        # features_list = []
        # for i in range(num_nodes):
        #     features_list.append(features_json.get(str(i), []))

        # mlb = MultiLabelBinarizer(sparse_output=False)
        # X = mlb.fit_transform(features_list)
        # x = torch.tensor(X, dtype=torch.float)

        # Dummy features
        max_edge_node = int(max(edges_df["node_1"].max(), edges_df["node_2"].max()))
        max_node_id = max(max_edge_node, int(y_full.size(0) - 1))
        num_nodes = max_node_id + 1
        feature_dim = 1
        x = torch.ones((num_nodes, feature_dim), dtype=torch.float)




        if y_full.size(0) < num_nodes:
            pad = torch.zeros(num_nodes - y_full.size(0), dtype=torch.long)
            y = torch.cat([y_full, pad], dim=0)
        else:
            # truncate or keep as-is
            y = y_full[:num_nodes]

        # train/test split
        num_train = int(0.7 * num_nodes)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        num_classes = int(torch.unique(y).numel())
        return data, train_mask, test_mask, num_classes

    elif dataset == "twitch_pt":
        sub_dataset = "PT"
        data_dir = f"/home/users/zc186/interpolated-laplacian-embeddings/data/twitch/{sub_dataset}"

        # edges
        edges_df = pd.read_csv(f"{data_dir}/musae_{sub_dataset}_edges.csv")
        edges_df = edges_df.astype({"from": int, "to": int})
        edge_index = torch.tensor(edges_df[["from", "to"]].values.T, dtype=torch.long).contiguous()

        # targets
        targets_df = pd.read_csv(f"{data_dir}/musae_{sub_dataset}_target.csv")
        targets_df["mature"] = targets_df["mature"].astype(bool).astype(int)
        max_id_targets = int(targets_df["new_id"].max())
        y_full = torch.zeros(max_id_targets + 1, dtype=torch.long)
        y_full[targets_df["new_id"].values] = torch.tensor(targets_df["mature"].values, dtype=torch.long)

        # features
        with open(f"{data_dir}/musae_{sub_dataset}_features.json", "r") as fh:
            features_json = json.load(fh)

        max_edge_node = int(edge_index.max().item())
        max_node_id = max(max_edge_node, int(y_full.size(0) - 1))
        num_nodes = max_node_id + 1

        features_list = []
        for i in range(num_nodes):
            features_list.append(features_json.get(str(i), []))

        mlb = MultiLabelBinarizer(sparse_output=False)
        X = mlb.fit_transform(features_list)
        x = torch.tensor(X, dtype=torch.float)

        # # Dummy features
        # max_edge_node = int(edge_index.max().item())
        # max_node_id = max(max_edge_node, int(y_full.size(0) - 1))
        # num_nodes = max_node_id + 1
        # feature_dim = 1
        # x = torch.ones((num_nodes, feature_dim), dtype=torch.float)



        if y_full.size(0) < num_nodes:
            pad = torch.zeros(num_nodes - y_full.size(0), dtype=torch.long)
            y = torch.cat([y_full, pad], dim=0)
        else:
            # truncate or keep as-is
            y = y_full[:num_nodes]

        # train/test split
        num_train = int(0.7 * num_nodes)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask

        num_classes = int(torch.unique(y).numel())
        return data, train_mask, test_mask, num_classes





