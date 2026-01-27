from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from torch_geometric.utils import to_networkx
import torch
import numpy as np
import networkx as nx
import netlsd
from sklearn.preprocessing import StandardScaler
import scipy


def get_features(data, feature_name, k, p, t = None):
    
    x_raw = data.x.clone()
    x = torch.tensor(StandardScaler().fit_transform(x_raw), dtype=torch.float32)

    # Inject noise into p fraction of nodes
    if p > 0.0:
        num_nodes = x.size(0)
        num_noisy = int(p * num_nodes)
        noisy_indices = torch.randperm(num_nodes)[:num_noisy]
        noise = torch.randn_like(x[noisy_indices])  # standard Gaussian noise
        x[noisy_indices] = noise
    
    if feature_name == "none":
        return x
    
    elif feature_name == "laplacian":
        adj = to_scipy_sparse_matrix(data.edge_index).tocsc()
        laplacian = csgraph.laplacian(adj, normed=False)

        # Use dense matrix
        dense_laplacian = laplacian.toarray()
        eigval, eigvec = eigh(dense_laplacian)
        lap_features = torch.from_numpy(eigvec[:, 1:k+1]).float()

        lap_features = torch.tensor(StandardScaler().fit_transform(lap_features), dtype=torch.float32)
        return torch.cat([x, lap_features], dim=1)
    
    elif feature_name == "adjacency":
        A = to_scipy_sparse_matrix(data.edge_index).tocsc()
        # Use dense matrix
        dense_A = A.toarray()
        eigval, eigvec = eigh(dense_A)
        topk_indices = np.argsort(eigval)[-k:]
        eigvec_topk = eigvec[:, topk_indices]
        adj_features = torch.from_numpy(eigvec_topk).float()

        return torch.cat([x, adj_features], dim=1)
    
    elif feature_name == "general_family":
        if t is None:
            raise ValueError("Parameter 't' must be provided when using 'general_family' feature type.")

        A = to_scipy_sparse_matrix(data.edge_index).tocsc()
        dense_A = A.toarray()

        L_sparse = csgraph.laplacian(A, normed=False)
        dense_L = L_sparse.toarray()

        M = t * dense_L + (1.0 - t) * dense_A

        eigval, eigvec = eigh(M)

        topk_indices = np.argsort(eigval)[:k]
        eigvec_topk = eigvec[:, topk_indices]

        general_features = torch.from_numpy(eigvec_topk).float()

        general_features = torch.tensor(
            StandardScaler().fit_transform(general_features),
            dtype=torch.float32
        )

        return torch.cat([x, general_features], dim=1)

    
    else:
        raise ValueError(f"Feature type '{feature_name}' is not supported.")
