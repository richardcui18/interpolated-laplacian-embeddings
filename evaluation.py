import torch
import numpy as np
from scipy.linalg import eigh
from scipy.stats import rankdata
from torch_geometric.utils import to_scipy_sparse_matrix
    
## Here we evaluate the model

@torch.no_grad()
def evaluate_model(trained_model, features, data, test_mask):
    trained_model.eval()
    edge_input = data.edge_index
    _, z = trained_model(features, edge_input)
    pred = z.argmax(dim=1)
    acc = accuracy(pred[test_mask], data.y[test_mask])
    return acc.item()

# Accuracy Function
def accuracy(pred_y, y):
    return (pred_y == y).sum().float() / len(y)


@torch.no_grad()
def spectral_alignment_scores(data, t, k=None, observed_mask=None):
    """
    Computes:
      - correlation_score: weighted sum (lambda_i * rho_i) / trace
      - spearman_score: same but using ranks (Spearman-style)

    Args:
      data: PyG data object (must include edge_index and y).
      t: single parameter (float in [0,1]) forming M = t D - (1-t) A.
      k: number of eigenpairs to use; if None, use full spectrum.
      observed_mask: boolean 1D array (len = n) selecting the nodes s in S (observed labels).
                     If None, use data.train_mask if present; otherwise use all nodes where y != -1.
    Returns:
      dict with keys: 'correlation_score', 'spearman_score', 'eigvals' (np.array), 'eigvecs' (np.array)
    """
    # Build A
    A = to_scipy_sparse_matrix(data.edge_index).tocsc()
    dense_A = A.toarray()
    degrees = dense_A.sum(axis=1)
    D = np.diag(degrees)

    M = t * D - (1.0 - t) * dense_A

    eigvals, eigvecs = eigh(M)

    if k is not None:
        idx = np.argsort(np.abs(eigvals))[:k]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

    # Determine observed set S
    n = dense_A.shape[0]
    if observed_mask is None:
        if hasattr(data, "train_mask"):
            observed_mask = data.train_mask.numpy().astype(bool)
        else:
            raise Exception("Observed mask not available.")


    S_idx = np.where(observed_mask)[0]
    if len(S_idx) == 0:
        raise ValueError("Observed set S is empty; cannot compute spectral alignment.")

    f_vals = data.y.numpy().astype(float)
    f_s = f_vals[S_idx]

    # Per-eigenvector rho_i = (1/|S|) * sum_{s in S} f_s * v_s^i
    # Also compute Spearman-style rho_i using ranks
    rho_list = []
    rho_spearman_list = []
    for i in range(eigvecs.shape[1]):
        v_i = eigvecs[:, i]
        v_s = v_i[S_idx]

        rho_i = (f_s * v_s).sum() / float(len(S_idx))
        rho_list.append(rho_i)

        # ranks for Spearman-style
        r_f = rankdata(f_s)        # ranks 1..|S|
        r_v = rankdata(v_s)
        rho_sp_i = (r_f * r_v).sum() / float(len(S_idx))
        # center/normalize ranks so rho is comparable: subtract mean
        rho_sp_i = rho_sp_i - ( (len(S_idx)+1) / 2.0 )**2 / len(S_idx)
        rho_spearman_list.append(rho_sp_i)

    rho_arr = np.array(rho_list)
    rho_sp_arr = np.array(rho_spearman_list)

    # # Weighted sum by eigenvalues; normalize by trace (=sum eigenvalues)
    # trace = eigvals.sum() if eigvals.sum() != 0 else 1.0
    # corr_score = (eigvals * rho_arr).sum() / trace
    # spearman_score = (eigvals * rho_sp_arr).sum() / trace

    # Weighted sum by absolute eigenvalues; normalize by sum of abs eigenvalues
    abs_eigvals = np.abs(eigvals)
    trace = abs_eigvals.sum() if abs_eigvals.sum() != 0 else 1.0

    corr_score = (abs_eigvals * rho_arr).sum() / trace
    spearman_score = (abs_eigvals * rho_sp_arr).sum() / trace


    return {
        "correlation_score": float(corr_score),
        "spearman_score": float(spearman_score),
        "eigvals": eigvals,
        "eigvecs": eigvecs
    }
