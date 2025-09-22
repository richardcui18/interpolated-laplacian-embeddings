"""
Define the GNN models used and a getter
"""
import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import argparse
import torch.nn as nn
from basis_net_dependencies import IGN2to1
from torch_geometric.nn import GINConv, GATConv
from torch.nn import Sequential, BatchNorm1d, Dropout
from torch_geometric.nn import SAGEConv

def get_model(modelname, hyperparams, k, net_type):
    input_size = hyperparams["input_size"]
    hidden_size = hyperparams["hidden_size"]
    num_classes = hyperparams["num_classes"]
    if modelname == "gcn":
        return GCN(input_size, hidden_size, num_classes, k, net_type)
    elif modelname == "mlp":
        return MLP(input_size, hidden_size, num_classes, k, net_type)
    elif modelname == "gin":
        return GIN(input_size, hidden_size, num_classes, k, net_type)
    # elif modelname == "gat":
    #     heads = hyperparams.get("heads", 4)  # allow multi-head param
    #     return GAT(input_size, hidden_size, num_classes, k, net_type, heads=heads)
    elif modelname == "sage":
        return GraphSAGE(input_size, hidden_size, num_classes, k, net_type)
    else:
        raise Exception("model {} not supported".format(modelname))


class SignNet(nn.Module):
    """
    Tensor-only SignNet (simplified)

    Input:  n x k   (spectral embeddings, e.g. Laplacian eigenvectors)
    Output: n x n_hid   (sign-invariant embedding)

    Sign-invariance: f(x) + f(-x)
    """
    def __init__(self, in_dim: int, n_hid: int, nlayer: int = 2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, n_hid))
        layers.append(nn.ReLU())
        for _ in range(nlayer - 1):
            layers.append(nn.Linear(n_hid, n_hid))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def reset_parameters(self):
        for m in self.mlp:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x_spectral: torch.Tensor) -> torch.Tensor:
        # sign-invariance: apply MLP to x and -x, sum results
        return self.mlp(x_spectral) + self.mlp(-x_spectral)

class IGNBasisInv(nn.Module):
    """ IGN based basis invariant neural network
    """
    def __init__(self, mult_lst, in_channels, hidden_channels=16, num_layers=2):
        super(IGNBasisInv, self).__init__()
        self.encs = nn.ModuleList()
        self.mult_to_idx = {}
        curr_idx = 0
        for mult in mult_lst:
            # get a phi for each choice of multiplicity
            self.encs.append(IGN2to1(1, hidden_channels, mult, num_layers=num_layers))
            self.mult_to_idx[mult] = curr_idx
            curr_idx += 1


    def forward(self, proj, mult):
        enc_idx = self.mult_to_idx[mult]
        x = self.encs[enc_idx](proj)
        return x

# Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k, net_type):
        super().__init__()
        self.k = k  # number of spectral components

        if net_type == "sign_net":
            self.net = SignNet(k, hidden_channels, nlayer=2)
            out_dim_spectral = hidden_channels
        elif net_type == "basis_net":
            mult_lst = [k]
            self.net = IGNBasisInv(mult_lst, in_channels=k, hidden_channels=hidden_channels, num_layers=2)
            out_dim_spectral = k
        else:
            raise ValueError("net_type must be 'sign_net' or 'basis_net'")

        if self.k > 0:
            in_channels_new = (in_channels - k) + out_dim_spectral
        else:
            in_channels_new = in_channels

        self.conv1 = GCNConv(in_channels_new, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, out_channels)


    def forward(self, x, edge_index):
        if self.k > 0:
            # Split into spectral part and raw node features
            x_spectral = x[:, -self.k:]       # last k columns = Laplacian eigenvectors
            x_node = x[:, :-self.k]           # everything else = node features

            # Apply the right network
            if isinstance(self.net, SignNet):
                # tensor-only SignNet
                x_spectral = self.net(x_spectral)   # [N, hidden_channels]

            elif isinstance(self.net, IGNBasisInv):
                # reshape to [batch=1, D=1, n=N, m=k]
                proj = x_spectral.unsqueeze(0).unsqueeze(0)  # [1,1,N,k]
                proj = proj.to(next(self.net.parameters()).device)  # align device
                x_spectral = self.net(proj, self.k)          # BasisNet forward
                # now squeeze back to [N, hidden]
                x_spectral = x_spectral.squeeze(0).transpose(0,1)

            else:
                raise RuntimeError("Unknown spectral net type")

            # Concatenate back
            x = torch.cat([x_node, x_spectral], dim=-1)
        
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()
        h = self.conv4(h, edge_index).relu()
        z = self.out(h)
        return h, z

# Define MLP Model
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k, net_type):
        super().__init__()
        self.k = k  # number of spectral components
        if net_type == "sign_net":
            self.net = SignNet(k, hidden_channels, nlayer=2)
            out_dim_spectral = hidden_channels
        elif net_type == "basis_net":
            mult_lst = [k]
            self.net = IGNBasisInv(mult_lst, in_channels=k, hidden_channels=hidden_channels, num_layers=2)
            out_dim_spectral = k
        else:
            raise ValueError("net_type must be 'sign_net' or 'basis_net'")

        if self.k > 0:
            in_channels_new = (in_channels - k) + out_dim_spectral
        else:
            in_channels_new = in_channels

        self.lin1 = Linear(in_channels_new, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, hidden_channels)
        self.lin4 = Linear(hidden_channels, hidden_channels)
        self.lin5 = Linear(hidden_channels, out_channels)
        self.relu = ReLU()

    def forward(self, x, edge_index=None):  # edge_index ignored for MLP
        if self.k > 0:
            # Split into spectral part and raw node features
            x_spectral = x[:, -self.k:]       # last k columns = Laplacian eigenvectors
            x_node = x[:, :-self.k]           # everything else = node features

            # Apply the right network
            if isinstance(self.net, SignNet):
                # tensor-only SignNet
                x_spectral = self.net(x_spectral)   # [N, hidden_channels]

            elif isinstance(self.net, IGNBasisInv):
                # reshape to [batch=1, D=1, n=N, m=k]
                proj = x_spectral.unsqueeze(0).unsqueeze(0)  # [1,1,N,k]
                proj = proj.to(next(self.net.parameters()).device)  # align device
                x_spectral = self.net(proj, self.k)          # BasisNet forward
                # now squeeze back to [N, hidden]
                x_spectral = x_spectral.squeeze(0).transpose(0,1)

            else:
                raise RuntimeError("Unknown spectral net type")

            # Concatenate back
            x = torch.cat([x_node, x_spectral], dim=-1)

        h = self.relu(self.lin1(x))
        h = self.relu(self.lin2(h))
        h = self.relu(self.lin3(h))
        h = self.relu(self.lin4(h))
        z = self.lin5(h)
        return h, z


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k, net_type):
        super().__init__()
        self.k = k

        # spectral net selection
        if net_type == "sign_net":
            self.net = SignNet(k, hidden_channels, nlayer=2)
            out_dim_spectral = hidden_channels
        elif net_type == "basis_net":
            mult_lst = [k]
            self.net = IGNBasisInv(mult_lst, in_channels=k,
                                   hidden_channels=hidden_channels,
                                   num_layers=2)
            out_dim_spectral = k
        else:
            raise ValueError("net_type must be 'sign_net' or 'basis_net'")

        if self.k > 0:
            in_channels_new = (in_channels - k) + out_dim_spectral
        else:
            in_channels_new = in_channels

        # Build 5-layer GIN with small MLPs inside
        mlp1 = Sequential(
            Linear(in_channels_new, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(mlp1)

        mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(), 
            Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(mlp2)

        mlp3 = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.conv3 = GINConv(mlp3)

        mlp4 = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.conv4 = GINConv(mlp4)

        self.out = Linear(hidden_channels, out_channels)
        self.scaling_factor = 10.0

        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels) 
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        
        self.dropout = Dropout(0.5)
    def forward(self, x, edge_index):
        if self.k > 0:
            x_spectral = x[:, -self.k:]
            x_node = x[:, :-self.k]

            if isinstance(self.net, SignNet):
                x_spectral = self.net(x_spectral)
            elif isinstance(self.net, IGNBasisInv):
                proj = x_spectral.unsqueeze(0).unsqueeze(0)
                proj = proj.to(next(self.net.parameters()).device)
                x_spectral = self.net(proj, self.k)
                x_spectral = x_spectral.squeeze(0).transpose(0, 1)

            x = torch.cat([x_node, x_spectral*self.scaling_factor], dim=-1)

        h = x
        h = self.dropout(self.bn1(self.conv1(h, edge_index).relu()))
        h = self.dropout(self.bn2(self.conv2(h, edge_index).relu()))
        h = self.dropout(self.bn3(self.conv3(h, edge_index).relu()))
        h = self.dropout(self.bn4(self.conv4(h, edge_index).relu()))
        z = self.out(h)
        return h, z


# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, k, net_type, heads=4):
#         super().__init__()
#         self.k = k

#         # spectral net selection
#         if net_type == "sign_net":
#             self.net = SignNet(k, hidden_channels, nlayer=2)
#             out_dim_spectral = hidden_channels
#         elif net_type == "basis_net":
#             mult_lst = [k]
#             self.net = IGNBasisInv(mult_lst, in_channels=k,
#                                    hidden_channels=hidden_channels,
#                                    num_layers=2)
#             out_dim_spectral = k
#         else:
#             raise ValueError("net_type must be 'sign_net' or 'basis_net'")

#         if self.k > 0:
#             in_channels_new = (in_channels - k) + out_dim_spectral
#         else:
#             in_channels_new = in_channels

#         # First 4 layers: multi-head attention
#         self.conv1 = GATConv(in_channels_new, hidden_channels, heads=heads, concat=True)
#         self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
#         self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
#         self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)

 
#         self.scaling_factor = 5
#         self.out = Linear(hidden_channels, out_channels)
        
#         self.bn1 = BatchNorm1d(hidden_channels * heads)
#         self.bn2 = BatchNorm1d(hidden_channels * heads) 
#         self.bn3 = BatchNorm1d(hidden_channels * heads)
#         self.bn4 = BatchNorm1d(hidden_channels)
        
#         self.dropout = Dropout(0.5)
#     def forward(self, x, edge_index):
#         if self.k > 0:
#             x_spectral = x[:, -self.k:]
#             x_node = x[:, :-self.k]

#             if isinstance(self.net, SignNet):
#                 x_spectral = self.net(x_spectral)
#             elif isinstance(self.net, IGNBasisInv):
#                 proj = x_spectral.unsqueeze(0).unsqueeze(0)
#                 proj = proj.to(next(self.net.parameters()).device)
#                 x_spectral = self.net(proj, self.k)
#                 x_spectral = x_spectral.squeeze(0).transpose(0, 1)

#             x = torch.cat([x_node, x_spectral*self.scaling_factor], dim=-1)

#         h = x
   
#         # h = self.dropout(self.bn1(self.conv1(h, edge_index).relu()))
#         # h = self.dropout(self.bn2(self.conv2(h, edge_index).relu()))
#         # h = self.dropout(self.bn3(self.conv3(h, edge_index).relu()))
#         # h = self.dropout(self.bn4(self.conv4(h, edge_index).relu()))
   
#         h = self.dropout(self.conv1(h, edge_index).relu())
#         h = self.dropout(self.conv2(h, edge_index).relu())
#         h = self.dropout(self.conv3(h, edge_index).relu())
#         h = self.dropout(self.conv4(h, edge_index).relu())
#         z = self.out(h)
#         return h, z


# Define SAGE Model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k, net_type):
        super().__init__()
        self.k = k  # number of spectral components

        if net_type == "sign_net":
            self.net = SignNet(k, hidden_channels, nlayer=2)
            out_dim_spectral = hidden_channels
        elif net_type == "basis_net":
            mult_lst = [k]
            self.net = IGNBasisInv(mult_lst, in_channels=k, hidden_channels=hidden_channels, num_layers=2)
            out_dim_spectral = k
        else:
            raise ValueError("net_type must be 'sign_net' or 'basis_net'")

        if self.k > 0:
            in_channels_new = (in_channels - k) + out_dim_spectral
        else:
            in_channels_new = in_channels

        self.conv1 = SAGEConv(in_channels_new, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv4 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.out = Linear(hidden_channels, out_channels)
        
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels) 
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        
        self.dropout = Dropout(0.5)
        self.scaling_factor = 10.0

    def forward(self, x, edge_index):
        if self.k > 0:
            # Split into spectral part and raw node features
            x_spectral = x[:, -self.k:]       # last k columns = Laplacian eigenvectors
            x_node = x[:, :-self.k]           # everything else = node features

            # Apply the right network
            if isinstance(self.net, SignNet):
                # tensor-only SignNet
                x_spectral = self.net(x_spectral)   # [N, hidden_channels]

            elif isinstance(self.net, IGNBasisInv):
                # reshape to [batch=1, D=1, n=N, m=k]
                proj = x_spectral.unsqueeze(0).unsqueeze(0)  # [1,1,N,k]
                proj = proj.to(next(self.net.parameters()).device)  # align device
                x_spectral = self.net(proj, self.k)          # BasisNet forward
                # now squeeze back to [N, hidden]
                x_spectral = x_spectral.squeeze(0).transpose(0,1)

            else:
                raise RuntimeError("Unknown spectral net type")

            # Concatenate back
            x = torch.cat([x_node, x_spectral*self.scaling_factor], dim=-1)
        
        h = self.dropout(self.bn1(self.conv1(x, edge_index).relu()))
        h = self.dropout(self.bn2(self.conv2(h, edge_index).relu()))
        h = self.dropout(self.bn3(self.conv3(h, edge_index).relu()))
        h = self.dropout(self.bn4(self.conv4(h, edge_index).relu()))
        z = self.out(h)
        return h, z