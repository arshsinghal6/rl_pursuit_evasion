import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_QNetwork(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super(GCN_QNetwork, self).__init__()
        self.gcn1 = GCNConv(in_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.q_linear = nn.Linear(hidden_dim, 1)  # Output Q-value per node

    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_features]
        edge_index: [2, num_edges] (PyTorch Geometric format)
        returns: Q-values per node [num_nodes]
        """
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        q_vals = self.q_linear(x).squeeze(-1)  # [num_nodes]
        return q_vals
