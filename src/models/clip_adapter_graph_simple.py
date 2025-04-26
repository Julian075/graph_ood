import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CLIPAdapterGraphSimple(nn.Module):
    def __init__(
        self,
        reduction_factor=32,
        device="cuda",
        gnn_hidden_dim=256,
        num_gnn_layers=2,
        seed=None
    ):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.device = device
        self.reduction_factor = reduction_factor
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_gnn_layers = num_gnn_layers
        
        # CLIP feature dimension (assuming standard CLIP model)
        self.clip_feature_dim = 768
        
        # Adapter layers
        self.down_proj = nn.Linear(self.clip_feature_dim, self.clip_feature_dim // self.reduction_factor)
        self.up_proj = nn.Linear(self.clip_feature_dim // self.reduction_factor, self.clip_feature_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # First GNN layer (input layer)
        self.gnn_layers.append(GCNConv(self.clip_feature_dim, self.gnn_hidden_dim))
        
        # Hidden GNN layers
        for _ in range(num_gnn_layers - 2):
            self.gnn_layers.append(GCNConv(self.gnn_hidden_dim, self.gnn_hidden_dim))
        
        # Last GNN layer (output layer)
        self.gnn_layers.append(GCNConv(self.gnn_hidden_dim, self.clip_feature_dim))
        
        self.to(device)

    def forward(self, x, edge_index=None, edge_weight=None):
        # Apply adapter
        identity = x
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        
        # Add residual connection
        x = x + identity
        
        # If edge_index is provided, apply GNN layers
        if edge_index is not None:
            # Apply GNN layers
            for i, gnn_layer in enumerate(self.gnn_layers):
                x = gnn_layer(x, edge_index, edge_weight)
                # Apply ReLU to all but the last layer
                if i < len(self.gnn_layers) - 1:
                    x = F.relu(x)
        
        return x 