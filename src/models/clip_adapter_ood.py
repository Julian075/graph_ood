import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.nn as gnn
from torch_geometric.nn import knn_graph
from typing import Tuple, Optional, Dict

class CLIPAdapterOOD(nn.Module):
     def __init__(self, reduction_factor=8, seed=42, device="cuda", gnn_hidden_dim=256, num_gnn_layers=2, num_classes=46, train=True, k_neighbors=10):
        super().__init__()
        self.device = device
        self.training = train
        self.reduction_factor = reduction_factor
        self.k_neighbors = k_neighbors
        # Get CLIP embedding dimension
        self.clip_dim = 512
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_classes = num_classes
        # Calculate bottleneck dimension
        self.bottleneck_dim = self.clip_dim // reduction_factor
        
        # Adapter layers (trainable)
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        self.adapter = nn.Sequential(
            nn.Linear(self.clip_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, self.clip_dim)
        )
        
        # GNN layers
        if self.training:
            # Simple 2-layer GCN
            self.gnn_layers = nn.ModuleList()
            for _ in range(num_gnn_layers-1):
                self.gnn_layers.append(gnn.GCNConv(self.clip_dim, self.gnn_hidden_dim))
            self.gnn_layers.append(gnn.GCNConv(self.gnn_hidden_dim, self.num_classes))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
 
        # Move model to device
        self.to(device)
        self.seed = seed
        self.set_seed()

     def set_seed(self):
        """Set seed for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

     def create_knn_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create KNN graph from features."""
        edge_index = knn_graph(x, k=self.k_neighbors, batch=None)
        row, col = edge_index
        edge_weight = F.cosine_similarity(x[row], x[col], dim=1)
        return edge_index, edge_weight

     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            x: Input features tensor of shape [batch_size, clip_dim]
            
        Returns:
            Tuple containing:
            - adapter_embeddings: Adapted features [batch_size, clip_dim]
            - x_graph: GNN output [batch_size, num_classes] (None during inference)
        """
        # Ensure input is float32
        x = x.float()
        original_features = x

        # Adapter forward pass
        adapted = self.adapter(x)
        adapted = adapted + original_features  # Skip connection
        adapted = self.alpha * adapted + (1 - self.alpha) * original_features
        adapter_embeddings = F.normalize(adapted, dim=-1)
        
        if self.training:
            # Training mode: process through GNN
            edge_index, edge_weight = self.create_knn_graph(adapter_embeddings)
            
            # GNN forward pass
            x_graph = adapter_embeddings
            for gnn_layer in self.gnn_layers[:-1]:
                x_graph = gnn_layer(x_graph, edge_index, edge_weight)
                x_graph = F.relu(x_graph)
                x_graph = self.dropout(x_graph)
            x_graph = self.gnn_layers[-1](x_graph, edge_index, edge_weight)
            
            return adapter_embeddings, x_graph   
        else:
            return adapter_embeddings, None

    