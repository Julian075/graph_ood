import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.nn as gnn
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import knn_graph


class CLIPAdapterGraph(nn.Module):
    def __init__(self, reduction_factor=8, seed=42, device="cuda", gnn_hidden_dim=256, num_gnn_layers=2, k_neighbors=10):
        super().__init__()
        self.device = device
        self.reduction_factor = reduction_factor
        self.k_neighbors = k_neighbors
        # Get CLIP embedding dimension
        self.clip_dim = 512
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_gnn_layers = num_gnn_layers
        # Calculate bottleneck dimension
        self.bottleneck_dim = self.clip_dim // reduction_factor
        
        # Adapter layers (trainable)
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        self.down_proj = nn.Linear(self.clip_dim, self.bottleneck_dim)
        self.non_linear = nn.ReLU()
        self.up_proj = nn.Linear(self.bottleneck_dim, self.clip_dim)
        
        # GNN layers with layer normalization and residual connections
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First layer
        self.gnn_layers.append(gnn.GCNConv(self.clip_dim, self.gnn_hidden_dim))
        self.layer_norms.append(nn.LayerNorm(self.gnn_hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_gnn_layers-2):
            self.gnn_layers.append(gnn.GCNConv(self.gnn_hidden_dim, self.gnn_hidden_dim))
            self.layer_norms.append(nn.LayerNorm(self.gnn_hidden_dim))
        
        # Output layer
        self.gnn_layers.append(gnn.GCNConv(self.gnn_hidden_dim, self.clip_dim))
        self.layer_norms.append(nn.LayerNorm(self.clip_dim))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
 
        # Move model to device
        self.to(device)
        self.seed = seed
        self.set_seed(seed)


    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
    
    def create_adjacency_matrix(self, features):
        """Create KNN graph from features."""
        # Ensure input is float32
        features = features.float()
        
        # Compute KNN graph
        edge_index = knn_graph(features, k=self.k_neighbors, batch=None)
        
        # Calculate edge weights using cosine similarity
        row, col = edge_index
        edge_weight = F.cosine_similarity(features[row], features[col], dim=1)
        
        return edge_index, edge_weight
        
    def forward(self, x, training=True):
        """
        Forward pass with explicit precision handling
        Args:
            x: Input features [batch_size, clip_dim]
            training: Whether in training mode (use GNN) or inference mode (adapter only)
        Returns:
            adapted: Adapted features [batch_size, clip_dim]
            gnn_features: GNN processed features if training=True, None otherwise
        """
        # Ensure input is float32
        x = x.float()
        
        # Store original features
        original_features = x
        
        # Adapter forward pass
        x = self.down_proj(x)
        x = self.non_linear(x)
        x = self.up_proj(x)
        
        # Skip connection and normalization
        adapted = x + original_features
        adapted = self.alpha * adapted + (1 - self.alpha) * original_features
        adapted = F.normalize(adapted, dim=-1)
        
        if training:
            # Create graph from adapted features
            edge_index, edge_weight = self.create_adjacency_matrix(adapted)
            
            # GNN forward pass
            x_graph = adapted
            for i, gnn_layer in enumerate(self.gnn_layers):
                # Apply GNN layer
                x_graph = gnn_layer(x_graph, edge_index, edge_weight)
                # Apply layer norm
                x_graph = self.layer_norms[i](x_graph)
                # Apply non-linearity
                x_graph = F.relu(x_graph)
                # Normalize
                x_graph = F.normalize(x_graph, dim=-1)
                
                # Debug info
                #if torch.isnan(x_graph).any():
                #    print(f"NaN detected in GNN layer {i}")
                #    print(f"Edge weights stats - min: {edge_weight.min().item():.4f}, max: {edge_weight.max().item():.4f}")
                #    print(f"Layer output stats - min: {x_graph.min().item():.4f}, max: {x_graph.max().item():.4f}")
            
            return adapted, x_graph
        
        return adapted, None 