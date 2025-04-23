import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import torch_geometric.nn as gnn
from torch_geometric.utils import dense_to_sparse


class CLIPAdapterGraph(nn.Module):
    def __init__(self, reduction_factor=8, seed=42, device="cuda", gnn_hidden_dim=256, num_gnn_layers=2):
        super().__init__()
        self.device = device
        self.reduction_factor = reduction_factor
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
    
    @autocast()    
    def create_adjacency_matrix(self, features):
        """
        Create adjacency matrix based on cosine similarity between node features
        Args:
            features: Node features [num_nodes, feature_dim]
        Returns:
            edge_index: Sparse adjacency matrix
            edge_weight: Edge weights based on cosine similarity
        """
        # Compute cosine similarity matrix
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # Remove self-loops and ensure positive weights
        similarity_matrix.fill_diagonal_(0)
        
        # Apply ReLU to ensure positive weights and sparsity
        similarity_matrix = F.relu(similarity_matrix)
        
        # Add small epsilon to avoid numerical instability
        similarity_matrix = similarity_matrix + 1e-6
        
        # Row-normalize the similarity matrix
        row_sum = similarity_matrix.sum(dim=-1, keepdim=True)
        similarity_matrix = similarity_matrix / row_sum.clamp(min=1e-6)
        
        # Convert to sparse format
        edge_index, edge_weight = dense_to_sparse(similarity_matrix)
        
        return edge_index, edge_weight
        
    def forward(self, x, training=True):
        """
        Forward pass with optional GNN during training
        Args:
            x: Input features [batch_size, clip_dim]
            training: Whether in training mode (use GNN) or inference mode (adapter only)
        Returns:
            adapted: Adapted features [batch_size, clip_dim]
            gnn_features: GNN processed features if training=True, None otherwise
        """
        # Store original features
        original_features = x
        #print(f"Input features stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        # Adapter forward pass
        x = self.down_proj(x)
        #print(f"After down_proj - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        
        x = self.non_linear(x)
        #print(f"After non_linear - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        
        x = self.up_proj(x)
        #print(f"After up_proj - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        
        # Skip connection and normalization
        adapted = x + original_features
        #print(f"After skip connection - min: {adapted.min().item():.4f}, max: {adapted.max().item():.4f}, mean: {adapted.mean().item():.4f}")
        
        adapted = self.alpha * adapted + (1 - self.alpha) * original_features
        #print(f"After alpha interpolation - min: {adapted.min().item():.4f}, max: {adapted.max().item():.4f}, mean: {adapted.mean().item():.4f}")
        
        adapted = F.normalize(adapted, dim=-1)
        #print(f"After normalization - min: {adapted.min().item():.4f}, max: {adapted.max().item():.4f}, mean: {adapted.mean().item():.4f}")
        
        if training:
            # Create graph from adapted features
            edge_index, edge_weight = self.create_adjacency_matrix(adapted)
            #print(f"Edge weights stats - min: {edge_weight.min().item():.4f}, max: {edge_weight.max().item():.4f}, mean: {edge_weight.mean().item():.4f}")
            
            # GNN forward pass
            x_graph = adapted
            for i, gnn_layer in enumerate(self.gnn_layers):
                x_graph = gnn_layer(x_graph, edge_index, edge_weight)
                x_graph = self.layer_norms[i](x_graph)
                x_graph = F.relu(x_graph)
                x_graph = F.normalize(x_graph, dim=-1)
                #print(f"After GNN layer {i} - min: {x_graph.min().item():.4f}, max: {x_graph.max().item():.4f}, mean: {x_graph.mean().item():.4f}")
            
            return adapted, x_graph
        
        return adapted, None 