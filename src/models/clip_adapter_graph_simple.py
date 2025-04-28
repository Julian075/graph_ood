import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import knn_graph
import numpy as np

class CLIPAdapterGraphSimple(nn.Module):
    def __init__(
        self,
        reduction_factor=32,
        device="cuda",
        gnn_hidden_dim=256,
        num_gnn_layers=2,
        seed=None,
        k_neighbors=10,
        num_classes=46
    ):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.device = device
        self.reduction_factor = reduction_factor
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.k_neighbors = k_neighbors
        
       
        self.clip_feature_dim = 512
        # Calculate bottleneck dimension
        self.bottleneck_dim = self.clip_feature_dim // reduction_factor
        #parameter
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        # Adapter layers
        self.down_proj = nn.Linear(self.clip_feature_dim, self.bottleneck_dim)
        self.up_proj = nn.Linear(self.bottleneck_dim, self.clip_feature_dim)
        
        # GNN layers
        self.num_classes = num_classes

        self.gnn_layers = nn.ModuleList()
        
        if self.num_gnn_layers >  1:
            self.gnn_layers.append(GCNConv(self.clip_feature_dim, self.gnn_hidden_dim))
        for _ in range(num_gnn_layers - 2):
            self.gnn_layers.append(GCNConv(self.gnn_hidden_dim, self.gnn_hidden_dim))
        if self.num_gnn_layers > 1:
            self.gnn_layers.append(GCNConv(self.gnn_hidden_dim, self.num_classes))
        else:
            self.gnn_layers.append(GCNConv(self.clip_feature_dim, self.num_classes))

        
        self.set_seed(seed)
        
        self.to(device)

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
        # Apply adapter
        x=x.float()
        identity = x
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        
        # Add residual connection
        x = x + identity
        x= self.alpha * x + (1 - self.alpha) * identity
        #x=F.normalize(x, dim=1)
        
        # If edge_index is provided, apply GNN layers
        if training:
            adapter_output = x
            edge_index, edge_weight = self.create_adjacency_matrix(x)
            # Apply GNN layers
            for _, gnn_layer in enumerate(self.gnn_layers[:-1]):
                x = gnn_layer(x, edge_index, edge_weight)
                x = F.relu(x)
            x = self.gnn_layers[-1](x, edge_index, edge_weight)
            x = F.softmax(x, dim=1)
            return adapter_output, x 
        else:
            return x, x