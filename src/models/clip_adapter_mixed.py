import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import torch_geometric.nn as gnn
from torch_geometric.nn import knn_graph
from typing import Tuple, Optional, Dict

class CLIPAdapterMixed(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        reduction_factor: int = 8,
        k_neighbors: int = 10,
        gnn_hidden_dim: int = 256,
        num_gnn_layers: int = 2,
        temperature: float = 0.07,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.num_classes = num_classes
        
        # CLIP embedding dimension
        self.clip_dim = 512
        self.bottleneck_dim = self.clip_dim // reduction_factor
        
        # Adapter layers
        self.adapter = nn.Sequential(
            nn.Linear(self.clip_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, self.clip_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Node classifier for real samples
        self.node_classifier = nn.Linear(self.clip_dim, num_classes)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First GNN layer
        self.gnn_layers.append(gnn.GCNConv(self.clip_dim, gnn_hidden_dim))
        self.layer_norms.append(nn.LayerNorm(gnn_hidden_dim))
        
        # Hidden layers
        for _ in range(num_gnn_layers - 2):
            self.gnn_layers.append(gnn.GCNConv(gnn_hidden_dim, gnn_hidden_dim))
            self.layer_norms.append(nn.LayerNorm(gnn_hidden_dim))
        
        # Output layer
        self.gnn_layers.append(gnn.GCNConv(gnn_hidden_dim, self.clip_dim))
        self.layer_norms.append(nn.LayerNorm(self.clip_dim))
        
        self.dropout = nn.Dropout(0.1)
        self.to(device)

    def create_knn_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create KNN graph from features."""
        edge_index = knn_graph(x, k=self.k_neighbors, batch=None)
        row, col = edge_index
        edge_weight = F.cosine_similarity(x[row], x[col], dim=1)
        return edge_index, edge_weight

    def forward(
        self, 
        x: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        is_synthetic: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through adapter and GNN.
        
        Args:
            x: Input image features [batch_size, clip_dim]
            text_embeddings: Text embeddings [num_classes, clip_dim] (optional)
            is_synthetic: Boolean mask indicating synthetic samples (optional)
            training: Whether in training mode
            
        Returns:
            Dictionary containing:
            - 'adapter_embeddings': Embeddings after adapter
            - 'node_embeddings': Embeddings after GNN (if training)
            - 'logits': Classification logits (if not training)
            - 'final_embeddings': Combined embeddings (if training)
        """
        # Ensure input is float32
        x = x.float()
        original_features = x

        # Adapter forward pass
        adapted = self.adapter(x)
        adapted = adapted + original_features  # Skip connection
        adapted = self.alpha * adapted + (1 - self.alpha) * original_features
        adapter_embeddings = F.normalize(adapted, dim=-1)

        if not training:
            # In testing, only use adapter embeddings for classification
            if text_embeddings is not None:
                text_embeddings = F.normalize(text_embeddings, dim=-1)
                logits = torch.matmul(adapter_embeddings, text_embeddings.T) / self.temperature
                return {'logits': logits, 'adapter_embeddings': adapter_embeddings}
            return {'adapter_embeddings': adapter_embeddings}

        # Training mode: process through GNN
        edge_index, edge_weight = self.create_knn_graph(adapter_embeddings)
        
        # GNN forward pass
        x_graph = adapter_embeddings
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            x_graph = gnn_layer(x_graph, edge_index, edge_weight)
            x_graph = layer_norm(x_graph)
            x_graph = F.relu(x_graph)
            x_graph = self.dropout(x_graph)
            x_graph = F.normalize(x_graph, dim=-1)

        node_embeddings = x_graph
        
        # Final embeddings combine adapter and GNN outputs
        final_embeddings = self.alpha * adapter_embeddings + (1 - self.alpha) * node_embeddings
        final_embeddings = F.normalize(final_embeddings, dim=-1)

        # Node classification logits for real samples
        if is_synthetic is not None:
            real_node_logits = self.node_classifier(node_embeddings[~is_synthetic])
        else:
            real_node_logits = None

        return {
            'adapter_embeddings': adapter_embeddings,
            'node_embeddings': node_embeddings,
            'final_embeddings': final_embeddings,
            'real_node_logits': real_node_logits
        }

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
        is_synthetic: torch.Tensor,
        loss_weights: Dict[str, float] = {'contrastive': 1.0, 'mmd': 1.0, 'node_ce': 1.0}
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for training.
        
        Args:
            outputs: Forward pass outputs
            text_embeddings: Text embeddings [num_classes, clip_dim]
            labels: Class labels
            is_synthetic: Boolean mask for synthetic samples
            loss_weights: Weights for each loss component
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        
        # 1. Contrastive loss between real adapted embeddings and text embeddings
        real_adapter_emb = outputs['adapter_embeddings'][~is_synthetic]
        real_labels = labels[~is_synthetic]
        
        # Get text embeddings for real labels
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        label_embeddings = text_embeddings[real_labels]
        
        # Compute similarity
        similarity = torch.matmul(real_adapter_emb, text_embeddings.T) / self.temperature
        losses['contrastive'] = F.cross_entropy(similarity, real_labels) * loss_weights['contrastive']
        
        # 2. MMD loss between real and synthetic node embeddings
        if is_synthetic.any() and (~is_synthetic).any():
            real_nodes = outputs['node_embeddings'][~is_synthetic]
            synth_nodes = outputs['node_embeddings'][is_synthetic]
            losses['mmd'] = self.mmd_loss(real_nodes, synth_nodes) * loss_weights['mmd']
        else:
            losses['mmd'] = torch.tensor(0.0, device=self.device)
        
        # 3. Cross entropy for real node classification
        if outputs['real_node_logits'] is not None:
            losses['node_ce'] = F.cross_entropy(
                outputs['real_node_logits'],
                real_labels
            ) * loss_weights['node_ce']
        
        # Total loss
        losses['total'] = sum(loss for loss in losses.values())
        
        return losses

    def mmd_loss(
        self, 
        real_features: torch.Tensor,
        synthetic_features: torch.Tensor,
        kernel_multiplier: float = 2.0
    ) -> torch.Tensor:
        """Calculate MMD loss between real and synthetic features."""
        n_samples = real_features.size(0) + synthetic_features.size(0)
        n_features = real_features.size(1)
        sigma = kernel_multiplier * np.sqrt(n_features)

        real_kernel = torch.exp(-torch.cdist(real_features, real_features) / (2 * sigma ** 2))
        synthetic_kernel = torch.exp(-torch.cdist(synthetic_features, synthetic_features) / (2 * sigma ** 2))
        cross_kernel = torch.exp(-torch.cdist(real_features, synthetic_features) / (2 * sigma ** 2))

        mmd = (real_kernel.mean() + synthetic_kernel.mean() - 2 * cross_kernel.mean())
        return mmd 