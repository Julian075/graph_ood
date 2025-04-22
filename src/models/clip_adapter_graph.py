import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class CLIPAdapterGraph(nn.Module):
    def __init__(self, reduction_factor=8, device="cuda"):
        super().__init__()
        self.device = device
        self.reduction_factor = reduction_factor
        # Get CLIP embedding dimension
        self.clip_dim =512
        
        # Calculate bottleneck dimension
        self.bottleneck_dim = self.clip_dim // reduction_factor
        
        # Adapter layers (trainable)
        self.down_proj = nn.Linear(self.clip_dim, self.bottleneck_dim)
        self.non_linear = nn.ReLU()
        self.up_proj = nn.Linear(self.bottleneck_dim, self.clip_dim)
        self.layer_norm = nn.LayerNorm(self.clip_dim)
 
        # Move model to device
        self.to(device)
    
    @autocast()
    def forward(self, x):
        """
        Forward pass with automatic mixed precision
        Args:
            x: Input features [batch_size, clip_dim]
        Returns:
            adapted: Adapted features [batch_size, clip_dim]
        """
        # Store residual
        residual = x
        
        # Adapter forward pass
        x = self.down_proj(x)
        x = self.non_linear(x)
        x = self.up_proj(x)
        
        # Skip connection
        adapted = x + residual
        
        # Layer norm and final normalization
        adapted = self.layer_norm(adapted)
        adapted = F.normalize(adapted, dim=-1)
        
        return adapted 