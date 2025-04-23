import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np

class CLIPAdapter(nn.Module):
    def __init__(self, reduction_factor=8, seed=42, device="cuda"):
        super().__init__()
        self.device = device
        self.reduction_factor = reduction_factor
        # Get CLIP embedding dimension
        self.clip_dim =512
        
        # Calculate bottleneck dimension
        self.bottleneck_dim = self.clip_dim // reduction_factor
        
        #parameter
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        # Adapter layers (trainable)
        self.down_proj = nn.Linear(self.clip_dim, self.bottleneck_dim)
        self.non_linear = nn.ReLU()
        self.up_proj = nn.Linear(self.bottleneck_dim, self.clip_dim)
        #self.layer_norm = nn.LayerNorm(self.clip_dim)
        #self.logit_scale_CLIP = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
 
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
    
    #@autocast()   
    def forward(self, x):
        """
        Forward pass with explicit precision handling
        Args:
            x: Input features [batch_size, clip_dim]
        Returns:
            adapted: Adapted features [batch_size, clip_dim]
        """
        # Ensure input is float32
        x = x.float()
        
        # Store residual
        residual = x
        
        # Adapter forward pass
        x = self.down_proj(x)
        x = self.non_linear(x)
        x = self.up_proj(x)
        
        # Skip connection
        adapted = x + residual
        
        # Alpha interpolation
        adapted = self.alpha * adapted + (1 - self.alpha) * residual
        
        # Final normalization
        adapted = F.normalize(adapted, dim=-1)
        
        # Debug info
        #if torch.isnan(adapted).any():
        #    print("NaN detected in adapter output")
        #    print(f"Input stats - min: {residual.min().item():.4f}, max: {residual.max().item():.4f}")
        #    print(f"Adapter output stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        #    print(f"Final output stats - min: {adapted.min().item():.4f}, max: {adapted.max().item():.4f}")
        
        return adapted 