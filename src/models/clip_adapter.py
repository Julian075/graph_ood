import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import clip

class CLIPAdapter(nn.Module):
    def __init__(self, reduction_factor=8, device="cuda", clip_model="ViT-B/32"):
        super().__init__()
        self.device = device
        self.reduction_factor = reduction_factor
        # Load CLIP model
        self.clip, _ = clip.load(clip_model, device=device)
        self.clip.eval()  # Set to eval mode to freeze parameters
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Get CLIP embedding dimension
        self.clip_dim = self.clip.visual.output_dim
        
        # Calculate bottleneck dimension
        self.bottleneck_dim = self.clip_dim // reduction_factor
        
        # Adapter layers (trainable)
        self.down_proj = nn.Linear(self.clip_dim, self.bottleneck_dim)
        self.non_linear = nn.ReLU()
        self.up_proj = nn.Linear(self.bottleneck_dim, self.clip_dim)
        self.layer_norm = nn.LayerNorm(self.clip_dim)
        
        # Move to device and convert to half precision
        self.down_proj = self.down_proj.to(device=device, dtype=torch.float16)
        self.up_proj = self.up_proj.to(device=device, dtype=torch.float16)
        self.layer_norm = self.layer_norm.to(device=device, dtype=torch.float16)
    

    def encode_text(self, text_prompts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode text prompts using CLIP"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.clip.encode_text(text_tokens)
            if normalize:
                text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return text_features

    
    def forward(self, features, text_features):
        """
        Forward pass computing similarity between adapted features and text.
        
        Args:
            features: Pre-extracted image features
            text: List of text prompts
            
        Returns:
            similarity: Similarity matrix between adapted features and text
            adapted_features: Adapted image features
            text_features: Text features
        """
        # Move features to device
        features = features.to(self.device)
        
        # Adapter forward pass
        adapted = self.down_proj(features)
        adapted = self.non_linear(adapted)
        adapted = self.up_proj(adapted)
        
        # Skip connection
        adapted = adapted + features
        
        # Layer norm and final normalization
        adapted = self.layer_norm(adapted)
        adapted_features = F.normalize(adapted, dim=-1)
        
        # Compute similarity
        similarity = torch.matmul(adapted_features, text_features.T)
        
        return similarity, adapted_features, text_features 