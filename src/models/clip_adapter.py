import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from src.feature_extraction.feature_extractor import FeatureExtractor

class CLIPAdapter(nn.Module):
    def __init__(self, clip_model, adapter_dim=512, reduction_factor=8, device="cuda"):
        super().__init__()
        self.device = device
        
        # CLIP components (frozen)
        self.clip = clip_model
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
    
    def adapt_features(self, features):
        """Adapt pre-extracted image features with skip connection."""
        features = features.to(self.device)
        
        # Adapter forward pass
        adapted = self.down_proj(features)
        adapted = self.non_linear(adapted)
        adapted = self.up_proj(adapted)
        
        # Skip connection
        adapted = adapted + features
        
        # Layer norm and final normalization
        adapted = self.layer_norm(adapted)
        adapted = F.normalize(adapted, dim=-1)
        
        return adapted
    
    def encode_image(self, image):
        """Extract and adapt image features through CLIP and adapter."""
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)
        return self.adapt_features(image_features)
    
    def encode_text(self, text, batch_size=32):
        """Extract text features through CLIP (frozen)."""
        if isinstance(text, str):
            text = [text]
        
        text_features = []
        with torch.no_grad():
            for i in range(0, len(text), batch_size):
                batch_text = text[i:i + batch_size]
                tokens = clip.tokenize(batch_text).to(self.device)
                batch_features = self.clip.encode_text(tokens)
                batch_features = F.normalize(batch_features, dim=-1)
                text_features.append(batch_features)
            
            # Concatenate all batches
            text_features = torch.cat(text_features, dim=0)
        
        return text_features
    
    def forward(self, features, text):
        """Forward pass computing similarity between adapted features and text."""
        adapted_features = self.adapt_features(features)
        text_features = self.encode_text(text)
        
        # Compute similarity
        similarity = torch.matmul(adapted_features, text_features.T)
        
        return similarity, adapted_features, text_features

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_features, text_features, labels):
        """
        Compute InfoNCE/NTXent loss between image and text features.
        
        Args:
            image_features: Tensor of shape [batch_size, feature_dim]
            text_features: Tensor of shape [num_classes, feature_dim]
            labels: Tensor of shape [batch_size] containing class indices
            
        Returns:
            Loss value (scalar)
        """
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # For each image, its positive pair is the text embedding of its class
        positive_logits = logits[torch.arange(len(labels)), labels]
        
        # InfoNCE loss: -log(exp(pos_logit) / sum(exp(all_logits)))
        # = -pos_logit + log(sum(exp(all_logits)))
        nll = -positive_logits + torch.logsumexp(logits, dim=1)
        
        # Average over the batch
        loss = nll.mean()
        
        return loss

def create_clip_adapter(device="cuda", reduction_factor=8):
    """Helper function to create a CLIP adapter instance.
    
    Args:
        device: Device to run the model on (default: "cuda")
        reduction_factor: Factor to reduce the dimension in adapter (default: 8)
    """
    feature_extractor = FeatureExtractor(device)
    model = CLIPAdapter(feature_extractor.model, device=device, reduction_factor=reduction_factor)
    criterion = ContrastiveLoss()
    
    return model, criterion 