import torch
import torch.nn.functional as F
import os
import clip
from typing import List, Dict, Union, Tuple
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def encode_text(clip_model, device, text_prompts: List[str], normalize: bool = True) -> torch.Tensor:
    """Encode text prompts using CLIP with explicit precision handling"""
    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts).to(device)
        text_features = clip_model.encode_text(text_tokens).float()  # Ensure float32
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
    return text_features

def compute_contrastive_loss(image_features, text_features, labels, temperature):
    """
    Compute contrastive loss between image and text features.
    
    Args:
        image_features: Tensor of shape [batch_size, feature_dim]
        text_features: Tensor of shape [num_classes, feature_dim]
        labels: Tensor of shape [batch_size] containing class indices
        temperature: Temperature parameter for scaling logits
    """
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.T)  # [batch_size, num_classes]
    # Scale logits by temperature
    logits = logits / temperature
    
    # Get positive pair logits using the labels
    positive_logits = logits[torch.arange(logits.size(0)), labels]  # [batch_size]
    
    # Compute exp(logits) for all pairs
    exp_logits = torch.exp(logits)  # [batch_size, num_classes]
    
    # Sum exp(logits) for denominator
    sum_exp_logits = exp_logits.sum(dim=1)  # [batch_size]
    
    # Compute log(exp(pos_logits) / sum(exp(logits))) = pos_logits - log(sum(exp(logits)))
    loss = -positive_logits + torch.log(sum_exp_logits)
    
    # Average over the batch
    return loss.mean()

def prepare_data(data: Dict, batch_size: int, class_to_idx: Dict) -> tuple[DataLoader, list]:
    """
    Prepare data loaders for training/validation.
    
    Args:
        data: Dictionary containing:
            - features: Tensor of shape [N, feature_dim]
            - labels: List or Tensor of labels
            - paths: List of image paths
        batch_size: Batch size for the dataloader
        class_to_idx: Dictionary mapping class names to indices
    """
    # Convert labels to indices if they're not already
    labels = data['labels']
    if isinstance(labels, list):
        # Convert labels to indices
        label_indices = [class_to_idx[str(label) if isinstance(label, tuple) else label] for label in labels]
        labels = torch.tensor(label_indices)
    elif isinstance(labels, torch.Tensor) and labels.dtype != torch.int64:
        # If tensor but not indices, convert to indices
        label_indices = [class_to_idx[str(label.item())] for label in labels]
        labels = torch.tensor(label_indices)
        
    # Create dataset with features and labels
    dataset = TensorDataset(
        data['features'],
        labels,
        torch.arange(len(data['paths']))  # Create indices for paths lookup
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    return dataloader, data['paths']  # Return paths separately for reference

def set_seed(seed: int):
    """Set seed for reproducibility with additional settings."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, val_acc: float, config, output_dir: str, name_model: str, use_synthetic: bool) -> str:
    """Save model checkpoint with accuracy in filename."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config
    }
    
    # Format accuracy for filename
    acc_str = f"{val_acc:.2f}".replace(".", "_")
    
    if use_synthetic:
        checkpoint_path = os.path.join(
            output_dir, name_model,
            f'adapter_checkpoint_synthetic_acc_{acc_str}.pt'
        )
    else:
        checkpoint_path = os.path.join(
            output_dir, name_model,
            f'adapter_checkpoint_acc_{acc_str}.pt'
        )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path: str) -> float:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['val_acc']

def load_data(feature_dir: str, use_synthetic: bool, class_to_idx: Dict) -> Dict:
    """Load and prepare data from files"""
    feature_file = os.path.join(feature_dir, "real_data.pt")
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found at {feature_file}")
        
    all_data = torch.load(feature_file)
    if use_synthetic:
        all_data = merge_synthetic_data(all_data, feature_dir, class_to_idx)
    return all_data

def merge_synthetic_data(all_data: Dict, feature_dir: str, class_to_idx: Dict) -> Dict:
    """Merge synthetic data with real data"""
    feature_file_synthetic = os.path.join(feature_dir, "synthetic_features.pt")
    if not os.path.exists(feature_file_synthetic):
        raise FileNotFoundError(f"Feature file not found at {feature_file_synthetic}")
        
    all_data_synthetic = torch.load(feature_file_synthetic)
    all_data['train']['features'] = torch.cat([all_data['train']['features'], all_data_synthetic['features']])
    
    # Convert and merge labels
    train_labels = convert_labels(all_data['train']['labels'], class_to_idx)
    synthetic_labels = convert_labels(all_data_synthetic['labels'], class_to_idx)
    all_data['train']['labels'] = torch.cat([train_labels, synthetic_labels])
    all_data['train']['paths'] = all_data['train']['paths'] + all_data_synthetic['paths']
    
    return all_data

def convert_labels(labels: Union[List, torch.Tensor], class_to_idx: Dict) -> torch.Tensor:
    """Convert labels to indices"""
    if isinstance(labels, list):
        return torch.tensor([class_to_idx[str(label) if isinstance(label, tuple) else label] for label in labels])
    elif isinstance(labels, torch.Tensor) and labels.dtype != torch.int64:
        return torch.tensor([class_to_idx[str(label.item())] for label in labels])
    return labels

def prepare_dataloaders(all_data: Dict, batch_size: int, class_to_idx: Dict) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and validation dataloaders"""
    train_loader, _ = prepare_data(
        {'features': all_data['train']['features'], 
         'labels': all_data['train']['labels'], 
         'paths': all_data['train']['paths']}, 
        batch_size,
        class_to_idx
    )
    val_loader, _ = prepare_data(
        {'features': all_data['val']['features'], 
         'labels': all_data['val']['labels'], 
         'paths': all_data['val']['paths']}, 
        batch_size,
        class_to_idx
    )
    return train_loader, val_loader 