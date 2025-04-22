import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast
import clip
from tqdm import tqdm
import os
from typing import List, Dict, Tuple, Optional

class CLIPAdapterEvaluator:
    def __init__(self, checkpoint_path: str, config):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to the trained adapter checkpoint
            config: Configuration object containing model settings
        """
        self.config = config
        self.device = config.device
        self.prompt_template = config.prompt_template
        
        # Load CLIP model
        self.clip, _ = clip.load(config.clip_model, device=config.device)
        self.clip.eval()
        
        # Load adapter model from checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model = checkpoint['model_state_dict']
        self.model.to(self.device)
        self.model.eval()
        
    def encode_text(self, text_prompts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode text prompts using CLIP"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.clip.encode_text(text_tokens)
            if normalize:
                text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def prepare_data(self, features: torch.Tensor, batch_size: int) -> DataLoader:
        """
        Prepare data loader for evaluation.
        
        Args:
            features: Tensor of shape [N, feature_dim]
            batch_size: Batch size for the dataloader
        """
        dataset = TensorDataset(features)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def get_text_features(self, class_names: List[str]) -> torch.Tensor:
        """
        Get text features for all classes.
        
        Args:
            class_names: List of class names
        Returns:
            text_features: Normalized text features [num_classes, feature_dim]
        """
        templates = [self.prompt_template] if isinstance(self.prompt_template, str) else self.prompt_template
        all_text_features = []
        
        with torch.no_grad():
            for template in templates:
                prompts = [template.format(class_name) for class_name in class_names]
                text_features = []
                
                for i in range(0, len(prompts), self.config.clip_adapter['batch_size']):
                    batch_prompts = prompts[i:i + self.config.clip_adapter['batch_size']]
                    batch_features = self.encode_text(batch_prompts)
                    text_features.append(batch_features)
                
                template_features = torch.cat(text_features, dim=0)
                template_features = F.normalize(template_features, dim=-1)
                all_text_features.append(template_features)
        
        # Average text features across templates if using multiple
        if len(all_text_features) > 1:
            text_features = torch.stack(all_text_features).mean(dim=0)
            text_features = F.normalize(text_features, dim=-1)
        else:
            text_features = all_text_features[0]
            
        return text_features
    
    def predict(self, 
                features: torch.Tensor, 
                class_names: List[str],
                return_similarities: bool = False
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict classes for input features.
        
        Args:
            features: Input features [N, feature_dim]
            class_names: List of class names
            return_similarities: Whether to return similarity scores
            
        Returns:
            predictions: Predicted class indices [N]
            similarities: Optional similarity scores [N, num_classes]
        """
        # Get text features for classes
        text_features = self.get_text_features(class_names)
        
        # Prepare data loader
        data_loader = self.prepare_data(features, self.config.clip_adapter['batch_size'])
        
        all_predictions = []
        all_similarities = [] if return_similarities else None
        
        # Predict in batches
        with torch.no_grad():
            for batch_features, in tqdm(data_loader, desc="Predicting"):
                batch_features = batch_features.to(self.device)
                batch_features = F.normalize(batch_features, dim=-1)
                
                with autocast():
                    adapted_features = self.model(batch_features)
                    similarity = torch.matmul(adapted_features, text_features.T)
                
                predictions = torch.argmax(similarity, dim=1)
                all_predictions.append(predictions.cpu())
                
                if return_similarities:
                    all_similarities.append(similarity.cpu())
        
        # Concatenate results
        predictions = torch.cat(all_predictions)
        similarities = torch.cat(all_similarities) if return_similarities else None
        
        return predictions, similarities
    
    def evaluate(self, 
                features: torch.Tensor, 
                labels: torch.Tensor,
                class_names: List[str]
               ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            features: Input features [N, feature_dim]
            labels: Ground truth labels [N]
            class_names: List of class names
            
        Returns:
            metrics: Dictionary containing evaluation metrics
        """
        predictions, _ = self.predict(features, class_names)
        
        # Calculate accuracy
        correct = (predictions == labels).sum().item()
        total = len(labels)
        accuracy = correct / total
        
        # Calculate per-class accuracy
        per_class_acc = {}
        for i, class_name in enumerate(class_names):
            mask = labels == i
            if mask.sum() > 0:
                class_correct = (predictions[mask] == labels[mask]).sum().item()
                class_total = mask.sum().item()
                per_class_acc[class_name] = class_correct / class_total
        
        return {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc
        } 