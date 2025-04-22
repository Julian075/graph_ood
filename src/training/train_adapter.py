import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm
import clip
from typing import List
class CLIPAdapterTrainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.feature_dir = config.feature_dir
        self.prompt_template = config.prompt_template
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.clip_adapter['learning_rate']
        )
        self.temperature = config.clip_adapter['temperature']
        self.clip, _ = clip.load(config.clip_model, device=config.device)
        self.clip.eval()  # Set to eval mode to freeze parameters
        
        # Initialize gradient scaler for AMP
        self.scaler = GradScaler()
    
    def encode_text(self, text_prompts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode text prompts using CLIP"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.clip.encode_text(text_tokens)
            if normalize:
                text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return text_features
        
    def compute_cross_entropy_loss(self, image_features, text_features, labels):
        """
        Compute standard cross entropy loss between image and text features.
        
        Args:
            image_features: Tensor of shape [batch_size, feature_dim]
            text_features: Tensor of shape [num_classes, feature_dim]
            labels: Tensor of shape [batch_size] containing class indices
        """
        # Compute similarity scores
        logits = torch.matmul(image_features, text_features.T)/self.temperature
        
        # Apply temperature scaling
        logits = logits
        
        # Use standard cross entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def compute_contrastive_loss(self, image_features, text_features, labels):
        """
   
        Args:
            image_features: Tensor of shape [batch_size, feature_dim]
            text_features: Tensor of shape [num_classes, feature_dim]
            labels: Tensor of shape [batch_size] containing class indices
        """
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T)  # [batch_size, num_classes]
        
        # Scale logits by temperature
        logits = logits / self.temperature
        
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
    
    def prepare_data(self, data, batch_size,class_to_idx):
        """
        Prepare data loaders for training/validation.
        
        Args:
            data: Dictionary containing:
                - features: Tensor of shape [N, feature_dim]
                - labels: List or Tensor of labels
                - paths: List of image paths
            batch_size: Batch size for the dataloader
        """
        # Convert labels to tensor if they're not already
        # Create mapping for all classes
        
        labels = data['labels']
        labels = torch.tensor([class_to_idx[label] for label in labels])
            
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
    
    def train_epoch(self, train_loader, text_features):
        """Train for one epoch with automatic mixed precision."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Debug: verificar parámetros entrenables al inicio de la época
        #trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #print(f"Número de parámetros entrenables: {trainable_params}")
        #params_before = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        
        for batch_features, batch_labels, _ in tqdm(train_loader, desc="Training"):
            # Move batch to device and normalize
            batch_features = batch_features.to(self.device)
            batch_features = F.normalize(batch_features, dim=-1)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass with automatic mixed precision
            with autocast():
                adapted_features = self.model(batch_features)
                #loss = self.compute_cross_entropy_loss(adapted_features, text_features, batch_labels)+
                loss = self.compute_contrastive_loss(adapted_features, text_features, batch_labels)
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Optimizer step with gradient scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Debug: verificar si los parámetros cambiaron
            #if num_batches == 0:
            #    print("\nCambios en parámetros después del primer batch:")
            #    for name, param in self.model.named_parameters():
            #        if param.requires_grad:
            #            param_change = (param - params_before[name]).norm().item()
            #            print(f"{name}: {param_change:.6f}")
            #
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('nan')
    
    
    def validate(self, val_loader, text_features):
        """Validate the model."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels,_ in tqdm(val_loader, desc="Validating"):
                # Move batch to device
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass with autocast for consistent dtype handling
                with autocast():
                    adapted_features = self.model(batch_features)
                    similarity = torch.matmul(adapted_features, text_features.T)
                
                # Get predictions
                predictions = torch.argmax(similarity, dim=1)
                
                # Update metrics
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        accuracy = correct / total
        return accuracy
    
    def save_checkpoint(self, val_acc):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.output_dir,'clip_adapter',
            f'adapter_checkpoint.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['val_acc']
    
    def set_seed(self):
        """Set seed for reproducibility."""
        torch.manual_seed(self.config.clip_adapter['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.clip_adapter['seed'])
    
    def train(self, classes_names):
        """Main training loop with early stopping."""
        # Load features from the feature directory
        self.set_seed()

        unique_classes = sorted({class_name for _, class_name in classes_names})
        #unique_classes.append('empty')
        class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}

        feature_file = os.path.join(self.feature_dir, "real_data.pt")
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found at {feature_file}")
        all_data = torch.load(feature_file)

        # Get train and val splits
        train_features = all_data['train']['features']
        train_labels = all_data['train']['labels']
        val_features = all_data['val']['features']
        val_labels = all_data['val']['labels']

        print("\nStarting training...")
        print(f"Training samples: {len(train_features)}")
        print(f"Validation samples: {len(val_features)}")
        
        # Prepare data loaders
        train_loader, train_paths = self.prepare_data({'features': train_features, 'labels': train_labels, 'paths': all_data['train']['paths']}, self.config.clip_adapter['batch_size'],class_to_idx)
        val_loader, val_paths = self.prepare_data({'features': val_features, 'labels': val_labels, 'paths': all_data['val']['paths']}, self.config.clip_adapter['batch_size'],class_to_idx)

        templates = [self.prompt_template] if isinstance(self.prompt_template, str) else self.prompt_template
        all_text_features = []
        with torch.no_grad():
            for template in templates:
                prompts = [template.format(class_name) for class_name in unique_classes]
                text_features = []
                
                #print(f"\nGenerating text embeddings for template: '{template}'")
                for i in range(0, len(prompts), self.config.clip_adapter['batch_size']):
                    batch_prompts = prompts[i:i + self.config.clip_adapter['batch_size']]
                    batch_features = self.encode_text(batch_prompts)
                    text_features.append(batch_features)
                
                # Concatenate and normalize features for this template
                template_features = torch.cat(text_features, dim=0)
                template_features = torch.nn.functional.normalize(template_features, dim=-1)
                all_text_features.append(template_features)
        
        # Average text features across templates if using multiple
        if len(all_text_features) > 1:
            text_features = torch.stack(all_text_features).mean(dim=0)
            # Re-normalize after averaging
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
        else:
            text_features = all_text_features[0]
            #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config.clip_adapter['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config.clip_adapter['num_epochs']}")
            
            # Train epoch
            train_loss = self.train_epoch(train_loader, text_features)
            print(f"Training loss: {train_loss:.4f}")
            
            # Validate
            val_acc = self.validate(val_loader, text_features)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(val_acc)
            else:
                patience_counter += 1
                
                print(f"Validation accuracy did not improve. Patience: {patience_counter}/{self.config.clip_adapter['patience']}")
            
            if patience_counter >= self.config.clip_adapter['patience']:
                print("\nEarly stopping triggered!")
                break
        
        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
        return best_val_acc
