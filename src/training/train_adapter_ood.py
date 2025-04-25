import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import os
from tqdm import tqdm
import clip
from typing import List, Dict, Tuple, Iterator
from src.models.clip_adapter_ood import CLIPAdapterOOD
import numpy as np
import random

class MixedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that ensures each batch contains a mix of real and synthetic data.
    Automatically adjusts batch size based on dataset sizes while maintaining a minimum
    number of samples from each dataset.
    """
    def __init__(self, real_size: int, synthetic_size: int, max_batch_size: int = None):
        self.real_size = real_size
        self.synthetic_size = synthetic_size
        self.larger_size = max(real_size, synthetic_size)
        self.smaller_size = min(real_size, synthetic_size)
        self.is_real_larger = real_size >= synthetic_size

        # Ensure we have at least 4 samples from smaller dataset
        min_samples_smaller = min(4, self.smaller_size)
        
        # Calculate a reasonable batch size
        suggested_batch_size = min(
            max(8, self.smaller_size // 2),  # Try to get at least 2 batches from smaller dataset
            max_batch_size if max_batch_size is not None else 256  # Cap at max_batch_size or 256
        )
        self.batch_size = suggested_batch_size

        # Ensure minimum samples from smaller dataset (at least 2, but no more than half batch)
        self.min_smaller = min(
            max(2, min(self.batch_size // 4, self.smaller_size)),
            self.batch_size // 2
        )

        # Calculate minimum samples from larger dataset
        ratio = max(1.0, self.larger_size / max(1, self.smaller_size))
        remaining_space = self.batch_size - self.min_smaller
        self.min_larger = min(
            max(2, int(remaining_space * 0.8 * ratio)),  # Scale by ratio, at least 2 samples
            min(remaining_space, self.larger_size)  # But no more than available or remaining space
        )

        # Ensure total minimum samples don't exceed batch size
        if self.min_larger + self.min_smaller > self.batch_size:
            # Reduce larger minimum while keeping smaller minimum
            self.min_larger = self.batch_size - self.min_smaller

        # Print batch composition for verification
        print(f"\nBatch Sampler Configuration:")
        print(f"Real dataset size: {real_size}")
        print(f"Synthetic dataset size: {synthetic_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Minimum samples from smaller dataset: {self.min_smaller}")
        print(f"Minimum samples from larger dataset: {self.min_larger}")
        print(f"{'Real' if self.is_real_larger else 'Synthetic'} dataset is larger")

    def __iter__(self) -> Iterator[List[int]]:
        # Create lists of indices for real and synthetic data
        real_indices = list(range(self.real_size))
        synthetic_indices = list(range(self.real_size, self.real_size + self.synthetic_size))
        
        # Determine which list corresponds to larger/smaller dataset
        larger_indices = real_indices if self.is_real_larger else synthetic_indices
        smaller_indices = synthetic_indices if self.is_real_larger else real_indices
        
        # Shuffle indices
        random.shuffle(larger_indices)
        random.shuffle(smaller_indices)
        
        # Initialize counters
        larger_idx = 0
        smaller_idx = 0
        
        while True:
            # Reset indices if we've used all samples
            if larger_idx >= len(larger_indices):
                random.shuffle(larger_indices)
                larger_idx = 0
            if smaller_idx >= len(smaller_indices):
                random.shuffle(smaller_indices)
                smaller_idx = 0
                
            # Break if we can't form another complete batch
            if larger_idx + self.min_larger > len(larger_indices) or \
               smaller_idx + self.min_smaller > len(smaller_indices):
                break
            
            # Calculate remaining space in batch
            remaining_larger = self.batch_size - self.min_smaller
            remaining_smaller = self.batch_size - self.min_larger
            
            # Sample random number of indices from each dataset
            n_larger = random.randint(
                self.min_larger,
                min(remaining_larger, len(larger_indices) - larger_idx)
            )
            n_smaller = random.randint(
                self.min_smaller,
                min(remaining_smaller, len(smaller_indices) - smaller_idx)
            )
            
            # Adjust if total exceeds batch size
            if n_larger + n_smaller > self.batch_size:
                excess = (n_larger + n_smaller) - self.batch_size
                if random.random() < 0.5:
                    n_larger = max(self.min_larger, n_larger - excess)
                else:
                    n_smaller = max(self.min_smaller, n_smaller - excess)
            
            # Get indices for this batch
            batch_larger = larger_indices[larger_idx:larger_idx + n_larger]
            batch_smaller = smaller_indices[smaller_idx:smaller_idx + n_smaller]
            
            # Update counters
            larger_idx += n_larger
            smaller_idx += n_smaller
            
            # Combine and shuffle batch indices
            batch = batch_larger + batch_smaller
            random.shuffle(batch)
            
            yield batch
    
    def __len__(self) -> int:
        # Calculate number of complete batches we can make
        n_batches = min(
            self.larger_size // self.min_larger,
            self.smaller_size // self.min_smaller
        )
        return max(1, n_batches)  # Ensure at least one batch

class MixedDataset(Dataset):
    def __init__(self, real_features, real_labels, real_paths, 
                 synthetic_features, synthetic_labels, synthetic_paths):
        self.real_features = real_features
        self.real_labels = real_labels
        self.real_paths = real_paths
        self.synthetic_features = synthetic_features
        self.synthetic_labels = synthetic_labels
        self.synthetic_paths = synthetic_paths
        
        # Calculate sizes
        self.real_size = len(real_features)
        self.synthetic_size = len(synthetic_features)
        self.total_size = self.real_size + self.synthetic_size
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Determine if we should sample from real or synthetic
        if idx < self.real_size:
            features = self.real_features[idx]
            labels = self.real_labels[idx]
            path = self.real_paths[idx]
            is_real = torch.tensor(1)  # 1 for real
        else:
            synthetic_idx = idx - self.real_size
            features = self.synthetic_features[synthetic_idx]
            labels = self.synthetic_labels[synthetic_idx]
            path = self.synthetic_paths[synthetic_idx]
            is_real = torch.tensor(0)  # 0 for synthetic
            
        return features, labels, path, is_real

class CLIPAdapterOODTrainer:
    def __init__(self, config):
        self.model = CLIPAdapterOOD(
            reduction_factor=config.clip_adapter_ood['reduction_factor'],
            seed=config.clip_adapter_ood['seed'],
            device=config.device,
            gnn_hidden_dim=config.clip_adapter_ood['gnn_hidden_dim'],
            num_gnn_layers=config.clip_adapter_ood['num_gnn_layers'],
            num_classes=config.num_classes,
            train=True
        )
        self.config = config
        self.device = config.device
        self.seed = config.clip_adapter_ood['seed']
        self.feature_dir = config.feature_dir
        self.prompt_template = config.prompt_template
        self.temperature = config.clip_adapter_ood['temperature']
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.clip_adapter_ood['learning_rate'],
            weight_decay=0.01
        )
        
        self.clip, _ = clip.load(config.clip_model, device=config.device)
        self.clip.eval()  # Set to eval mode to freeze parameters

    def mmd_loss(self, x, y, kernel='rbf'):
        """
        Maximum Mean Discrepancy (MMD) loss between two sets of features
        Args:
            x: first set of features [batch_size, feature_dim]
            y: second set of features [batch_size, feature_dim]
            kernel: kernel type ('rbf' or 'linear')
        """
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())
        
        if kernel == 'rbf':
            bandwidth_range = [0.1, 5.0, 10.0]  # Multiple bandwidths for better coverage
            loss = 0
            for bandwidth in bandwidth_range:
                xx_rbf = torch.exp(-torch.square(xx).div(2 * bandwidth))
                yy_rbf = torch.exp(-torch.square(yy).div(2 * bandwidth))
                xy_rbf = torch.exp(-torch.square(xy).div(2 * bandwidth))
                
                loss += xx_rbf.mean() + yy_rbf.mean() - 2 * xy_rbf.mean()
            loss = loss / len(bandwidth_range)
        else:  # linear kernel
            loss = xx.mean() + yy.mean() - 2 * xy.mean()
            
        return loss
    
    def prepare_mixed_data(self, real_data: Dict, synthetic_data: Dict, batch_size: int, class_to_idx: Dict) -> DataLoader:
        """
        Prepare mixed dataloader with both real and synthetic data.
        Ensures each batch contains a mix of both types of data.
        The batch size is automatically calculated based on the amount of synthetic data available.
        """
        # Convert labels to indices if needed
        real_labels = real_data['labels']
        synthetic_labels = synthetic_data['labels']
        
        if isinstance(real_labels, list):
            real_label_indices = [class_to_idx[str(label) if isinstance(label, tuple) else label] for label in real_labels]
            real_labels = torch.tensor(real_label_indices)
        
        if isinstance(synthetic_labels, list):
            synthetic_label_indices = [class_to_idx[str(label) if isinstance(label, tuple) else label] for label in synthetic_labels]
            synthetic_labels = torch.tensor(synthetic_label_indices)
            
        # Create mixed dataset
        dataset = MixedDataset(
            real_features=real_data['features'],
            real_labels=real_labels,
            real_paths=real_data['paths'],
            synthetic_features=synthetic_data['features'],
            synthetic_labels=synthetic_labels,
            synthetic_paths=synthetic_data['paths']
        )
        
        # Create batch sampler that ensures mixed batches with auto batch size
        batch_sampler = MixedBatchSampler(
            real_size=len(real_data['features']),
            synthetic_size=len(synthetic_data['features']),
            max_batch_size=batch_size  # This is now used as an upper limit
        )
        
        # Create dataloader with custom batch sampler
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            pin_memory=True
        )
        
        return dataloader
    
    def set_seed(self):
        """Set seed for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def encode_text(self, text_prompts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode text prompts using CLIP"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.clip.encode_text(text_tokens).float()
            if normalize:
                text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def train_epoch(self, train_loader, text_features):
        """Train for one epoch."""
        self.model.train()
        total_contrastive_loss = 0
        total_ce_loss = 0
        total_mmd_loss = 0
        num_batches = 0
        
        for batch_features, batch_labels, _, is_real in tqdm(train_loader, desc="Training"):
            # Move batch to device and ensure float32
            batch_features = batch_features.to(self.device).float()
            batch_labels = batch_labels.to(self.device)
            is_real = is_real.to(self.device)
            
            # Normalize input features
            batch_features = F.normalize(batch_features, dim=-1)
            
            # Forward pass
            adapter_features, node_embeddings = self.model(batch_features)
            
            # Compute losses
            contrastive_loss, ce_loss, mmd = self.compute_losses(
                adapter_features, node_embeddings, text_features, batch_labels, is_real
            )
            
            # Total loss with weighting
            total_loss = (
                self.config.clip_adapter_ood['contrastive_weight'] * contrastive_loss +
                self.config.clip_adapter_ood['ce_weight'] * ce_loss +
                self.config.clip_adapter_ood['mmd_weight'] * mmd
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_contrastive_loss += contrastive_loss.item()
            total_ce_loss += ce_loss.item()
            total_mmd_loss += mmd.item()
            num_batches += 1
            
            # Debug info cada N batches
            if num_batches % 10 == 0:
                print(f"\nBatch {num_batches} stats:")
                print(f"Contrastive Loss: {contrastive_loss.item():.4f}")
                print(f"CE Loss: {ce_loss.item():.4f}")
                print(f"MMD Loss: {mmd.item():.4f}")
                print(f"Total Loss: {total_loss.item():.4f}")
        
        avg_losses = {
            'contrastive': total_contrastive_loss / num_batches,
            'ce': total_ce_loss / num_batches,
            'mmd': total_mmd_loss / num_batches
        }
        return avg_losses

    def validate(self, val_loader, text_features):
        """Validate the model."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_contrastive_loss = 0
        total_ce_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_labels, _, is_real in tqdm(val_loader, desc="Validating"):
                # Move batch to device and ensure float32
                batch_features = batch_features.to(self.device).float()
                batch_labels = batch_labels.to(self.device)
                is_real = is_real.to(self.device)
                
                # Normalize input features
                batch_features = F.normalize(batch_features, dim=-1)
                
                # Forward pass
                adapter_features, node_embeddings = self.model(batch_features)
                
                # Compute accuracy using adapter features
                similarity = torch.matmul(adapter_features, text_features.T)
                predictions = torch.argmax(similarity, dim=1)
                total_correct += (predictions == batch_labels).sum().item()
                total_samples += len(batch_labels)
                
                # Compute contrastive loss for monitoring
                adapter_logits = torch.matmul(adapter_features, text_features.T) / self.temperature
                adapter_logits = F.normalize(adapter_logits, dim=-1)
                positive_logits = adapter_logits[torch.arange(adapter_logits.size(0)), batch_labels]
                exp_logits = torch.exp(adapter_logits)
                sum_exp_logits = exp_logits.sum(dim=1)
                contrastive_loss = (-positive_logits + torch.log(sum_exp_logits)).mean()
                
                total_contrastive_loss += contrastive_loss.item()
                num_batches += 1
        
        accuracy = total_correct / total_samples
        avg_contrastive_loss = total_contrastive_loss / num_batches
        
        return accuracy, {'contrastive': avg_contrastive_loss}

    def save_checkpoint(self, val_acc):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        # Format accuracy for filename
        acc_str = f"{val_acc:.2f}".replace(".", "_")
        checkpoint_path = os.path.join(
            self.config.output_dir, 'clip_adapter_ood',
            f'adapter_ood_checkpoint_acc_{acc_str}.pt'
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['val_acc']

    def compute_losses(self, 
                      adapter_features: torch.Tensor,
                      node_embeddings: torch.Tensor,
                      text_features: torch.Tensor,
                      labels: torch.Tensor,
                      is_real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all three losses:
        1. Contrastive loss between adapter features and text embeddings
        2. Cross entropy loss for real node embeddings
        3. MMD loss between real and synthetic node embeddings
        """
        # 1. Contrastive Loss
        adapter_logits = torch.matmul(adapter_features, text_features.T) / self.temperature
        adapter_logits = F.normalize(adapter_logits, dim=-1)
        positive_logits = adapter_logits[torch.arange(adapter_logits.size(0)), labels]
        exp_logits = torch.exp(adapter_logits)
        sum_exp_logits = exp_logits.sum(dim=1)
        contrastive_loss = (-positive_logits + torch.log(sum_exp_logits)).mean()
        
        # 2. Cross Entropy Loss for real samples
        real_mask = is_real == 1
        if real_mask.any():
            real_node_embeddings = node_embeddings[real_mask]
            real_labels = labels[real_mask]
            ce_loss = F.cross_entropy(real_node_embeddings, real_labels)
        else:
            ce_loss = torch.tensor(0.0).to(self.device)
        
        # 3. MMD Loss between real and synthetic node embeddings
        real_embeddings = node_embeddings[real_mask]
        synthetic_embeddings = node_embeddings[~real_mask]
        
        if len(real_embeddings) > 0 and len(synthetic_embeddings) > 0:
            mmd = self.mmd_loss(real_embeddings, synthetic_embeddings)
        else:
            mmd = torch.tensor(0.0).to(self.device)
            
        return contrastive_loss, ce_loss, mmd

    def train(self, classes_names):
        """Main training loop with early stopping."""
        self.set_seed()

        # Prepare class mapping
        unique_classes = sorted({class_name for _, class_name in classes_names})
        class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}

        # Load features
        real_feature_file = os.path.join(self.feature_dir, "real_data.pt")
        synthetic_feature_file = os.path.join(self.feature_dir, "synthetic_features.pt")
        
        if not os.path.exists(real_feature_file) or not os.path.exists(synthetic_feature_file):
            raise FileNotFoundError("Feature files not found")
        
        real_data = torch.load(real_feature_file)
        synthetic_data = torch.load(synthetic_feature_file)

        # Get train and val splits
        train_loader = self.prepare_mixed_data(
            real_data['train'],
            synthetic_data,
            self.config.clip_adapter_ood['batch_size'],
            class_to_idx
        )
        
        val_loader = self.prepare_mixed_data(
            real_data['val'],
            synthetic_data,  # Using same synthetic data for validation
            self.config.clip_adapter_ood['batch_size'],
            class_to_idx
        )

        print("\nStarting training...")
        print(f"Real training samples: {len(real_data['train']['features'])}")
        print(f"Synthetic training samples: {len(synthetic_data['features'])}")
        print(f"Validation samples: {len(real_data['val']['features'])}")

        # Prepare text features
        templates = [self.prompt_template] if isinstance(self.prompt_template, str) else self.prompt_template
        all_text_features = []
        
        with torch.no_grad():
            for template in templates:
                prompts = [template.format(class_name) for class_name in unique_classes]
                text_features = []
                
                for i in range(0, len(prompts), self.config.clip_adapter_ood['batch_size']):
                    batch_prompts = prompts[i:i + self.config.clip_adapter_ood['batch_size']]
                    batch_features = self.encode_text(batch_prompts)
                    text_features.append(batch_features)
                
                template_features = torch.cat(text_features, dim=0)
                template_features = F.normalize(template_features, dim=-1)
                all_text_features.append(template_features)
        
        if len(all_text_features) > 1:
            text_features = torch.stack(all_text_features).mean(dim=0)
            text_features = F.normalize(text_features, dim=-1)
        else:
            text_features = all_text_features[0]

        # Training loop
        best_val_acc = 0
        patience_counter = 0
        best_checkpoint_path = None
        
        for epoch in range(self.config.clip_adapter_ood['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config.clip_adapter_ood['num_epochs']}")
            
            # Train
            train_losses = self.train_epoch(train_loader, text_features)
            print(f"Training losses:")
            print(f"  Contrastive: {train_losses['contrastive']:.4f}")
            print(f"  CE: {train_losses['ce']:.4f}")
            print(f"  MMD: {train_losses['mmd']:.4f}")
            
            # Validate
            val_acc, val_losses = self.validate(val_loader, text_features)
            print(f"Validation accuracy: {val_acc:.4f}")
            print(f"Validation losses:")
            print(f"  Contrastive: {val_losses['contrastive']:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                patience_counter = 0
                best_checkpoint_path = self.save_checkpoint(val_acc)
            else:
                patience_counter += 1
                print(f"Validation accuracy did not improve. Patience: {patience_counter}/{self.config.clip_adapter_ood['patience']}")
            
            if patience_counter >= self.config.clip_adapter_ood['patience']:
                print("\nEarly stopping triggered!")
                break
        
        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
        print(f"Best model saved at: {best_checkpoint_path}")
        return best_val_acc, best_checkpoint_path 