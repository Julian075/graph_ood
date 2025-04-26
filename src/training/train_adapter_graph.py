import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import clip
from typing import List
from src.models.clip_adapter_graph import CLIPAdapterGraph
import numpy as np
import wandb

class CLIPAdapterGraphTrainer:
    def __init__(self, config):
        self.model = CLIPAdapterGraph(
            reduction_factor=config.clip_adapter_graph['reduction_factor'],
            seed=config.clip_adapter_graph['seed'],
            device=config.device,
            gnn_hidden_dim=config.clip_adapter_graph['gnn_hidden_dim'],
            num_gnn_layers=config.clip_adapter_graph['num_gnn_layers']
        )
        self.config = config
        self.device = config.device
        self.seed = config.clip_adapter_graph['seed']
        self.feature_dir = config.feature_dir
        self.prompt_template = config.prompt_template
        self.use_synthetic_data = config.use_synthetic_data
        
        # Get Weights & Biases token from environment
        wandb_token = os.getenv('WANDB_TOKEN')
        if not wandb_token:
            raise ValueError("WANDB_TOKEN not found in environment variables. Please set it with 'export WANDB_TOKEN=...'")
            
        # Initialize wandb with token
        wandb.login(key=wandb_token)
        wandb.init(
            project="clip-adapter-graph-training",
            config={
                "architecture": "CLIPAdapterGraph",
                "learning_rate": config.clip_adapter_graph['learning_rate'],
                "epochs": config.clip_adapter_graph['num_epochs'],
                "batch_size": config.clip_adapter_graph['batch_size'],
                "temperature": config.clip_adapter_graph['temperature'],
                "reduction_factor": config.clip_adapter_graph['reduction_factor'],
                "gnn_hidden_dim": config.clip_adapter_graph['gnn_hidden_dim'],
                "num_gnn_layers": config.clip_adapter_graph['num_gnn_layers'],
                "seed": config.clip_adapter_graph['seed'],
                "use_synthetic": config.use_synthetic_data
            }
        )
        
        # Initialize SGD optimizer with momentum
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.clip_adapter_graph['learning_rate'],
            momentum=0.9,
            weight_decay=0.01
        )
        
        self.temperature = config.clip_adapter_graph['temperature']
        self.clip, _ = clip.load(config.clip_model, device=config.device)
        self.clip.eval()  # Set to eval mode to freeze parameters
    
    def encode_text(self, text_prompts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode text prompts using CLIP with explicit precision handling"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.clip.encode_text(text_tokens).float()  # Ensure float32
            if normalize:
                text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def compute_contrastive_loss(self, image_features, gnn_features, text_features, labels):
        """
        Compute contrastive loss for both adapter and GNN features
        
        Args:
            image_features: Adapted features [batch_size, clip_dim]
            gnn_features: GNN processed features [batch_size, clip_dim]
            text_features: Text features [num_classes, clip_dim]
            labels: Class indices [batch_size]
        """
        # Compute adapter loss
        adapter_logits = torch.matmul(image_features, text_features.T) / self.temperature
        adapter_loss = self.compute_single_contrastive_loss(adapter_logits, labels)
        print(f"Adapter loss: {adapter_loss.item():.4f}")
        
        # Compute GNN loss
        gnn_logits = torch.matmul(gnn_features, text_features.T) / self.temperature
        gnn_loss = self.compute_single_contrastive_loss(gnn_logits, labels)
        print(f"GNN loss: {gnn_loss.item():.4f}")
        # Total loss is the sum of both losses
        total_loss = adapter_loss + gnn_loss
        return total_loss
    
    def compute_single_contrastive_loss(self, logits, labels):
        """Helper function to compute contrastive loss for a single set of logits"""
        logits = F.normalize(logits, dim=-1)
        positive_logits = logits[torch.arange(logits.size(0)), labels]
        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=1)
        loss = -positive_logits + torch.log(sum_exp_logits)
        return loss.mean()
    
    def prepare_data(self, data, batch_size, class_to_idx):
        """Prepare data loaders for training/validation."""
        labels = data['labels']
        if isinstance(labels, list):
            label_indices = [class_to_idx[str(label) if isinstance(label, tuple) else label] for label in labels]
            labels = torch.tensor(label_indices)
        elif isinstance(labels, torch.Tensor) and labels.dtype != torch.int64:
            label_indices = [class_to_idx[str(label.item())] for label in labels]
            labels = torch.tensor(label_indices)
            
        dataset = TensorDataset(
            data['features'],
            labels,
            torch.arange(len(data['paths']))
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        return dataloader, data['paths']
    
    def train_epoch(self, train_loader, text_features):
        """Train for one epoch with explicit precision handling."""
        self.model.train()
        total_adapter_loss = 0
        total_gnn_loss = 0
        total_combined_loss = 0
        num_batches = 0
        
        for batch_features, batch_labels, _ in tqdm(train_loader, desc="Training"):
            # Move batch to device and ensure float32
            batch_features = batch_features.to(self.device).float()
            batch_labels = batch_labels.to(self.device)
            
            # Normalize input features
            batch_features = F.normalize(batch_features, dim=-1)
            
            # Forward pass with explicit precision
            adapted_features, gnn_features = self.model(batch_features, training=True)
            
            # Normalize features
            adapted_features = F.normalize(adapted_features, dim=-1)
            gnn_features = F.normalize(gnn_features, dim=-1)
            
            # Compute individual losses
            adapter_logits = torch.matmul(adapted_features, text_features.T) / self.temperature
            gnn_logits = torch.matmul(gnn_features, text_features.T) / self.temperature
            
            adapter_loss = self.compute_single_contrastive_loss(adapter_logits, batch_labels)
            gnn_loss = self.compute_single_contrastive_loss(gnn_logits, batch_labels)
            
            # Total loss is the sum of both losses
            total_loss = adapter_loss + gnn_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_adapter_loss += adapter_loss.item()
            total_gnn_loss += gnn_loss.item()
            total_combined_loss += total_loss.item()
            num_batches += 1
        
        avg_losses = {
            'adapter': total_adapter_loss / num_batches,
            'gnn': total_gnn_loss / num_batches,
            'total': total_combined_loss / num_batches
        }
        return avg_losses
    
    def validate(self, val_loader, text_features):
        """Validate the model with explicit precision handling."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels, _ in tqdm(val_loader, desc="Validating"):
                # Move batch to device and ensure float32
                batch_features = batch_features.to(self.device).float()
                batch_labels = batch_labels.to(self.device)
                
                # Normalize input features
                batch_features = F.normalize(batch_features, dim=-1)
                
                # Forward pass
                adapted_features, _ = self.model(batch_features, training=False)
                adapted_features = F.normalize(adapted_features, dim=-1)
                
                similarity = torch.matmul(adapted_features, text_features.T)
                similarity = F.normalize(similarity, dim=-1)
                
                predictions = torch.argmax(similarity, dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        accuracy = correct / total
        return accuracy
    
    def save_checkpoint(self, val_acc):
        """Save model checkpoint with accuracy in filename."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        # Format accuracy for filename
        acc_str = f"{val_acc:.2f}".replace(".", "_")
        
        if self.use_synthetic_data:
            checkpoint_path = os.path.join(
                self.config.output_dir, 'clip_adapter_graph',
                f'adapter_graph_checkpoint_synthetic_acc_{acc_str}.pt'
            )
        else:
            checkpoint_path = os.path.join(
                self.config.output_dir, 'clip_adapter_graph',
                f'adapter_graph_checkpoint_acc_{acc_str}.pt'
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path  # Return the path for reference
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['val_acc']
    
    def set_seed(self):
        """Set seed for reproducibility with additional settings."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def train(self, classes_names):
        """
        Main training loop with early stopping.
        
        Returns:
            tuple: (best_val_acc, checkpoint_path)
        """
        self.set_seed()

        unique_classes = sorted({class_name for _, class_name in classes_names})
        class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}

        # Load features
        feature_file = os.path.join(self.feature_dir, "real_data.pt")
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found at {feature_file}")
        
        all_data = torch.load(feature_file)
        if self.use_synthetic_data:
            feature_file_synthetic = os.path.join(self.feature_dir, "synthetic_features.pt")
            if not os.path.exists(feature_file_synthetic):
                raise FileNotFoundError(f"Feature file not found at {feature_file_synthetic}")
            
            all_data_synthetic = torch.load(feature_file_synthetic)
            all_data['train']['features'] = torch.cat([all_data['train']['features'], all_data_synthetic['features']])
            
            if isinstance(all_data['train']['labels'], list):
                train_label_indices = [class_to_idx[str(label) if isinstance(label, tuple) else label] for label in all_data['train']['labels']]
                all_data['train']['labels'] = torch.tensor(train_label_indices)
            
            if isinstance(all_data_synthetic['labels'], list):
                synthetic_label_indices = [class_to_idx[str(label) if isinstance(label, tuple) else label] for label in all_data_synthetic['labels']]
                all_data_synthetic['labels'] = torch.tensor(synthetic_label_indices)
            
            all_data['train']['labels'] = torch.cat([all_data['train']['labels'], all_data_synthetic['labels']])
            all_data['train']['paths'] = all_data['train']['paths'] + all_data_synthetic['paths']

        # Get train and val splits
        train_features = all_data['train']['features']
        train_labels = all_data['train']['labels']
        val_features = all_data['val']['features']
        val_labels = all_data['val']['labels']

        print("\nStarting training...")
        print(f"Training samples: {len(train_features)}")
        print(f"Validation samples: {len(val_features)}")
        
        # Prepare data loaders
        train_loader, _ = self.prepare_data(
            {'features': train_features, 'labels': train_labels, 'paths': all_data['train']['paths']},
            self.config.clip_adapter_graph['batch_size'],
            class_to_idx
        )
        val_loader, _ = self.prepare_data(
            {'features': val_features, 'labels': val_labels, 'paths': all_data['val']['paths']},
            self.config.clip_adapter_graph['batch_size'],
            class_to_idx
        )

        # Prepare text features
        templates = [self.prompt_template] if isinstance(self.prompt_template, str) else self.prompt_template
        all_text_features = []
        
        with torch.no_grad():
            for template in templates:
                prompts = [template.format(class_name) for class_name in unique_classes]
                text_features = []
                
                for i in range(0, len(prompts), self.config.clip_adapter_graph['batch_size']):
                    batch_prompts = prompts[i:i + self.config.clip_adapter_graph['batch_size']]
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

        # Ensure text features are float32
        text_features = text_features.float()
        #print(f"\nText features dtype: {text_features.dtype}")
        #print(f"Text features shape: {text_features.shape}")
        #print(f"Text features stats - min: {text_features.min().item():.4f}, max: {text_features.max().item():.4f}")

        # Training loop
        best_val_acc = 0
        patience_counter = 0
        best_checkpoint_path = None
        
        for epoch in range(self.config.clip_adapter_graph['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config.clip_adapter_graph['num_epochs']}")
            
            train_losses = self.train_epoch(train_loader, text_features)
            print(f"Training losses:")
            print(f"  Adapter: {train_losses['adapter']:.4f}")
            print(f"  GNN: {train_losses['gnn']:.4f}")
            print(f"  Total: {train_losses['total']:.4f}")
            
            val_acc = self.validate(val_loader, text_features)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train/losses/adapter": train_losses['adapter'],
                "train/losses/gnn": train_losses['gnn'],
                "train/losses/total": train_losses['total'],
                "val/accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            # Create plots for wandb
            # 1. Training Losses Plot
            wandb.log({
                "plots/training_losses": wandb.plot.line_series(
                    xs=[epoch],
                    ys=[[train_losses['adapter']], 
                        [train_losses['gnn']], 
                        [train_losses['total']]],
                    keys=['Adapter', 'GNN', 'Total'],
                    title='Training Losses',
                    xname='Epoch'
                )
            })
            
            # 2. Loss Proportions Plot
            wandb.log({
                "plots/loss_proportions": wandb.plot.line_series(
                    xs=[epoch],
                    ys=[[train_losses['adapter'] / train_losses['total']], 
                        [train_losses['gnn'] / train_losses['total']]],
                    keys=['Adapter Proportion', 'GNN Proportion'],
                    title='Loss Proportions',
                    xname='Epoch'
                )
            })
            
            # 3. Training vs Validation Plot
            wandb.log({
                "plots/train_val_metrics": wandb.plot.line_series(
                    xs=[epoch],
                    ys=[[train_losses['total']], [val_acc]],
                    keys=['Training Loss', 'Validation Accuracy'],
                    title='Training vs Validation',
                    xname='Epoch'
                )
            })
            
            # 4. Loss Composition Bar Plot
            wandb.log({
                "plots/loss_composition": wandb.plot.bar(
                    wandb.Table(
                        columns=["Loss Type", "Value"],
                        data=[
                            ["Adapter", train_losses['adapter']],
                            ["GNN", train_losses['gnn']]
                        ]
                    ),
                    "Loss Type",
                    "Value",
                    title="Loss Components"
                )
            })
            
            if val_acc > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                patience_counter = 0
                best_checkpoint_path = self.save_checkpoint(val_acc)
                
                # Log best model to wandb
                wandb.save(best_checkpoint_path)
            else:
                patience_counter += 1
                print(f"Validation accuracy did not improve. Patience: {patience_counter}/{self.config.clip_adapter_graph['patience']}")
            
            if patience_counter >= self.config.clip_adapter_graph['patience']:
                print("\nEarly stopping triggered!")
                break
        
        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
        print(f"Best model saved at: {best_checkpoint_path}")
        
        # Close wandb run
        wandb.finish()
        
        return best_val_acc, best_checkpoint_path 