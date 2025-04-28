import os
import torch
import torch.nn.functional as F
from src.models.clip_adapter_graph_simple import CLIPAdapterGraphSimple
import clip
import wandb
import tqdm
from .utils import (
    encode_text,
    compute_contrastive_loss,
    set_seed,
    save_checkpoint,
    load_data,
    prepare_dataloaders
)

class CLIPAdapterGraphSimpleTrainer:
    def __init__(self, config):
        self.model = CLIPAdapterGraphSimple(config.clip_adapter['reduction_factor'],config.clip_adapter['seed'], config.device)
        self.config = config
        self.device = config.device
        self.seed = config.clip_adapter['seed']
        self.feature_dir = config.feature_dir
        self.prompt_template = config.prompt_template
        self.use_synthetic_data = config.use_synthetic_data
        self.name_model = 'clip_adapter_graph_simple'
        
        # Get Weights & Biases token from environment
        wandb_token = os.getenv('WANDB_TOKEN')
        if not wandb_token:
            raise ValueError("WANDB_TOKEN not found in environment variables. Please set it with 'export WANDB_TOKEN=...'")
            
        # Initialize wandb with token
        wandb.login(key=wandb_token)
        wandb.init(
            project="clip-adapter-graph-simple-training",
            config={
                "architecture": "CLIPAdapterGraphSimple",
                "learning_rate": config.clip_adapter['learning_rate'],
                "epochs": config.clip_adapter['num_epochs'],
                "batch_size": config.clip_adapter['batch_size'],
                "temperature": config.clip_adapter['temperature'],
                "reduction_factor": config.clip_adapter['reduction_factor'],
                "seed": config.clip_adapter['seed'],
                "use_synthetic": config.use_synthetic_data
            }
        )
        
        # Initialize SGD optimizer with momentum
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.clip_adapter['learning_rate'],
            momentum=0.9,
            weight_decay=0.01
        )
        
        self.temperature = config.clip_adapter['temperature']
        self.clip, _ = clip.load(config.clip_model, device=config.device)
        self.clip.eval()  # Set to eval mode to freeze parameters

    def process_batch(self, features, labels, training=True):
        """Common processing for training and validation batches"""
        features = features.to(self.device).float()
        labels = labels.to(self.device)
        features = F.normalize(features, dim=-1)
        output, output_gnn = self.model(features,training)
        output = F.normalize(output, dim=-1)
        output_gnn = F.normalize(output_gnn, dim=-1)
        return output, output_gnn, labels

    def train_epoch(self, train_loader, text_features):
        """Train for one epoch with explicit precision handling."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_features, batch_labels, _ in tqdm(train_loader, desc="Training"):
            output, output_gnn, batch_labels = self.process_batch(batch_features, batch_labels, training=True)
            loss = compute_contrastive_loss(output, text_features, batch_labels, self.temperature)
            loss_gnn=F.cross_entropy(output_gnn, batch_labels)
            loss_total = loss + loss_gnn
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            
            total_loss += loss_total.item()
            num_batches += 1
            
            # Log batch metrics
            wandb.log({
                "batch_loss": loss.item(),
                "batch_loss_gnn": loss_gnn.item()
            })
        
        return total_loss / num_batches if num_batches > 0 else float('nan')
    
    def validate(self, val_loader, text_features):
        """Validate the model with explicit precision handling."""
        self.model.eval()
        correct = total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels, _ in tqdm(val_loader, desc="Validating"):
                output, batch_labels = self.process_batch(batch_features, batch_labels, training=False)
                similarity = torch.matmul(output, text_features.T)
                predictions = torch.argmax(similarity, dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        return correct / total
    
    def train(self, classes_names):
        """Main training loop with early stopping."""
        set_seed(self.seed)
        unique_classes = sorted({class_name for _, class_name in classes_names})
        class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}

        # Load and prepare data
        all_data = load_data(self.feature_dir, self.use_synthetic_data, class_to_idx)
        train_loader, val_loader = prepare_dataloaders(
            all_data, 
            self.config.clip_adapter['batch_size'],
            class_to_idx
        )

        print(f"\nStarting training...\nTraining samples: {len(all_data['train']['features'])}\nValidation samples: {len(all_data['val']['features'])}")
        
        # Prepare text features
        templates = [self.prompt_template] if isinstance(self.prompt_template, str) else self.prompt_template
        all_text_features = []
        with torch.no_grad():
            for template in templates:
                prompts = [template.format(class_name) for class_name in unique_classes]
                text_features = []
                
                for i in range(0, len(prompts), self.config.clip_adapter['batch_size']):
                    batch_prompts = prompts[i:i + self.config.clip_adapter['batch_size']]
                    batch_features = encode_text(self.clip, self.device, batch_prompts)
                    text_features.append(batch_features)
                
                # Concatenate and normalize features for this template
                template_features = torch.cat(text_features, dim=0)
                template_features = F.normalize(template_features, dim=-1)
                all_text_features.append(template_features)
        
        # Average text features across templates if using multiple
        if len(all_text_features) > 1:
            text_features = torch.stack(all_text_features).mean(dim=0)
            text_features = F.normalize(text_features, dim=-1)
        else:
            text_features = all_text_features[0]

        # Ensure text features are float32
        text_features = text_features.float()
        
        # Training loop
        best_val_acc = patience_counter = 0
        best_checkpoint_path = None
        
        for epoch in range(self.config.clip_adapter['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config.clip_adapter['num_epochs']}")
            
            train_loss = self.train_epoch(train_loader, text_features)
            val_acc = self.validate(val_loader, text_features)
            print(f"Training loss: {train_loss:.4f}\nValidation accuracy: {val_acc:.4f}")
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            if val_acc > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                patience_counter = 0
                best_checkpoint_path = save_checkpoint(
                    self.model,
                    self.optimizer,
                    val_acc,
                    self.config,
                    self.config.output_dir,
                    self.name_model,
                    self.use_synthetic_data
                )
                wandb.save(best_checkpoint_path)
            else:
                patience_counter += 1
                print(f"Validation accuracy did not improve. Patience: {patience_counter}/{self.config.patience}")
                if patience_counter >= self.config.patience:
                    print("\nEarly stopping triggered!")
                    break
        
        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}\nBest model saved at: {best_checkpoint_path}")
        wandb.finish()
        return best_val_acc, best_checkpoint_path
