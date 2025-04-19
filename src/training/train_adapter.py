import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as utils
from tqdm import tqdm
import logging
import json
from pathlib import Path
import numpy as np
from dataclasses import asdict
from src.models.clip_adapter import create_clip_adapter
from src.evaluation.clip_adapter_evaluator import ClipAdapterEvaluator
from src.utils.hyperparameter_search import sample_hyperparameters, setup_random_seeds

def random_search_adapter(config, classes, prompt_template="a photo of a {}"):
    """Perform random search for hyperparameter optimization
    
    Args:
        config: Configuration object containing training parameters
        classes: List of tuples (folder_name, class_name) from class mapping
        prompt_template: Template string for text prompts (default: "a photo of a {}")
        
    Returns:
        tuple: (best_config, best_val_acc, best_model_path)
    """
    # Create directory for search results
    search_dir = Path(config.get_model_path("")).parent / "search_results"
    search_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename=search_dir / "search.log",
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    best_val_acc = 0
    best_config = None
    best_model_path = None
    all_results = []
    
    # Get search space from config
    search_space = config.get_search_space()
    
    for trial in range(config.n_trials):
        # Sample hyperparameters
        trial_seed = config.seed * 100 + trial if config.seed is not None else None
        setup_random_seeds(trial_seed)
        hp = sample_hyperparameters(search_space, seed=trial_seed)
        
        # Update config with sampled hyperparameters
        for key, value in hp.items():
            setattr(config, key, value)
        
        logging.info(f"\nTrial {trial + 1}/{config.n_trials}")
        logging.info(f"Parameters: {hp}")
        
        # Train with current hyperparameters
        metrics, val_acc = train_adapter(config, classes, prompt_template)
        
        # Save trial results
        trial_results = {
            'trial': trial,
            'hyperparameters': hp,
            'val_acc': val_acc,
            'metrics': metrics,
            'prompt_template': prompt_template
        }
        all_results.append(trial_results)
        
        # Update best if necessary
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = hp.copy()
            # Store path to the best model
            best_model_path = config.get_model_path("terra")
            logging.info(f"New best validation accuracy: {best_val_acc:.2f}%")
            logging.info(f"Best model saved at: {best_model_path}")
        
        # Save all results after each trial
        with open(search_dir / "search_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
    
    return best_config, best_val_acc, best_model_path

def train_adapter_with_config(config,dataset_name, classes, prompt_template="a photo of a {}"):
    """Train the CLIP adapter and return validation accuracy and model path.
    This function is designed to work with the generic random search.
    
    Args:
        config: Dictionary with hyperparameters or Config object
        dataset_name: Name of the dataset
        classes: List of tuples (folder_name, class_name)
        prompt_template: Template for text prompts
        
    Returns:
        tuple: (validation_accuracy, model_path)
    """
    # Train the model
    metrics, val_acc = train_adapter(config, dataset_name, classes, prompt_template)
    
    # Return the metric to optimize and the model path
    return val_acc, config.get_model_path(dataset_name)

class FeatureDataset(Dataset):
    def __init__(self, feature_path: str, split: str = 'train'):
        """
        Args:
            feature_path: Path to the feature file
            split: Which split to use ('train', 'val', 'test')
        """
        # Load pre-extracted features
        data = torch.load(feature_path)
        
        if isinstance(data, dict) and 'features' in data:
            # Single split format
            self.features = data['features']
            self.labels = data['labels']
        else:
            # Multiple splits format
            split_key = [k for k in data.keys() if split in k.lower()][0]
            split_data = data[split_key]
            self.features = split_data['features']
            self.labels = split_data['labels']
            
        # Convert string labels to indices for training
        unique_labels = sorted(list(set(self.labels)))
        self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
        self.label_indices = torch.tensor([self.class_to_idx[label] for label in self.labels])
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.label_indices[idx],
            'class_name': self.labels[idx]
        }
    
    @property
    def num_classes(self):
        return len(self.class_to_idx)

def train_adapter(config, dataset_name, classes, prompt_template="a photo of a {}"):
    """Train the CLIP adapter using pre-extracted features.
    
    Args:
        config: Configuration object containing training parameters
        classes: List of tuples (folder_name, class_name) from class mapping
        prompt_template: Template string for text prompts (default: "a photo of a {}")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_dir = Path(config.get_model_path(dataset_name)).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename=output_dir / "training.log",
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Create datasets and dataloaders
    train_dataset = FeatureDataset(os.path.join(config.feature_dir, "real_data.pt"), split='train')
    val_dataset = FeatureDataset(os.path.join(config.feature_dir, "real_data.pt"), split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model, criterion, optimizer and scheduler
    model, criterion = create_clip_adapter(
        device=device,
        reduction_factor=config.reduction_factor
    )
    
    # Update criterion temperature
    criterion.temperature = config.temperature
    
    # Pre-compute text embeddings for all classes once using provided prompt template
    print("Pre-computing text embeddings...")
    with torch.no_grad():
        # First collect all unique classes
        initial_classes = {class_name for _, class_name in classes}
        train_classes = set(train_dataset.labels)
        all_classes = list(initial_classes.union(train_classes))
        
        # Generate text prompts for all classes
        templates = [prompt_template] if isinstance(prompt_template, str) else prompt_template
        all_text_features = []
        
        for template in templates:
            prompts = [template.format(class_name) for class_name in all_classes]
            text_features = []
            batch_size = 32
            
            print(f"\nGenerating text embeddings for template: '{template}'")
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_features = model.encode_text(batch_prompts)
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
        
        # Move to device
        text_features = text_features.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        [
            {"params": model.down_proj.parameters()},
            {"params": model.up_proj.parameters()},
            {"params": model.layer_norm.parameters()}
        ],
        lr=config.learning_rate
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.learning_rate * 0.01
    )
    
    # Training state
    best_val_acc = 0.0
    patience_counter = 0
    train_metrics = []
    
    # Create evaluator for validation
    evaluator = ClipAdapterEvaluator(device)
    
    def evaluate(split_loader, split="val"):
        model.eval()
        total_loss = 0
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in split_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                # Get adapted features
                adapted_features = model.adapt_features(features)
                
                # Compute loss using pre-computed text features
                loss = criterion(adapted_features, text_features, labels)
                total_loss += loss.item()
                
                # Store for accuracy computation
                all_features.append(adapted_features.cpu())
                all_labels.extend(batch['class_name'])
        
        # Compute accuracy using provided classes
        all_features = torch.cat(all_features, dim=0)
        results = evaluator.evaluate_features(
            features=all_features,
            labels=all_labels,
            class_names=classes,
            prompt_template=prompt_template
        )
        
        avg_loss = total_loss / len(split_loader)
        return avg_loss, results['accuracy']
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch in progress_bar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # Get adapted features
            adapted_features = model.adapt_features(features)
            
            # Compute loss using pre-computed text features
            loss = criterion(adapted_features, text_features, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute training metrics
        train_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss, val_acc = evaluate(val_loader, "val")
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        }
        
        train_metrics.append(metrics)
        logging.info(f"Epoch {epoch+1}: {metrics}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config,
                'metrics': train_metrics,
                'text_features': text_features,  # Guardamos los embeddings pre-computados
                'classes': classes,  # Guardamos las clases y su mapeo
                'prompt_template': prompt_template  # Guardamos el template usado
            }
            
            torch.save(checkpoint, config.get_model_path("terra"))
            logging.info(f"Saved best model with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return train_metrics, best_val_acc 