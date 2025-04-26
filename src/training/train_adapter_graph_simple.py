import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn import knn_graph
import torch.nn.functional as F
from src.models.clip_adapter_graph_simple import CLIPAdapterGraphSimple

class CLIPAdapterGraphSimpleTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.feature_dir = config.feature_dir
        self.prompt_template = config.prompt_template
        self.k_neighbors = 10  # Número de vecinos para KNN
        
        # Initialize model
        self.model = CLIPAdapterGraphSimple(
            reduction_factor=config.clip_adapter_graph['reduction_factor'],
            device=self.device,
            gnn_hidden_dim=config.clip_adapter_graph['gnn_hidden_dim'],
            num_gnn_layers=config.clip_adapter_graph['num_gnn_layers'],
            seed=config.clip_adapter_graph['seed']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.clip_adapter_graph['learning_rate'],
            weight_decay=config.clip_adapter_graph['weight_decay']
        )
        
        # Training parameters
        self.num_epochs = config.clip_adapter_graph['num_epochs']
        self.batch_size = config.clip_adapter_graph['batch_size']
        self.patience = config.clip_adapter_graph['patience']
        
        # Create output directory
        self.output_dir = os.path.join(config.output_dir, 'clip_adapter_graph_simple')
        os.makedirs(self.output_dir, exist_ok=True)

    def create_adjacency_matrix(self, features):
        """Create KNN graph from features."""
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)
        
        # Create KNN graph
        edge_index = knn_graph(features, k=self.k_neighbors, batch=None)
        
        # Calculate edge weights using cosine similarity
        row, col = edge_index
        edge_weight = F.cosine_similarity(features[row], features[col], dim=1)
        
        return edge_index, edge_weight

    def prepare_data(self, features, labels, class_to_idx):
        # Convert labels to indices
        label_indices = torch.tensor([class_to_idx[label] for label in labels])
        
        # Create edge index and weights for the graph
        features_tensor = torch.tensor(features, dtype=torch.float32)
        edge_index, edge_weight = self.create_adjacency_matrix(features_tensor)
        
        # Create PyG Data object
        data = Data(
            x=features_tensor,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=label_indices
        )
        
        return data.to(self.device)

    def train_epoch(self, train_data):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(train_data.x, train_data.edge_index)
        loss = self.criterion(output, train_data.y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == train_data.y).float().mean()
        
        return loss.item(), acc.item()

    def validate(self, val_data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(val_data.x, val_data.edge_index)
            loss = self.criterion(output, val_data.y)
            
            pred = output.argmax(dim=1)
            acc = (pred == val_data.y).float().mean()
        
        return loss.item(), acc.item()

    def train(self, classes_names):
        """Main training loop with early stopping."""
        # Prepare class mapping
        unique_classes = sorted({class_name for _, class_name in classes_names})
        class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
        
        # Load features
        real_feature_file = os.path.join(self.feature_dir, "real_data.pt")
        if not os.path.exists(real_feature_file):
            raise FileNotFoundError("Feature file not found")
        
        real_data = torch.load(real_feature_file)
        
        # Prepare training and validation data
        train_data = self.prepare_data(
            real_data['train']['features'],
            real_data['train']['labels'],
            class_to_idx
        )
        
        val_data = self.prepare_data(
            real_data['val']['features'],
            real_data['val']['labels'],
            class_to_idx
        )
        
        print("\nStarting training...")
        print(f"Training samples: {len(train_data.x)}")
        print(f"Validation samples: {len(val_data.x)}")
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        checkpoint_path = os.path.join(self.output_dir, 'best_model.pt')
        
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(train_data)
            val_loss, val_acc = self.validate(val_data)
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        return best_val_acc, checkpoint_path 