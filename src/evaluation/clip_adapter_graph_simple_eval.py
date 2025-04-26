import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

class ClipAdapterGraphSimpleEvaluator:
    def __init__(self, model, classes, ood_test, config, checkpoint_path):
        self.model = model
        self.classes = classes
        self.ood_test = ood_test
        self.config = config
        self.device = config.device
        self.checkpoint_path = checkpoint_path
        self.k_neighbors = 10  # Mismo número de vecinos que en el trainer
        
        if self.ood_test:
            self.feature_dir = config.feature_dir_ood
        else:
            self.feature_dir = config.feature_dir

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

    def evaluate(self):
        """Evaluate the model on test data."""
        self.model.eval()
        
        # Load features
        real_feature_file = os.path.join(self.feature_dir, "real_data.pt")
        if not os.path.exists(real_feature_file):
            raise FileNotFoundError("Feature file not found")
        
        real_data = torch.load(real_feature_file)
        
        # Prepare class mapping
        unique_classes = sorted({class_name for _, class_name in self.classes})
        class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
        
        # Prepare test data
        test_data = self.prepare_data(
            real_data['test']['features'],
            real_data['test']['labels'],
            class_to_idx
        )
        
        print("\nStarting evaluation...")
        print(f"Test samples: {len(test_data.x)}")
        
        with torch.no_grad():
            # Forward pass
            output = self.model(test_data.x, test_data.edge_index)
            
            # Calculate loss and accuracy
            loss = F.cross_entropy(output, test_data.y)
            pred = output.argmax(dim=1)
            acc = (pred == test_data.y).float().mean()
            
            # Calculate per-class accuracy
            per_class_acc = {}
            for class_name, idx in class_to_idx.items():
                mask = test_data.y == idx
                if mask.sum() > 0:
                    class_acc = (pred[mask] == test_data.y[mask]).float().mean()
                    per_class_acc[class_name] = class_acc.item()
        
        results = {
            'test_loss': loss.item(),
            'test_acc': acc.item(),
            'per_class_acc': per_class_acc
        }
        
        print("\nTest Results:")
        print(f"Loss: {results['test_loss']:.4f}")
        print(f"Accuracy: {results['test_acc']:.4f}")
        print("\nPer-class Accuracy:")
        for class_name, class_acc in results['per_class_acc'].items():
            print(f"{class_name}: {class_acc:.4f}")
        
        return results 