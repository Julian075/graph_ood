import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from src.models.clip_adapter import create_clip_adapter
from src.generation.synthetic_data import get_classes_from_folder

class ClipAdapterEvaluator:
    def __init__(self, config_or_device):
        """Initialize evaluator with either a config object or a device
        
        Args:
            config_or_device: Either a Configuration object containing model paths and parameters,
                            or a torch.device for feature evaluation during training
        """
        if isinstance(config_or_device, torch.device):
            self.config = None
            self.device = config_or_device
        else:
            self.config = config_or_device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model, _ = create_clip_adapter(self.device)
        
        # If we have a config, load the checkpoint
        if self.config is not None:
            checkpoint_path = self.config.get_model_path("terra")
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"No checkpoint found at {checkpoint_path}")
                
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Store best validation accuracy from training
            self.best_val_acc = checkpoint.get('best_val_acc', None)
            if self.best_val_acc is not None:
                print(f"Best validation accuracy from training: {self.best_val_acc:.2f}%")
        
        self.model.eval()
    
    def evaluate(self) -> dict:
        """Evaluate the model on test data
        
        Returns:
            Dictionary containing evaluation results
        """
        if self.config is None:
            raise ValueError("Cannot run full evaluation without a config object")
            
        print("Loading test features...")
        data = torch.load(os.path.join(self.config.feature_dir, "real_data.pt"))
        
        # Get test splits
        test_splits = {split: split_data for split, split_data in data.items() 
                      if 'test' in split.lower()}
        
        if not test_splits:
            raise ValueError("No test splits found in the data!")
        
        print(f"\nFound test splits: {list(test_splits.keys())}")
        
        # Get class names from training directory
        classes = get_classes_from_folder(
            os.path.join(self.config.input_dir, "train"),
            self.config.class_mapping
        )
        print(f"Found {len(classes)} classes")
        
        # Evaluate each test split
        all_results = {}
        for split_name, split_data in test_splits.items():
            print(f"\nEvaluating {split_name} split:")
            print(f"Number of samples: {len(split_data['labels'])}")
            
            results = self.evaluate_features(
                features=split_data['features'],
                labels=split_data['labels'],
                class_names=classes,
                prompt_template=self.config.prompt_template
            )
            all_results[split_name] = results
        
        return all_results
    
    def evaluate_features(self, 
                         features: torch.Tensor,
                         labels: list,
                         class_names: list,
                         prompt_template: str = "a photo of a {}") -> dict:
        """
        Evaluate adapted CLIP features against text prompts.
        
        Args:
            features: Image features tensor [batch_size, feature_dim]
            labels: List of ground truth labels
            class_names: List of tuples (folder_name, class_name)
            prompt_template: Template for generating text prompts
            
        Returns:
            Dictionary containing evaluation results
        """
        # First collect all unique classes
        initial_classes = {class_name for _, class_name in class_names}
        all_classes = list(initial_classes.union(set(labels)))
        
        # Create mapping for all classes
        class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)}
        
        # Generate text prompts for all classes
        prompts = [prompt_template.format(class_name) for class_name in all_classes]
        
        # Move features to device
        features = features.to(self.device)
        
        # Forward pass through the model
        with torch.no_grad():
            similarity, adapted_features, text_features = self.model(features, prompts)
            
            # Move everything to CPU for accuracy calculation
            similarity = similarity.cpu()
            adapted_features = adapted_features.cpu()
            text_features = text_features.cpu()
        
        # Get predictions
        predictions = torch.argmax(similarity, dim=1)
        
        # Calculate accuracy
        correct = (predictions == torch.tensor([class_to_idx[label] for label in labels])).sum().item()
        total = len(labels)
        accuracy = correct / total * 100
        
        # Calculate per-class accuracy
        per_class_acc = {}
        for class_name, _ in class_names:
            mask = torch.tensor([l == class_name for l in labels])
            if mask.sum() > 0:
                class_correct = ((predictions == torch.tensor([class_to_idx[class_name]])) & mask).sum().item()
                class_total = mask.sum().item()
                per_class_acc[class_name] = (class_correct / class_total * 100)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_class_accuracy': per_class_acc,
            'predictions': predictions,
            'similarities': similarity
        }
    
    def print_evaluation_results(self, results: dict, detailed: bool = False) -> None:
        """Print evaluation results for all splits in a formatted way."""
        for split_name, split_results in results.items():
            print(f"\n=== Results for {split_name} ===")
            print(f"Overall Accuracy: {split_results['accuracy']:.2f}%")
            print(f"Correct predictions: {split_results['correct']}/{split_results['total']}")
            
            if detailed and 'per_class_accuracy' in split_results:
                print("\nPer-class Accuracy:")
                for class_name, acc in split_results['per_class_accuracy'].items():
                    print(f"  {class_name}: {acc:.2f}%") 