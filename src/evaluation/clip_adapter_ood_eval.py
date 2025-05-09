import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import clip
from tqdm import tqdm
import os
from typing import List, Dict, Optional, Tuple

class ClipAdapterOODEvaluator:
    def __init__(self, model, classes: List[str], ood_test: bool, config, checkpoint_path: str = None):
        """
        Initialize the evaluator for OOD adapter.
        
        Args:
            model: The CLIPAdapterOOD model
            classes: List of class names
            ood_test: Whether to evaluate on OOD test set
            config: Configuration object containing model settings
            checkpoint_path: Optional path to the checkpoint to load. If not provided,
                           will try to use a default name based on synthetic data usage.
        """
        self.config = config
        self.device = config.device
        self.ood_test = ood_test
        # Ensure classes are strings
        self.classes = [str(c) if isinstance(c, tuple) else c for c in classes]
        self.prompt_template = config.prompt_template
        
        # Load CLIP model
        self.clip, _ = clip.load(config.clip_model, device=config.device)
        self.clip.eval()
        
        # Load adapter model from checkpoint
        if checkpoint_path is None:
            # Use default paths as fallback
            if config.use_synthetic_data:
                checkpoint_path = os.path.join(config.output_dir, 'clip_adapter_ood', 'adapter_ood_checkpoint_synthetic.pt')
            else:
                checkpoint_path = os.path.join(config.output_dir, 'clip_adapter_ood', 'adapter_ood_checkpoint.pt')
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model = model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get text features for classes
        self.text_features = self.get_text_features()

    def get_missing_classes(self, test_labels: List[str]) -> List[str]:
        """Find classes in test set that weren't in training"""
        test_classes = set(test_labels)
        train_classes = set(self.classes)
        return sorted(list(test_classes - train_classes))

    def evaluate_test_set(self, test_features: torch.Tensor, test_labels: List[str], split_name: str) -> Dict[str, float]:
        """
        Evaluate model on a specific test set.
        
        Args:
            test_features: Test set features tensor
            test_labels: List of test labels
            split_name: Name of the split for reporting
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert test labels to strings if they're tuples
        test_labels = [str(label) if isinstance(label, tuple) else label for label in test_labels]
        
        # Collect all unique classes (from training and test)
        all_classes = sorted(set(self.classes).union(set(test_labels)))
        
        # Create mapping for all classes
        class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)}
        
        # Get text features for all classes (including new ones)
        self.text_features = self.get_text_features(all_classes)
        
        # Convert labels to indices
        label_indices = torch.tensor([class_to_idx[label] for label in test_labels])
        
        # Prepare data loader
        dataset = TensorDataset(test_features)
        data_loader = DataLoader(
            dataset, 
            batch_size=self.config.clip_adapter_ood['batch_size'],
            shuffle=False,
            pin_memory=True
        )
        
        all_predictions = []
        all_similarities = []
        all_node_logits = []
        
        # Predict in batches
        with torch.no_grad():
            for batch_features, in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                # Move batch to device and ensure float32
                batch_features = batch_features.to(self.device).float()
                batch_features = F.normalize(batch_features, dim=-1)
                
                # Forward pass
                adapter_features, node_logits = self.model(batch_features)
                
                # Normalize features
                adapter_features = F.normalize(adapter_features, dim=-1)
                
                # Compute similarities with text features
                similarity = torch.matmul(adapter_features, self.text_features.T)
                similarity = F.normalize(similarity, dim=-1)
                
                # Get predictions
                predictions = torch.argmax(similarity, dim=1)
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_similarities.append(similarity.cpu())
                if node_logits is not None:
                    all_node_logits.append(node_logits.cpu())
        
        # Concatenate all predictions and similarities
        predictions = torch.cat(all_predictions)
        similarities = torch.cat(all_similarities)
        if all_node_logits:
            node_logits = torch.cat(all_node_logits)
        else:
            node_logits = None
        
        # Calculate accuracy
        correct = (predictions == label_indices).sum().item()
        total = len(test_labels)
        accuracy = correct / total * 100
        
        # Calculate per-class accuracy
        per_class_acc = {}
        for class_name in all_classes:
            mask = torch.tensor([l == class_name for l in test_labels])
            if mask.sum() > 0:
                class_correct = ((predictions == label_indices) & mask).sum().item()
                class_total = mask.sum().item()
                per_class_acc[class_name] = (class_correct / class_total * 100)
        
        # Calculate node classification accuracy if available
        node_acc = None
        if node_logits is not None:
            node_predictions = torch.argmax(node_logits, dim=1)
            node_correct = (node_predictions == label_indices).sum().item()
            node_acc = node_correct / total * 100
        
        print(f"\nResults for {split_name}:")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"Correct predictions: {correct}/{total}")
        if node_acc is not None:
            print(f"Node Classification Accuracy: {node_acc:.2f}%")
        
        print("\nPer-class accuracy:")
        for class_name, acc in per_class_acc.items():
            prefix = "* " if class_name not in self.classes else "  "
            print(f"{prefix}{class_name}: {acc:.2f}%")
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_class_accuracy': per_class_acc,
            'predictions': predictions,
            'similarities': similarities,
            'all_classes': all_classes
        }
        
        if node_acc is not None:
            results.update({
                'node_accuracy': node_acc,
                'node_logits': node_logits
            })
            
        return results

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all test splits.
        
        Returns:
            Dictionary containing evaluation metrics for each split
        """
        # Load test data
        feature_dir = self.config.feature_dir_ood if self.ood_test else self.config.feature_dir
        data = torch.load(os.path.join(feature_dir, "real_data.pt"))
        
        # Get test splits
        test_splits = {split: split_data for split, split_data in data.items() 
                      if 'test' in split.lower()}
        
        if not test_splits:
            raise ValueError("No test splits found in the data!")
            
        print(f"\nFound test splits: {list(test_splits.keys())}")
        
        results = {}
        
        # Evaluate each test split
        for split_name, split_data in test_splits.items():
            print(f"\nEvaluating {split_name} split...")
            results[split_name] = self.evaluate_test_set(
                test_features=split_data['features'],
                test_labels=split_data['labels'],
                split_name=split_name
            )
        
        # Print summary
        print("\nEvaluation Summary:")
        for split_name, metrics in results.items():
            print(f"\n{split_name}:")
            print(f"Accuracy: {metrics['accuracy']:.2f}%")
            print(f"Correct predictions: {metrics['correct']}/{metrics['total']}")
            if 'node_accuracy' in metrics:
                print(f"Node Classification Accuracy: {metrics['node_accuracy']:.2f}%")
        
        return results

    def encode_text(self, text_prompts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode text prompts using CLIP with explicit precision handling"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.clip.encode_text(text_tokens).float()  # Ensure float32
            if normalize:
                text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def get_text_features(self, class_names: Optional[List[str]] = None) -> torch.Tensor:
        """Get text features for specified classes or self.classes if none specified."""
        if class_names is None:
            class_names = self.classes
            
        templates = [self.prompt_template] if isinstance(self.prompt_template, str) else self.prompt_template
        all_text_features = []
        
        with torch.no_grad():
            for template in templates:
                prompts = [template.format(class_name) for class_name in class_names]
                text_features = []
                
                for i in range(0, len(prompts), self.config.clip_adapter_ood['batch_size']):
                    batch_prompts = prompts[i:i + self.config.clip_adapter_ood['batch_size']]
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