import torch
from typing import List, Dict, Tuple, Union
from src.feature_extraction.feature_extractor import FeatureExtractor
import os
import json

class ClipEvaluator:
    def __init__(self,classes, config):
        self.config = config
        self.extractor = FeatureExtractor(
            classes=classes,
            device=config.device,
            model_name=config.clip_model,
            batch_size=32
        )
    
    def evaluate(self) -> Dict:
        """
        Main evaluation method that loads data and evaluates CLIP performance.
        """
        print("\nLoading features and preparing for evaluation...")
        
        # Load features from the feature directory
        feature_file = os.path.join(self.config.feature_dir, "real_data.pt")
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found at {feature_file}")
        
        # Load all splits
        all_data = torch.load(feature_file)
        
        # Get test splits
        test_splits = {split: split_data for split, split_data in all_data.items() 
                      if 'test' in split.lower()}
        
        if not test_splits:
            raise ValueError("No test splits found in the data!")
        
        print(f"\nFound test splits: {list(test_splits.keys())}")
        
        # Load class mapping
        if not os.path.exists(self.config.class_mapping):
            raise FileNotFoundError(f"Class mapping file not found at {self.config.class_mapping}")
        
        with open(self.config.class_mapping) as f:
            class_mapping = [(k, v) for k, v in json.loads(f.read()).items()]
        
        # Evaluate each test split
        all_results = {}
        for split_name, split_data in test_splits.items():
            print(f"\nEvaluating {split_name} split:")
            print(f"Number of samples: {len(split_data['labels'])}")
            
            results = self.evaluate_features(
                features=split_data['features'],
                labels=split_data['labels'],
                class_names=class_mapping,
                prompt_template=self.config.prompt_template
            )
            all_results[split_name] = results
            
            # Print results for this split
            print(f"\nResults for {split_name}:")
            self.print_evaluation_results(results, detailed=True)
        
        # If there's only one test split, return its results directly
        if len(all_results) == 1:
            return next(iter(all_results.values()))
        
        return all_results
    
    def evaluate_features(self, 
                         features: torch.Tensor,
                         labels: List[str],
                         class_names: List[Tuple[str, str]],
                         prompt_template: Union[str, List[str]] = "a photo of a {}") -> Dict:
        """
        Evaluate CLIP features against text prompts.
        
        Args:
            features: Image features tensor (already normalized during extraction)
            labels: List of ground truth labels
            class_names: List of tuples (folder_name, class_name)
            prompt_template: Single template or list of templates for generating text prompts.
                           If a string, uses that single template.
                           If a list, uses all templates and averages the results.
            
        Returns:
            Dictionary containing evaluation results
        """
        # First collect all unique classes (from class_names and labels)
        initial_classes = {class_name for _, class_name in class_names}
        #all_classes = list(initial_classes.union(set(labels))) #
        all_classes = sorted(initial_classes.union(set(labels)))
        # Create mapping for all classes
        class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)} 
        
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
                batch_features = self.extractor.encode_text(batch_prompts)
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
        
        # Convert labels to indices using the complete mapping
        label_indices = torch.tensor([class_to_idx[label] for label in labels])
        
        # Move features to the same device as text_features for matrix multiplication
        features = features.to(self.config.device)
        text_features = text_features.to(self.config.device)
        
        # Calculate similarity scores (cosine similarity since both are normalized)
        similarity = torch.matmul(features, text_features.T)
        
        # Get predictions
        predictions = torch.argmax(similarity, dim=1)
        
        # Move tensors to CPU for accuracy calculations
        predictions = predictions.cpu()
        label_indices = torch.tensor([class_to_idx[label] for label in labels])  # create on CPU
        
        # Calculate accuracy
        correct = (predictions == label_indices).sum().item()
        total = len(labels)
        accuracy = correct / total * 100
        
        # Calculate per-class accuracy
        per_class_acc = {}
        for class_name, _ in class_names:
            mask = torch.tensor([l == class_name for l in labels])  # create on CPU
            if mask.sum() > 0:
                class_correct = ((predictions == label_indices) & mask).sum().item()
                class_total = mask.sum().item()
                per_class_acc[class_name] = (class_correct / class_total * 100)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_class_accuracy': per_class_acc,
            'predictions': predictions,
            'similarities': similarity.cpu()  # move similarities to CPU for return
        }
    
    def print_evaluation_results(self, results: Dict, detailed: bool = False) -> None:
        """Print evaluation results in a formatted way."""
        print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")
        print(f"Correct predictions: {results['correct']}/{results['total']}")
        
        if detailed and 'per_class_accuracy' in results:
            print("\nPer-class Accuracy:")
            for class_name, acc in results['per_class_accuracy'].items():
                print(f"  {class_name}: {acc:.2f}%") 