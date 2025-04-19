import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
from typing import Dict, Tuple, List
from pathlib import Path

from src.config.config import Config
from src.utils.device_utils import get_device
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.generation.synthetic_data import generate_synthetic_data, get_classes_from_folder
from src.evaluation.clip_test import ClipEvaluator
from src.training.train_adapter import train_adapter, random_search_adapter
from src.utils.hyperparameter_search import random_search

class DataProcessor:
    """Main class for data processing pipeline"""
    def __init__(self, config: Config):
        self.config = config
        self.device = get_device()
        
        # Create necessary directories
        os.makedirs(config.feature_dir, exist_ok=True)
        if config.synthetic_dir:
            os.makedirs(config.synthetic_dir, exist_ok=True)
        
        # Load class mapping
        self.class_mapping = {}
        if os.path.exists(config.class_mapping):
            with open(config.class_mapping) as f:
                self.class_mapping = json.load(f)
        
        self.feature_extractor = None
    
    def extract_features(self, image_files):
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
            
        batch_size = self.config.feature_batch_size
        all_features = []
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i + batch_size]
            batch_features = self.feature_extractor.extract_image_features(batch_files)
            all_features.append(batch_features)
            
        return torch.cat(all_features)
    
    def process_data(self) -> None:
        """Main processing function based on mode"""
        if self.config.mode == "generate":
            print("\nGenerating synthetic images...")
            print("Configuration:")
            print(f"  images_per_class: {self.config.images_per_class}")
            print(f"  prompt_template: {self.config.prompt_template}")
            if self.config.start_idx is not None:
                print(f"  start_idx: {self.config.start_idx}")
            if self.config.end_idx is not None:
                print(f"  end_idx: {self.config.end_idx}")
            
            # Get classes for generation
            classes = get_classes_from_folder(
                os.path.join(self.config.input_dir, "train"),
                self.config.class_mapping
            )
            print(f"\nFound {len(classes)} classes")
            
            # Generate synthetic images
            generate_synthetic_data(
                self.config.synthetic_dir,
                classes,
                self.config.images_per_class,
                self.config.prompt_template,
                self.config.seed,
                self.config.start_idx,
                self.config.end_idx
            )
            print("\nSynthetic data generation completed!")
            
        elif self.config.mode == "extract":
            print("\nExtracting features from real images...")
            real_features = self.extract_features(self.config.input_files)
            torch.save(real_features, os.path.join(self.config.feature_dir, "real_data.pt"))
            print("Features saved to real_data.pt")
            
        elif self.config.mode == "clip_test":
            print("\nRunning CLIP test...")
            evaluator = ClipEvaluator(self.config)
            evaluator.evaluate()
            
        elif self.config.mode == "train_adapter":
            print("\nTraining adapter...")
            # Load real data features
            real_data = torch.load(os.path.join(self.config.feature_dir, "real_data.pt"))
            train_features = real_data
            train_labels = torch.arange(len(train_features))

            # Optionally load synthetic data for augmentation
            if self.config.use_synthetic and os.path.exists(os.path.join(self.config.feature_dir, "synthetic_data.pt")):
                print("Loading synthetic data for augmentation...")
                synthetic_data = torch.load(os.path.join(self.config.feature_dir, "synthetic_data.pt"))
                train_features = torch.cat([train_features, synthetic_data])
                train_labels = torch.cat([train_labels, torch.arange(len(synthetic_data))])

            # Update real_data with combined features and labels
            real_data = {
                'features': train_features,
                'labels': train_labels
            }

            # Get classes from training directory
            classes = get_classes_from_folder(self.config.train_dir, self.config.class_mapping)
            
            # Train the adapter
            train_adapter(
                config=self.config,
                classes=classes,
                real_data=real_data,
                prompt_template=self.config.prompt_template
            )
            print(f"Training completed. Model saved to {self.config.output_dir}")
            
        elif self.config.mode == "search_adapter":
            print("\nPerforming hyperparameter search...")
            # Load real data features
            real_data = torch.load(os.path.join(self.config.feature_dir, "real_data.pt"))
            train_features = real_data
            train_labels = torch.arange(len(train_features))

            # Optionally load synthetic data for augmentation
            if self.config.use_synthetic and os.path.exists(os.path.join(self.config.feature_dir, "synthetic_data.pt")):
                print("Loading synthetic data for augmentation...")
                synthetic_data = torch.load(os.path.join(self.config.feature_dir, "synthetic_data.pt"))
                train_features = torch.cat([train_features, synthetic_data])
                train_labels = torch.cat([train_labels, torch.arange(len(synthetic_data))])

            # Update real_data with combined features and labels
            real_data = {
                'features': train_features,
                'labels': train_labels
            }

            # Get classes from training directory
            classes = get_classes_from_folder(self.config.train_dir, self.config.class_mapping)
            
            # Set random seed for reproducibility
            if self.config.seed is not None:
                torch.manual_seed(self.config.seed)
                
            # Perform random search
            best_config, best_val_acc = random_search_adapter(
                config=self.config,
                classes=classes,
                real_data=real_data,
                prompt_template=self.config.prompt_template,
                n_trials=self.config.n_trials
            )
            
            print("\nBest configuration found:")
            for key, value in best_config.items():
                print(f"  {key}: {value}")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

def parse_args() -> Config:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Feature extraction and generation pipeline")
    
    parser.add_argument('--mode', type=str, required=True,
                      choices=['extract', 'clip_test', 'train_adapter', 'search_adapter', 'generate'],
                      help='Operation mode')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory containing class folders')
    parser.add_argument('--feature_dir', type=str, required=True,
                      help='Directory to save extracted features')
    parser.add_argument('--class_mapping', type=str, required=True,
                      help='JSON file mapping folder names to class names')
    parser.add_argument('--synthetic_dir', type=str, default='synthetic_data',
                        help='Directory to store synthetic images')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--temperature', type=float, default=0.07,
                      help='Temperature parameter for contrastive loss')
    parser.add_argument('--reduction_factor', type=int, default=8,
                      help='Reduction factor for adapter')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=10,
                      help='Patience for early stopping')
    parser.add_argument('--use_synthetic', action='store_true',
                      help='Use synthetic data for training augmentation')
    
    # Hyperparameter search arguments
    parser.add_argument('--n_trials', type=int, default=20,
                      help='Number of trials for hyperparameter search')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Data processing arguments
    parser.add_argument('--start_idx', type=int, default=None,
                      help='Starting index for synthetic image generation')
    parser.add_argument('--end_idx', type=int, default=None,
                      help='Ending index for synthetic image generation')
    parser.add_argument('--prompt_template', type=str, default="a photo of a {}",
                      help='Template for generating prompts')
    parser.add_argument('--images_per_class', type=int, default=100,
                      help='Number of synthetic images to generate per class')
    
    args = parser.parse_args()
    config = Config(**vars(args))
    return config

def main() -> None:
    """Main entry point of the application"""
    config = parse_args()
    processor = DataProcessor(config)
    processor.process_data()

if __name__ == "__main__":
    main() 