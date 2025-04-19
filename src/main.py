import os
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.config.config import Config
from src.models.feature_extractor import FeatureExtractor
from src.training.train_adapter import train_adapter, random_search_adapter
from src.evaluation.clip_test import clip_test
from src.data.synthetic_data import generate_synthetic_data
from src.utils.utils import get_classes_from_folder

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.feature_extractor = FeatureExtractor(model_name=config.clip_model)
        
    def extract_features(self, image_files):
        """Extract features from a list of image files using batched processing."""
        all_features = []
        batch_size = self.config.feature_batch_size
        
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i + batch_size]
            batch_features = self.feature_extractor.extract_image_features(batch_files)
            all_features.append(batch_features)
            
        return torch.cat(all_features, dim=0)
    
    def process_data(self):
        """Process data based on the specified mode."""
        if self.config.mode == "generate":
            # Get classes from training directory
            classes = get_classes_from_folder(self.config.train_dir)
            
            # Generate synthetic data
            generate_synthetic_data(
                output_folder=self.config.synthetic_dir,
                classes=classes,
                images_per_class=self.config.images_per_class,
                prompt_template=self.config.prompt_template,
                seed=self.config.seed,
                start_idx=self.config.start_idx,
                end_idx=self.config.end_idx
            )
            print(f"Generated synthetic data in {self.config.synthetic_dir}")
            
        elif self.config.mode == "extract":
            # Extract features from input images
            features = self.extract_features(self.config.input_files)
            
            # Save features
            feature_path = os.path.join(self.config.feature_dir, "features.pt")
            torch.save(features, feature_path)
            print(f"Saved features to {feature_path}")
            
        elif self.config.mode == "clip_test":
            # Run CLIP test
            classes = get_classes_from_folder(self.config.train_dir)
            clip_test(
                feature_dir=self.config.feature_dir,
                classes=classes,
                prompt_templates=self.config.prompt_templates
            )
            
        elif self.config.mode == "train_adapter":
            # Load real data features
            real_data = torch.load(os.path.join(self.config.feature_dir, "real_data.pt"))
            train_features = real_data["train_features"]
            train_labels = real_data["train_labels"]
            
            # Optionally load and combine synthetic data
            if self.config.use_synthetic:
                synthetic_data = torch.load(os.path.join(self.config.feature_dir, "synthetic_data.pt"))
                train_features = torch.cat([train_features, synthetic_data["features"]], dim=0)
                train_labels = torch.cat([train_labels, synthetic_data["labels"]], dim=0)
            
            # Update real_data with combined features
            real_data["train_features"] = train_features
            real_data["train_labels"] = train_labels
            
            # Get classes from training directory
            classes = get_classes_from_folder(self.config.train_dir)
            
            # Train adapter
            train_adapter(
                config=self.config,
                real_data=real_data,
                classes=classes,
                prompt_template=self.config.prompt_template
            )
            
        elif self.config.mode == "search_adapter":
            # Load real data features
            real_data = torch.load(os.path.join(self.config.feature_dir, "real_data.pt"))
            train_features = real_data["train_features"]
            train_labels = real_data["train_labels"]
            
            # Optionally load and combine synthetic data
            if self.config.use_synthetic:
                synthetic_data = torch.load(os.path.join(self.config.feature_dir, "synthetic_data.pt"))
                train_features = torch.cat([train_features, synthetic_data["features"]], dim=0)
                train_labels = torch.cat([train_labels, synthetic_data["labels"]], dim=0)
            
            # Update real_data with combined features
            real_data["train_features"] = train_features
            real_data["train_labels"] = train_labels
            
            # Get classes from training directory
            classes = get_classes_from_folder(self.config.train_dir)
            
            # Set random seed for reproducibility
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.seed)
            
            # Perform random search
            best_config, best_val_acc = random_search_adapter(
                config=self.config,
                classes=classes,
                prompt_template=self.config.prompt_template,
                n_trials=self.config.n_trials
            )
            
            print("\nBest configuration:")
            for key, value in best_config.items():
                print(f"  {key}: {value}")
            print(f"\nBest validation accuracy: {best_val_acc:.4f}")

def main():
    config = Config()
    processor = DataProcessor(config)
    processor.process_data()

if __name__ == "__main__":
    main() 