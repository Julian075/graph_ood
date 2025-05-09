import torch
import clip
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Union
import numpy as np
from tqdm import tqdm
import random

class ImageDataset(Dataset):
    """Dataset class for loading images"""
    def __init__(self, image_paths: List[str], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

class FeatureExtractor:
    """Handle all CLIP model operations and feature extraction"""
    def __init__(self, classes: List[str], device: str, model_name: str = "ViT-B/16", batch_size: int = 32, seed: int = 42):
        map_class = {}
        for folder_name, class_name in classes:
            map_class[folder_name] = class_name
        self.classes = map_class
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        print(f"Loading CLIP {model_name} model on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print("CLIP model loaded successfully")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def encode_text(self, text_prompts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encode text prompts using CLIP"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            if normalize:
                text_features = torch.nn.functional.normalize(text_features, dim=-1)
            text_features = text_features.float()  # Ensure float32
        return text_features
    
    def extract_batch(self, images: List[Image.Image], normalize: bool = True) -> torch.Tensor:
        """Extract features from a batch of images in memory"""
        with torch.no_grad():
            inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            features = self.model.encode_image(inputs)
            if normalize:
                features = torch.nn.functional.normalize(features, dim=-1)
            features = features.float()  # Ensure float32
        return features

    def extract_from_paths(self, image_paths: List[str], num_workers: int = 4) -> torch.Tensor:
        """Extract features from a list of image paths using DataLoader"""
        dataset = ImageDataset(image_paths, self.preprocess)
        def my_collate(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.default_collate(batch)

        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=my_collate
        )
        
        features = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if batch is None:
                    continue
                batch = batch.to(self.device)
                batch_features = self.model.encode_image(batch)
                batch_features = torch.nn.functional.normalize(batch_features, dim=-1)
                batch_features = batch_features.float()  # Ensure float32
                features.append(batch_features.cpu())
                
        return torch.cat(features)

    def get_image_paths_and_labels(self, directory: str) -> Tuple[List[str], List[str]]:
        """Get all image paths and their corresponding class labels from a directory"""
        image_paths = []
        labels = []
        
        for root, _, files in os.walk(directory):
            class_name = os.path.basename(root)
            if class_name in self.classes:  # Only process if class is in mapping
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
                        labels.append(self.classes[class_name])
                    
        return image_paths, labels

    def process_directory(self, data_dir: str) -> Dict[str, Dict[str, Union[torch.Tensor, List[str]]]]:
        """Process a directory of images and save features
        
        Args:
            data_dir: Path to the directory containing the images
            
        Returns:
            Dictionary with structure:
            {
                'split_name': {
                    'features': torch.Tensor,  # shape [N, 512]
                    'labels': List[str],
                    'paths': List[str]
                }
            }
        """
        print(f"\nProcessing directory: {data_dir}")
        
        result = {}

        # Process each split directory
        for split_dir in os.listdir(data_dir):
            # Skip hidden directories and .ipynb_checkpoints
            if split_dir.startswith('.') or split_dir == '.ipynb_checkpoints':
                continue
                
            split_path = os.path.join(data_dir, split_dir)
            if not os.path.isdir(split_path):
                continue

            # Handle special cases for split names
            split_name = split_dir
            if split_dir == 'VOC2007' or split_dir == 'photo':
                split_name = 'train'
            elif split_dir in ['SUN09', 'LabelMe', 'Caltech101', 'sketch', 'cartoon', 'art_painting']:
                split_name = f'test_{split_dir}'

            # Initialize lists for this split
            all_features = []
            all_labels = []
            all_paths = []
            
            # Process each class directory
            for class_dir in os.listdir(split_path):
                class_path = os.path.join(split_path, class_dir)
                if os.path.isdir(class_path):
                    image_paths, labels = self.get_image_paths_and_labels(class_path)
                    if image_paths:
                        features = self.extract_from_paths(image_paths)
                        all_features.append(features)
                        all_labels.extend(labels)
                        all_paths.extend(image_paths)
            
            if all_features:  # If we found any features
                # Concatenate all features for this split
                result[split_name] = {
                    'features': torch.cat(all_features),
                    'labels': all_labels,
                    'paths': all_paths
                }
        
        if not result:
            raise ValueError(f"No valid data found in {data_dir}")
            
        return result 

    def process_synthetic_directory(self, data_dir: str) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """Process a directory of synthetic images
        
        Args:
            data_dir: Path to the directory containing synthetic images
            
        Returns:
            Dictionary with structure:
            {
                'features': torch.Tensor,  # shape [N, 512]
                'labels': List[str],
                'paths': List[str]
            }
        """
        print(f"\nProcessing synthetic directory: {data_dir}")
        
        # Initialize lists for all data
        all_features = []
        all_labels = []
        all_paths = []
        
        # Process each class directory
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                image_paths, labels = self.get_image_paths_and_labels(class_path)
                if image_paths:
                    features = self.extract_from_paths(image_paths)
                    all_features.append(features)
                    all_labels.extend(labels)
                    all_paths.extend(image_paths)
        
        if not all_features:
            raise ValueError(f"No valid data found in {data_dir}")
            
        # Concatenate all features
        result = {
            'features': torch.cat(all_features),
            'labels': all_labels,
            'paths': all_paths
        }
            
        return result 
