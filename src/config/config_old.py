from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import numpy as np
import random
import os
import argparse

@dataclass
class Config:
    """Configuration class to store all parameters"""
    # Basic parameters (common for all models)
    mode: str
    input_dir: str
    synthetic_dir: str
    feature_dir: str
    class_mapping: str
    images_per_class: int
    start_idx: int = None
    end_idx: int = None
    batch_size: int = 32
    prompt_template: str = "a photo of a {}"
    
    # CLIP base model configuration
    clip_model: str = "ViT-B/16"  # Model architecture
    device: str = "cuda"  # Device to run on (cuda/cpu)
    clip_dim: int = 512   # Dimensión de CLIP (normalmente 512)
    
    # ====== CLIP Adapter specific parameters ======
    # Architecture
    reduction_factor: int = 8  # Factor de reducción para el cuello de botella del adapter
    
    # Training
    learning_rate: float = 1e-4  # Tasa de aprendizaje para el adapter
    temperature: float = 0.07    # Temperatura para la función de pérdida contrastiva
    num_epochs: int = 50         # Número máximo de épocas para entrenar el adapter
    patience: int = 10           # Épocas para early stopping del adapter
    min_delta: float = 0.1       # Mejora mínima en val_acc para early stopping
    use_synthetic: bool = True   # Usar datos sintéticos para entrenamiento del adapter
    
    # Hyperparameter search for adapter
    do_search: bool = False      # Whether to perform hyperparameter search
    n_trials: int = 20          # Número de trials para búsqueda
    seed: int = 42              # Semilla para reproducibilidad
    # ============================================
    
    # Evaluation configuration (common for all models)
    ensemble_prompt_templates: List[str] = None  # Will be set in __post_init__
    confidence_threshold: float = 0.5  # Threshold for confident predictions
    
    def __post_init__(self):
        if self.ensemble_prompt_templates is None:
            self.ensemble_prompt_templates = [
                "a photo of a {}",
                "a photograph of a {}",
                "an image of a {}",
                "a clear photo of a {}",
                "a clear image of a {}"
            ]
    
    def setup_seed(self):
        """Setup all random seeds for reproducibility"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    @property
    def bottleneck_dim(self) -> int:
        """Get bottleneck dimension based on CLIP dimension and reduction factor (for CLIP adapter)"""
        return self.clip_dim // self.reduction_factor
    
    def get_model_path(self, dataset_name: str) -> str:
        """Get path where CLIP adapter model should be saved/loaded"""
        return f"checkpoints/clip_adapter/{dataset_name}_{self.bottleneck_dim}_{self.temperature}/best_model.pt"
    
    @classmethod
    def get_search_space(cls) -> dict:
        """Get hyperparameter search space for CLIP adapter tuning"""
        return {
            'learning_rate': ((1e-5, 1e-3), 'log'),     # (range, scale)
            'batch_size': ([32, 64, 128, 256], 'choice'),
            'temperature': ((0.01, 0.1), 'linear'),
            'reduction_factor': ([4, 8, 16], 'choice'),
            'num_epochs': ([30, 50, 100], 'choice')
        }
    
    def __str__(self) -> str:
        """String representation for logging"""
        return (
            f"Config(\n"
            f"  Basic Parameters:\n"
            f"    mode={self.mode}\n"
            f"    batch_size={self.batch_size}\n"
            f"  CLIP Base Model:\n"
            f"    clip_model={self.clip_model}\n"
            f"    clip_dim={self.clip_dim}\n"
            f"  CLIP Adapter Parameters:\n"
            f"    reduction_factor={self.reduction_factor}\n"
            f"    bottleneck_dim={self.bottleneck_dim}\n"
            f"    learning_rate={self.learning_rate}\n"
            f"    temperature={self.temperature}\n"
            f"    num_epochs={self.num_epochs}\n"
            f"    patience={self.patience}\n"
            f"    min_delta={self.min_delta}\n"
            f"    use_synthetic={self.use_synthetic}\n"
            f"  Hyperparameter Search:\n"
            f"    do_search={self.do_search}\n"
            f"    n_trials={self.n_trials}\n"
            f"    seed={self.seed}\n"
            f"  Data:\n"
            f"    input_dir={self.input_dir}\n"
            f"    feature_dir={self.feature_dir}\n"
            f"    synthetic_dir={self.synthetic_dir}\n"
            f"    class_mapping={self.class_mapping}\n"
            f")"
        )

    def __init__(self):
        parser = argparse.ArgumentParser(description='CLIP Adapter Training and Evaluation')
        
        # Mode selection
        parser.add_argument('--mode', type=str, required=True,
                          choices=['generate', 'extract', 'clip_test', 'train_adapter', 'search_adapter'],
                          help='Operation mode')
        
        # Directory paths
        parser.add_argument('--input_dir', type=str, required=True,
                          help='Input directory containing images')
        parser.add_argument('--feature_dir', type=str, required=True,
                          help='Directory to save extracted features')
        parser.add_argument('--output_dir', type=str, required=True,
                          help='Directory to save output models')
        parser.add_argument('--synthetic_dir', type=str, default='synthetic_data',
                          help='Directory to save synthetic images')
        parser.add_argument('--train_dir', type=str, required=True,
                          help='Directory containing training data')
        parser.add_argument('--class_mapping', type=str, required=True,
                          help='Path to class mapping JSON file')
        
        # Generation parameters
        parser.add_argument('--images_per_class', type=int, default=100,
                          help='Number of synthetic images to generate per class')
        parser.add_argument('--start_idx', type=int, default=None,
                          help='Starting index for class generation')
        parser.add_argument('--end_idx', type=int, default=None,
                          help='Ending index for class generation')
        
        # Feature extraction parameters
        parser.add_argument('--feature_batch_size', type=int, default=32,
                          help='Batch size for feature extraction')
        parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                          help='CLIP model to use')
        
        # Training parameters
        parser.add_argument('--batch_size', type=int, default=128,
                          help='Training batch size')
        parser.add_argument('--learning_rate', type=float, default=1e-4,
                          help='Learning rate')
        parser.add_argument('--temperature', type=float, default=0.07,
                          help='Temperature parameter for contrastive loss')
        parser.add_argument('--reduction_factor', type=int, default=8,
                          help='Reduction factor for adapter')
        parser.add_argument('--num_epochs', type=int, default=100,
                          help='Number of training epochs')
        parser.add_argument('--patience', type=int, default=10,
                          help='Patience for early stopping')
        parser.add_argument('--use_synthetic', action='store_true',
                          help='Use synthetic data for training')
        
        # Prompt parameters
        parser.add_argument('--prompt_template', type=str, default='a photo of a {}',
                          help='Template for text prompts')
        parser.add_argument('--prompt_templates', nargs='+',
                          default=['a photo of a {}', 'a picture of a {}', 'an image of a {}'],
                          help='List of templates for text prompts')
        
        # Search parameters
        parser.add_argument('--n_trials', type=int, default=20,
                          help='Number of trials for hyperparameter search')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed for reproducibility')
        
        # Device
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                          help='Device to use for computation')
        
        args = parser.parse_args()
        
        # Set all arguments as attributes
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
            
        # Create directories if they don't exist
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.synthetic_dir, exist_ok=True)
        
        # Convert input files to list for feature extraction
        if hasattr(self, 'input_dir') and os.path.exists(self.input_dir):
            self.input_files = []
            for root, _, files in os.walk(self.input_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.input_files.append(os.path.join(root, file)) 