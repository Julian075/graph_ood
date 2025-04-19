from dataclasses import dataclass
import argparse
from typing import List, Optional

@dataclass
class Config:
    """Configuration class to store all parameters"""
    mode: str
    input_dir: str
    synthetic_dir: str
    feature_dir: str
    class_mapping: str
    images_per_class: int
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    batch_size: int = 32
    prompt_template: str = "a photo of a {}"
    test_mode: str = "clip_basic"
    
    # Additional CLIP evaluation parameters
    clip_model: str = "ViT-B/16"
    ensemble_prompts: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_prompts is None:
            self.ensemble_prompts = [
                "a photo of a {}",
                "a photograph of a {}",
                "an image of a {}",
                "a clear photo of a {}",
                "a clear photograph of a {}",
                "a bright photo of a {}",
                "a cropped photo of a {}",
                "a good photo of a {}",
                "a rendered photo of a {}",
                "a close-up photo of a {}"
            ]

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Process data for synthetic image generation and feature extraction")
    parser.add_argument("--mode", type=str, required=True, 
                      choices=["generate", "features", "train", "test"],
                      help="Operation mode: 'generate' for image generation, 'features' for feature extraction, 'train' and 'test' for model evaluation")
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Input directory containing real images in class subfolders")
    parser.add_argument("--synthetic_dir", type=str, required=True,
                      help="Directory for synthetic images")
    parser.add_argument("--feature_dir", type=str, required=True,
                      help="Directory for extracted features")
    parser.add_argument("--class_mapping", type=str,
                      help="JSON file mapping numeric folder names to actual class names")
    parser.add_argument("--images_per_class", type=int, default=100,
                      help="Number of synthetic images to generate per class")
    parser.add_argument("--start_idx", type=int,
                      help="Starting class index (inclusive)")
    parser.add_argument("--end_idx", type=int,
                      help="Ending class index (exclusive)")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for feature extraction")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}",
                      help="Template for image generation prompt. Use {} for class name placement")
    parser.add_argument("--test_mode", type=str, default="clip_basic",
                      choices=["clip_basic", "clip_ensemble"],
                      help="Test evaluation mode")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                      help="CLIP model to use")
    
    args = parser.parse_args()
    return Config(**vars(args)) 