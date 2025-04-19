import os
import argparse
from diffusers import DiffusionPipeline
import torch
from PIL import Image
from tqdm import tqdm
import random
import json


def generate_synthetic_data(output_folder, classes, images_per_class, prompt_template="a photo of a {}", 
                          seed=None, start_idx=None, end_idx=None):
    """Generate synthetic images for each class using Stable Diffusion."""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        pipe.to("cuda")
    except Exception as e:
        raise RuntimeError(f"Error initializing Stable Diffusion: {e}")
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    os.makedirs(output_folder, exist_ok=True)
    
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else len(classes)
        classes = classes[start:end]
        print(f"\nProcessing classes from index {start} to {end}")
    
    for folder_name, class_name in classes:
        print(f"\nGenerating images for class: {class_name}")
        if folder_name != class_name:
            print(f"Using folder name: {folder_name}")
        
        class_dir = os.path.join(output_folder, folder_name)
        os.makedirs(class_dir, exist_ok=True)
        
        existing_images = len([f for f in os.listdir(class_dir) 
                             if f.endswith(('.jpg', '.png'))])
        
        if existing_images >= images_per_class:
            print(f"Skipping {class_name}: already has {existing_images} images")
            continue

        for i in tqdm(range(existing_images, images_per_class)):
            prompt = prompt_template.format(class_name)
            if i == 1:
                print(f"Example prompt: {prompt}")
            try:
                image = pipe(
                    prompt=prompt,
                    generator=torch.manual_seed(seed + i if seed is not None else random.randint(0, 2**32 - 1))
                ).images[0]
                
                image_path = os.path.join(class_dir, f"{folder_name}_{i+1}.jpg")
                image = image.convert('RGB')
                image.save(image_path, format='JPEG')
                
            except Exception as e:
                print(f"Error generating image {i+1} for {class_name}: {e}")
                continue

