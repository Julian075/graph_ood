import os
import argparse
from diffusers import DiffusionPipeline
import torch
from PIL import Image
from tqdm import tqdm
import random
import json
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.inference import box_convert
import pkg_resources

def get_groundingdino_paths():
    """Get the paths for GroundingDINO config and weights from the installed package."""
    package_path = pkg_resources.resource_filename('groundingdino', '')
    config_path = os.path.join(package_path, 'config/GroundingDINO_SwinT_OGC.py')
    weights_path = os.path.join(package_path, 'weights/groundingdino_swint_ogc.pth')
    return config_path, weights_path

def generate_synthetic_data(output_folder, classes, images_per_class, prompt_template="a photo of a {}", 
                          seed=None, start_idx=None, end_idx=None):
    """Generate synthetic images for each class using Stable Diffusion and crop them using Grounding DINO."""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    try:
        # Initialize SDXL
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        pipe.to("cuda")
        
        # Initialize Grounding DINO
        print("Getting Grounding DINO paths...")
        config_path, weights_path = get_groundingdino_paths()
        print(f"Config path: {config_path}")
        print(f"Weights path: {weights_path}")
        print("Initializing Grounding DINO...")
        groundingdino_model = load_model(config_path, weights_path)
        print("Models initialized successfully")
        
    except Exception as e:
        raise RuntimeError(f"Error initializing models: {e}")
    
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
        
        # Contador de imágenes exitosamente guardadas
        successful_images = len([f for f in os.listdir(class_dir) 
                               if f.endswith(('.jpg', '.png'))])
        
        if successful_images >= images_per_class:
            print(f"Skipping {class_name}: already has {successful_images} images")
            continue

        attempts = 0
        while successful_images < images_per_class and attempts < images_per_class * 2:
            attempts += 1
            prompt = prompt_template.format(class_name)
            if successful_images == 0:
                print(f"Example prompt: {prompt}")
            try:
                # Generate image with SDXL
                image = pipe(
                    prompt=prompt,
                    generator=torch.manual_seed(seed + attempts if seed is not None else random.randint(0, 2**32 - 1))
                ).images[0]
                
                # Convert PIL Image to numpy array for Grounding DINO
                image_source = np.array(image)
                
                # Run Grounding DINO inference using the same prompt
                boxes, logits, phrases = predict(
                    model=groundingdino_model,
                    image=image_source,
                    caption=class_name,  # Using just the class name for better detection
                    box_threshold=0.35,
                    text_threshold=0.25
                )
                
                if len(boxes) > 0:
                    # Get the box with highest confidence
                    best_box_idx = logits.argmax()
                    box = boxes[best_box_idx]
                    
                    # Convert normalized coordinates to pixel coordinates
                    H, W = image_source.shape[:2]
                    box_abs = box_convert(box, in_fmt='xyxy', out_fmt='xyxy')
                    box_abs = np.array([
                        box_abs[0] * W,  # x1
                        box_abs[1] * H,  # y1
                        box_abs[2] * W,  # x2
                        box_abs[3] * H   # y2
                    ]).astype(int)
                    
                    # Crop image using the detected box
                    cropped_image = image.crop((box_abs[0], box_abs[1], box_abs[2], box_abs[3]))
                    
                    # Save only the cropped image
                    successful_images += 1
                    image_path = os.path.join(class_dir, f"{folder_name}_{successful_images}.jpg")
                    cropped_image = cropped_image.convert('RGB')
                    cropped_image.save(image_path, format='JPEG', quality=95)
                    print(f"Successfully saved image {successful_images}/{images_per_class} for {class_name}")
                else:
                    print(f"No object detected for {class_name}, attempt {attempts} - retrying")
                    continue
                
            except Exception as e:
                print(f"Error in attempt {attempts} for {class_name}: {e}")
                continue
            
        if successful_images < images_per_class:
            print(f"Warning: Only generated {successful_images}/{images_per_class} images for {class_name} after {attempts} attempts") 