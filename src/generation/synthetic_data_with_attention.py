import os
import argparse
from diffusers import DiffusionPipeline
import torch
from PIL import Image
from tqdm import tqdm
import random
import json
import numpy as np
from typing import Optional

class CustomAttnProcessor:
    def __init__(self, attention_maps):
        self.attention_maps = attention_maps

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_probs = attn.get_attention_scores(hidden_states, encoder_hidden_states, attention_mask)
        
        if encoder_hidden_states is not None:
            self.attention_maps.append(attention_probs.detach().cpu())
        
        hidden_states = torch.bmm(attention_probs, encoder_hidden_states or hidden_states)
        hidden_states = attn.reshape_batch_dim_to_heads(hidden_states)
        
        return hidden_states

def generate_synthetic_data_with_attention(output_folder, classes, images_per_class, prompt_template="a photo of a {}", 
                          seed=None, start_idx=None, end_idx=None):
    """Generate synthetic images and segment them using attention maps from SDXL."""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    try:
        # Initialize the pipeline
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Enable attention map extraction
        attention_maps = []
        custom_processor = CustomAttnProcessor(attention_maps)
        
        # Set the processor for all cross-attention blocks
        for name, module in pipe.unet.named_modules():
            if "attn2" in name:  # cross attention blocks
                module.processor = custom_processor
        
        pipe.to("cuda")
        
    except Exception as e:
        raise RuntimeError(f"Error initializing SDXL: {e}")
    
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
        print(f"\nGenerating segmented images for class: {class_name}")
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
                # Clear previous attention maps
                attention_maps.clear()
                
                # Generate image
                image = pipe(
                    prompt=prompt,
                    generator=torch.manual_seed(seed + i if seed is not None else random.randint(0, 2**32 - 1))
                ).images[0]
                
                # Process attention maps if available
                if attention_maps:
                    # Combine attention maps from different layers
                    combined_attention = torch.cat(attention_maps, dim=0)
                    # Average across heads and tokens
                    attention_mask = combined_attention.mean(0).mean(0)
                    
                    # Normalize the mask
                    attention_mask = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())
                    
                    # Convert mask to proper size and format
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                    attention_mask = torch.nn.functional.interpolate(
                        attention_mask, 
                        size=image.size[::-1],  # PIL Image size is (width, height)
                        mode='bilinear'
                    )
                    attention_mask = attention_mask.squeeze().numpy()
                    
                    # Threshold the mask to make it binary
                    threshold = 0.5  # Adjust this threshold as needed
                    binary_mask = (attention_mask > threshold).astype(np.uint8) * 255
                    
                    # Convert image to numpy array
                    image_array = np.array(image)
                    
                    # Apply mask to get segmented object
                    mask_3channel = np.stack([binary_mask] * 3, axis=-1)
                    
                    # Apply mask to image
                    segmented_image = image_array * (mask_3channel / 255.0)
                    
                    # Find bounding box of the mask
                    rows = np.any(binary_mask, axis=1)
                    cols = np.any(binary_mask, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    
                    # Crop the segmented image to the bounding box
                    segmented_image = segmented_image[rmin:rmax+1, cmin:cmax+1]
                    
                    # Convert back to PIL Image
                    segmented_image = Image.fromarray(segmented_image.astype(np.uint8))
                    
                    # Save segmented image
                    image_path = os.path.join(class_dir, f"{folder_name}_{i+1}_segmented.jpg")
                    segmented_image = segmented_image.convert('RGB')
                    segmented_image.save(image_path, format='JPEG', quality=95)
                
            except Exception as e:
                print(f"Error generating image {i+1} for {class_name}: {e}")
                continue

def generate_segmented_image(output_folder, prompt, output_name, seed=None):
    """Generate a single image and segment it using attention maps from SDXL."""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    try:
        # Initialize the pipeline
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Enable attention map extraction
        attention_maps = []
        custom_processor = CustomAttnProcessor(attention_maps)
        
        # Set the processor for all cross-attention blocks
        for name, module in pipe.unet.named_modules():
            if "attn2" in name:  # cross attention blocks
                module.processor = custom_processor
        
        pipe.to("cuda")
        
    except Exception as e:
        raise RuntimeError(f"Error initializing SDXL: {e}")
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Clear previous attention maps
        attention_maps.clear()
        
        # Generate image
        image = pipe(
            prompt=prompt,
            generator=torch.manual_seed(seed if seed is not None else random.randint(0, 2**32 - 1))
        ).images[0]
        
        # Process attention maps if available
        if attention_maps:
            # Combine attention maps from different layers
            combined_attention = torch.cat(attention_maps, dim=0)
            # Average across heads and tokens
            attention_mask = combined_attention.mean(0).mean(0)
            
            # Normalize the mask
            attention_mask = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())
            
            # Convert mask to proper size and format
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            attention_mask = torch.nn.functional.interpolate(
                attention_mask, 
                size=image.size[::-1],  # PIL Image size is (width, height)
                mode='bilinear'
            )
            attention_mask = attention_mask.squeeze().numpy()
            
            # Threshold the mask to make it binary
            threshold = 0.5  # Adjust this threshold as needed
            binary_mask = (attention_mask > threshold).astype(np.uint8) * 255
            
            # Convert image to numpy array
            image_array = np.array(image)
            
            # Apply mask to get segmented object
            mask_3channel = np.stack([binary_mask] * 3, axis=-1)
            
            # Apply mask to image
            segmented_image = image_array * (mask_3channel / 255.0)
            
            # Find bounding box of the mask
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop the segmented image to the bounding box
            segmented_image = segmented_image[rmin:rmax+1, cmin:cmax+1]
            
            # Convert back to PIL Image
            segmented_image = Image.fromarray(segmented_image.astype(np.uint8))
            
            # Save segmented image
            image_path = os.path.join(output_folder, f"{output_name}_segmented.jpg")
            segmented_image = segmented_image.convert('RGB')
            segmented_image.save(image_path, format='JPEG', quality=95)
            
            return image_path
            
    except Exception as e:
        print(f"Error generating segmented image: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic segmented images using SDXL attention maps")
    parser.add_argument("--output_folder", type=str, required=True,
                      help="Directory to store segmented images")
    parser.add_argument("--class_mapping", type=str, required=True,
                      help="JSON file mapping folder names to class names")
    parser.add_argument("--images_per_class", type=int, default=100,
                      help="Number of synthetic images to generate per class")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}",
                      help="Template for generating prompts")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducibility")
    parser.add_argument("--start_idx", type=int, default=None,
                      help="Starting index for generation")
    parser.add_argument("--end_idx", type=int, default=None,
                      help="Ending index for generation")
    
    args = parser.parse_args()
    
    with open(args.class_mapping, 'r') as f:
        class_mapping = json.load(f)
    
    classes = [(folder_name, class_name) for folder_name, class_name in class_mapping.items()]
    
    generate_synthetic_data_with_attention(
        output_folder=args.output_folder,
        classes=classes,
        images_per_class=args.images_per_class,
        prompt_template=args.prompt_template,
        seed=args.seed,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    ) 