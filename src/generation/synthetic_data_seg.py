import os
import torch
import random
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, Union

def generate_synthetic_data_seg(
    output_folder: str,
    classes: list,
    images_per_class: int,
    prompt_templates: Union[str, List[str]] = ["a photo of a {}"],
    seed: int = None,
    start_idx: int = None,
    end_idx: int = None
):
    """Generate synthetic images for each class using Stable Diffusion and crop them using Grounding DINO via Hugging Face.
    
    Args:
        output_folder: Directory to save generated images
        classes: List of (folder_name, class_name) tuples
        images_per_class: Number of images to generate per class
        prompt_templates: Single template string or list of template strings (e.g. ["a photo of a {}", "an image of a {}"]).
                        One will be randomly chosen for each image.
        seed: Random seed for reproducibility
        start_idx: Start index in classes list
        end_idx: End index in classes list
    """
    random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")

    # Convert single template to list for uniform processing
    if isinstance(prompt_templates, str):
        prompt_templates = [prompt_templates]
    
    print(f"Using {len(prompt_templates)} different prompt templates:")
    for template in prompt_templates:
        print(f"  - {template}")

    # Initialize Stable Diffusion XL
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")

    # Initialize Grounding DINO from Hugging Face
    print("Loading Grounding DINO from Hugging Face Transformers...")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")
    print("Models initialized successfully")

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

        successful_images = len([
            f for f in os.listdir(class_dir)
            if f.endswith(('.jpg', '.png'))
        ])

        if successful_images >= images_per_class:
            print(f"Skipping {class_name}: already has {successful_images} images")
            continue

        attempts = 0
        while successful_images < images_per_class and attempts < images_per_class * 2:
            attempts += 1
            # Randomly select a prompt template for each generation
            template = random.choice(prompt_templates)
            prompt = template.format(class_name)
            
            if successful_images == 0:
                print(f"Example prompt: {prompt}")
            try:
                image = pipe(
                    prompt=prompt,
                    generator=torch.manual_seed(seed + attempts if seed is not None else random.randint(0, 2**32 - 1))
                ).images[0]

                # Preprocess and run Grounding DINO
                inputs = processor(images=image, text=f"{class_name}.", return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.35,
                    text_threshold=0.25,
                    target_sizes=[image.size[::-1]]
                )

                if results and len(results[0]["boxes"]) > 0:
                    best_idx = results[0]["scores"].argmax().item()
                    box = results[0]["boxes"][best_idx].tolist()
                    box = [int(b) for b in box]

                    # Crop and save
                    cropped_image = image.crop(box)
                    successful_images += 1
                    save_path = os.path.join(class_dir, f"{folder_name}_{successful_images}.jpg")
                    cropped_image.convert("RGB").save(save_path, "JPEG", quality=95)
                    print(f"Saved {successful_images}/{images_per_class} for {class_name} using template: {template}")
                else:
                    print(f"No object detected for {class_name}, attempt {attempts}")
            except Exception as e:
                print(f"Error in attempt {attempts} for {class_name}: {e}")
                continue

        if successful_images < images_per_class:
            print(f"Warning: Only generated {successful_images}/{images_per_class} for {class_name}")
