# Synthetic Data Generation with Stable Diffusion

This repository contains scripts for generating synthetic image data using Stable Diffusion XL.

## Environment Setup

1. Create a new conda environment:
```bash
conda create -n synthetic_data python=3.10
conda activate synthetic_data
```

2. Install PyTorch with CUDA support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install other required packages:
```bash
pip install diffusers==0.24.0
pip install transformers
pip install accelerate
pip install safetensors
pip install tqdm
pip install pillow
```

## File Structure

- `synthetic_data.py`: Main script for generating synthetic images
- `class_mapping.json`: (Optional) Mapping file for numeric folder names to class names

## Usage

### Basic Usage

1. If your folders are already named with class names:
```bash
python synthetic_data.py \
  --train_folder "/path/to/train/folder" \
  --output_folder "/path/to/output/folder" \
  --images_per_class 100 \
  --preset diverse
```

2. If your folders are numeric and you have a mapping file:
```bash
python synthetic_data.py \
  --train_folder "/path/to/train/folder" \
  --output_folder "/path/to/output/folder" \
  --class_mapping "class_mapping.json" \
  --images_per_class 100 \
  --preset diverse
```

### Available Presets

- `default`: Balanced quality and adherence to prompt
- `diverse`: More diverse outputs, good for training data
- `creative`: Very creative outputs, less faithful to prompt
- `precise`: More precise adherence to prompt
- `fast`: Faster generation, slightly lower quality
- `quality`: Higher quality, slower generation

To see all presets:
```bash
python synthetic_data.py --list-presets
```

### Custom Configuration

You can override preset values:
```bash
python synthetic_data.py \
  --train_folder "/path/to/train/folder" \
  --output_folder "/path/to/output/folder" \
  --preset diverse \
  --guidance_scale 5.0 \
  --num_inference_steps 50
```

## Folder Structure

Your training data should be organized as follows:

```
train_folder/
  ├── class1/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── class2/
  │   └── ...
  └── class3/
      └── ...
```

Or with numeric folders:
```
train_folder/
  ├── 0/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── 1/
  │   └── ...
  └── 2/
      └── ...
```

## Class Mapping File

If using numeric folders, create a `class_mapping.json`:
```json
{
    "0": "actual_class_name1",
    "1": "actual_class_name2",
    "2": "actual_class_name3"
}
```

## Output Structure

The generated images will maintain the same folder structure as your input:
```
output_folder/
  ├── class1/
  │   ├── class1_1.jpg
  │   └── class1_2.jpg
  ├── class2/
  │   └── ...
  └── class3/
      └── ...
```

## Tips

1. Start with the `diverse` preset for training data generation
2. Use `--seed` for reproducible results
3. The script will resume from where it left off if interrupted
4. Generated images are saved in JPG format
5. Use `quality` preset if you need higher quality images 