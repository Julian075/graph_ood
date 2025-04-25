# Graph OOD Project

## Installation

### Prerequisites
- CUDA compatible GPU
- Conda or Miniconda installed

### Setup Instructions

1. Create and activate conda environment:
```bash
conda create -n synthetic_domain python=3.10 -y
conda activate synthetic_domain
```

2. Install PyTorch and torchvision (separate for CUDA compatibility):
```bash
pip install torch==2.2.2 torchvision==0.17.2
```

3. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generating Synthetic Data
To generate synthetic data, use the script `scripts/synthetic_data_seg.sh`:

```bash
bash scripts/synthetic_data_seg.sh
```

This script will:
- Generate synthetic images using Stable Diffusion XL
- Apply segmentation using Grounding DINO
- Save the processed images in the specified output directory

### Parameters
- `--input_dir`: Directory containing real data
- `--synthetic_dir`: Output directory for synthetic data
- `--class_mapping`: JSON file mapping class names
- `--images_per_class`: Number of images to generate per class
- `--prompt_template`: Either a single template string (e.g. "a photo of a {}") or path to a JSON file containing a list of templates
- `--use_attention`: Whether to use attention maps
- `--start_idx` and `--end_idx`: Range of classes to process

### Prompt Templates
You can use either a single prompt template or a JSON file containing multiple templates. For the latter, create a `prompts_templates.json` file with a list of diverse prompts.

The provided `prompts_templates.json` was created using an LLM with the following prompt:
```
Generate a list of 100 prompt templates for Stable Diffusion where {} is the main subject. The prompts should:
- Be versatile (work with any object/being)
- Vary in perspective, lighting, and context
- Avoid technical or quality terms (4k, HD, etc.)
- Be natural and concise

Format: ["prompt 1", "prompt 2", ...]
```

Example usage:
```bash
# Using a single template
python main.py --prompt_template "a photo of a {}"

# Using multiple templates from JSON
python main.py --prompt_template prompts_templates.json
```

## Project Structure
```
.
├── data/
│   ├── real_data/
│   │   └── serengeti/          # Input directory with real images
│   └── synthetic_data_segmented/
│       └── serengeti/          # Output directory for synthetic images
├── scripts/
│   └── synthetic_data_seg.sh   # Main execution script
├── src/
│   └── generation/
│       ├── synthetic_data_seg.py         # Main generation code
│       └── synthetic_data_with_attention.py
├── requirements.txt            # Project dependencies
└── README.md
```

## Data Organization

### Input Data Structure
Your input data should be organized as follows:
```
data/real_data/serengeti/
  ├── class1/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── class2/
  │   └── ...
  └── classN/
      └── ...
```

### Output Data Structure
Generated images will maintain a similar structure:
```
data/synthetic_data_segmented/serengeti/
  ├── class1/
  │   ├── class1_1.jpg
  │   └── class1_2.jpg
  ├── class2/
  │   └── ...
  └── classN/
      └── ...
```

## Tips
1. The script supports parallel processing for faster generation
2. Generated images are automatically cropped using Grounding DINO
3. The process will resume from where it left off if interrupted
4. Use `--start_idx` and `--end_idx` to process specific ranges of classes 