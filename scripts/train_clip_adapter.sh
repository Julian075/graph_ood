#!/bin/bash
#SBATCH --job-name=clip_adapter_train        # Name of your job
#SBATCH --output=./logs/%x_%j.out           # Output file (%x for job name, %j for job ID)
#SBATCH --error=./logs/%x_%j.err            # Error file
#SBATCH --partition=A40                      # Select a partition to submit
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=8                   # Request 8 CPU cores
#SBATCH --mem=32G                           # Request 32 GB of memory
#SBATCH --time=24:00:00                     # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
conda init bash
source ~/.bashrc
conda activate synthetic_domain
echo "Current env: $CONDA_DEFAULT_ENV"

# Define paths
DATA_DIR="./data/real_data/serengeti"
FEATURE_DIR="./data/features/serengeti"
OUTPUT_DIR="./checkpoints"
CLASS_MAPPING="./data/real_data/serengeti/class_mapping.json"
DATA_DIR_OOD="./data/real_data/terra"
FEATURE_DIR_OOD="./data/features/terra"

# Run CLIP adapter training
echo "Starting CLIP adapter training..."
python main.py \
    --mode train_clip_adapter \
    --input_dir ${DATA_DIR} \
    --input_dir_ood ${DATA_DIR_OOD} \
    --feature_dir ${FEATURE_DIR} \
    --feature_dir_ood ${FEATURE_DIR_OOD} \
    --class_mapping ${CLASS_MAPPING} \
    --use_synthetic_data True \
    --prompt_template "a photo of a {}" \
    --OOD_test True \


# Print job completion time
echo "Job finished at: $(date)" 