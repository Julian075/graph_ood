#!/bin/bash
#SBATCH --job-name=terra_clip_eval          # Name of your job
#SBATCH --output=./logs/%x_%j.out                  # Output file (%x for job name, %j for job ID)
#SBATCH --error=./logs/%x_%j.err                   # Error file
#SBATCH --partition=V100                    # Select a partition to submit
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=8                   # Request 8 CPU cores
#SBATCH --mem=32G                           # Request 32 GB of memory
#SBATCH --time=2:00:00                      # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
conda init bash
source ~/.bashrc
conda activate synthetic_domain
echo "Current env: $CONDA_DEFAULT_ENV"

# Execute the Python script with clip_test mode
python main.py \
    --input_dir ./data/real_data/terra \
    --mode clip_test \
    --feature_dir ./data/features/terra \
    --class_mapping ./data/real_data/terra/class_mapping.json \
    --prompt_template a photo of a {}

# Print job completion time
echo "Job finished at: $(date)" 