#!/bin/bash
#SBATCH --job-name=terra_feature_extraction    # Name of your job
#SBATCH --output=%x_%j.out                     # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err                      # Error file
#SBATCH --partition=P100                        # Select a partition to submit
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores
#SBATCH --mem=32G                             # Request 32 GB of memory
#SBATCH --time=24:00:00                       # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
conda init bash
source ~/.bashrc
conda activate synthetic_domain
echo "Current env: $CONDA_DEFAULT_ENV"

# Execute the Python script with specific arguments
# Note: We set synthetic_dir to empty string to skip synthetic data processing
python main.py \
    --mode extract \
    --input_dir ./data/real_data/terra \
    --feature_dir ./data/features/terra \
    --class_mapping ./data/real_data/terra/class_mapping.json \
    --batch_size 32

# Print job completion time
echo "Job finished at: $(date)" 