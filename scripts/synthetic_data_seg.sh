#!/bin/bash
#SBATCH --job-name=synthetic_generation        # Name of your job
#SBATCH --output=./logs/%x_%j.out             # Output file (%x for job name, %j for job ID)
#SBATCH --error=./logs/%x_%j.err              # Error file
#SBATCH --partition=L40S                        # Select a partition to submit
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --cpus-per-task=8                     # Request 8 CPU cores
#SBATCH --mem=32G                             # Request 32 GB of memory
#SBATCH --time=24:00:00                       # Time limit for the job (hh:mm:ss)
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=3

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
conda init bash
source ~/.bashrc
conda activate synthetic_domain
echo "Current env: $CONDA_DEFAULT_ENV"

# Launch multiple processes in parallel
python main.py --mode generate --input_dir ./data/real_data/serengeti --synthetic_dir ./data/synthetic_data_segmented/serengeti --class_mapping ./data/real_data/serengeti/class_mapping.json --images_per_class 100 --prompt_template "a photo of a {}" --use_attention True --start_idx 0 --end_idx 15 &

python main.py --mode generate --input_dir ./data/real_data/serengeti --synthetic_dir ./data/synthetic_data_segmented/serengeti --class_mapping ./data/real_data/serengeti/class_mapping.json --images_per_class 100 --prompt_template "a photo of a {}" --use_attention True --start_idx 15 --end_idx 30 &

python main.py --mode generate --input_dir ./data/real_data/serengeti --synthetic_dir ./data/synthetic_data_segmented/serengeti --class_mapping ./data/real_data/serengeti/class_mapping.json --images_per_class 100 --prompt_template "a photo of a {}" --use_attention True --start_idx 130  &

# Wait for all background processes to complete
wait

# Print job completion time
echo "Job finished at: $(date)" 