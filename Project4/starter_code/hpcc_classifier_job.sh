#!/bin/bash
#SBATCH --job-name=classifier_training
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=classifier_job_%j.log

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"

# Unload any Python modules first (required before loading Conda)
module purge
module load Miniforge3  # Or use: module load Conda/3 if using manual installation

# Activate your conda environment (create it first with: conda create -n project4 python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia)
source activate project4

# Make sure we're in the correct directory
cd $SLURM_SUBMIT_DIR

# Show environment info
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "GPU model: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Set TensorFloat32 to improve performance on NVIDIA Ampere+ GPUs
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the classifier training script
python classifier.py

# Print job end info
echo "End time: $(date)"
echo "Job completed" 