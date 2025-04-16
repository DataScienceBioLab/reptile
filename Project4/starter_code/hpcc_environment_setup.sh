#!/bin/bash
# Script to set up the Project4 environment on MSU HPCC
# This script follows the guidelines from https://docs.icer.msu.edu/Using_conda/

echo "======================================"
echo "Setting up Project4 conda environment"
echo "======================================"

# Unload any Python modules
module purge

# Load the Miniforge3 module (Quick Start Option)
echo "Loading Miniforge3 module..."
module load Miniforge3  # Use module load Conda/3 if using manually installed conda

# Create a conda environment for Project4
echo "Creating project4 conda environment with PyTorch..."
conda create -n project4 python=3.10 -y

# Activate the environment
echo "Activating environment..."
source activate project4

# Install PyTorch and related packages
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other required packages
echo "Installing additional packages..."
conda install matplotlib numpy tqdm -y
conda install -c conda-forge ipywidgets jupyterlab -y

# Show environment info
echo "======================================"
echo "Environment setup complete!"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "GPU model: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "======================================"
echo "To activate this environment, run:"
echo "module load Miniforge3"
echo "source activate project4"
echo "To submit a job, run:"
echo "sbatch hpcc_classifier_job.sh"
echo "======================================" 