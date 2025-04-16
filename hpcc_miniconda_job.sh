#!/bin/bash
#SBATCH --job-name=miniconda_torch
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --output=%x-%j.log

# Use conda directly from installation path
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Create a new conda environment with PyTorch
echo "Creating conda environment with PyTorch..."
conda create -n torch_env -y python=3.9
conda init
source ~/.bashrc

# Activate the environment with explicit command
echo "Activating conda environment..."
source ~/miniconda3/bin/activate torch_env

# Install PyTorch
echo "Installing PyTorch..."
conda install -y pytorch torchvision torchaudio -c pytorch

# Navigate to your project directory
cd ~/Project4/starter_code

# List available packages
echo "Installed packages:"
pip list | grep torch

# List files in directory
echo "Files in directory:"
ls -la

# Run your Python script
echo "Testing PyTorch installation:"
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device count:', torch.cuda.device_count())
    print('CUDA device name:', torch.cuda.get_device_name(0))
"

# Main training run
echo "Starting classifier training..."
python -c "
import classifier
print('Available functions in classifier module:')
for func in dir(classifier):
    if not func.startswith('_'):
        print(f'- {func}')

print('Starting classifier training...')
classifier.train_classifier(
    num_epochs=10,
    batch_size=64,
    lr=0.0085,
    max_lr=0.059,
    weight_decay=1e-4,
    grad_accumulation_steps=1
)
print('Training completed')
"

# Deactivate the environment
conda deactivate

echo "Job completed" 