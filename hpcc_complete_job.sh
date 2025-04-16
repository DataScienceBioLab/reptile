#!/bin/bash
#SBATCH --job-name=full_torch
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --output=%x-%j.log

# Use conda directly from installation path
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Check if environment exists, create if it doesn't
ENV_NAME="torch_env"
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment with PyTorch..."
    conda create -n $ENV_NAME -y python=3.9
fi

# Activate the environment with explicit command
echo "Activating conda environment..."
source ~/miniconda3/bin/activate $ENV_NAME

# Install required packages
echo "Installing dependencies..."
conda install -y pytorch torchvision torchaudio -c pytorch
conda install -y scipy numpy matplotlib pandas scikit-learn

# Navigate to your project directory
cd ~/Project4/starter_code

# List available packages
echo "Installed packages:"
pip list | grep -E "torch|scipy|numpy|pandas"

# List files in directory
echo "Files in directory:"
ls -la

# Run your Python script
echo "Testing PyTorch installation:"
python -c "
import torch
import scipy
import numpy
print('PyTorch version:', torch.__version__)
print('SciPy version:', scipy.__version__)
print('NumPy version:', numpy.__version__)
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