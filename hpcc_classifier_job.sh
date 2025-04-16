#!/bin/bash
#SBATCH --job-name=classifier_train
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --output=%x-%j.log

# Load required modules
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
module load PyTorch/1.13.1-foss-2022a-CUDA-11.7.0

# Activate your virtual environment if needed
# source ~/path/to/venv/bin/activate

# Navigate to your project directory
cd ~/Project4/starter_code

# List available files
echo "Files in directory:"
ls -la

# Run your Python script
python -c "
import classifier
print('Available functions in classifier module:')
for func in dir(classifier):
    if not func.startswith('_'):
        print(f'- {func}')
"

# Main training run
python -c "
import classifier
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

echo "Job completed" 