#!/bin/bash
#SBATCH --job-name=python_test
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=general
#SBATCH --output=%x-%j.log

# Load required modules
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4

# Navigate to your project directory
cd ~/Project4/starter_code

# Run your Python script
python test_script.py

echo "Job completed" 