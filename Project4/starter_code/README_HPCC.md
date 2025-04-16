# Running the Classifier on MSU HPCC

This guide explains how to run the diffusion model classifier on MSU's High Performance Computing Center (HPCC) system.

## Setup Instructions

Follow these steps to set up your environment and run the classifier on HPCC:

### 1. Connect to HPCC

SSH into the HPCC gateway:

```bash
ssh your_msu_netid@hpcc.msu.edu
```

### 2. Set Up the Environment

There are two methods to set up the required Conda environment:

#### Option 1: Automatic Setup (Recommended)

Run the setup script to create a conda environment with all required packages:

```bash
# Navigate to your project directory
cd path/to/Project4/starter_code

# Make the setup script executable
chmod +x hpcc_environment_setup.sh

# Run the setup script
./hpcc_environment_setup.sh
```

#### Option 2: Manual Setup

If you prefer to set up the environment manually:

```bash
# Unload any Python modules
module purge

# Load Miniforge3 module
module load Miniforge3

# Create a new environment
conda create -n project4 python=3.10 -y
conda activate project4

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install additional packages
conda install matplotlib numpy tqdm -y
```

### 3. Submit the Job

Once the environment is set up, you can submit the job to the SLURM scheduler:

```bash
# Make the job script executable
chmod +x hpcc_classifier_job.sh

# Submit the job
sbatch hpcc_classifier_job.sh
```

You can monitor your job status with:

```bash
squeue -u your_msu_netid
```

### 4. Advanced Usage

The classifier script supports various command-line arguments for flexibility:

```bash
# Example with custom parameters
python classifier.py --batch_size 2048 --num_epochs 100 --lr 0.003 --max_lr 0.01 --validate_every 10 --reset
```

Available arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Batch size for training | 4096 |
| `--num_epochs` | Number of epochs to train | 200 |
| `--device` | Device to use (cuda or cpu) | Auto-detect |
| `--lr` | Base learning rate | 5.0e-03 |
| `--max_lr` | Maximum learning rate | 2.0e-02 |
| `--cache_dir` | Directory to cache mixed data | cached_data |
| `--num_workers` | Number of workers for data loading | 4 |
| `--checkpoint_dir` | Directory to save checkpoints | checkpoints |
| `--validate_every` | Validate every N epochs | 5 |
| `--reset` | Reset training (ignore existing checkpoints) | False |

### 5. Customizing Resources

If you need to customize the resources requested for your job, edit the SLURM directives at the top of `hpcc_classifier_job.sh`:

```bash
#SBATCH --job-name=classifier_training
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
```

Adjust these values based on your specific requirements.

## Troubleshooting

### Common Issues

1. **Conda conflicts**: If you see errors related to Conda packages, it might be due to conflicts between locally installed packages and Conda packages. Try moving your locally installed packages to a backup location:

   ```bash
   mv $HOME/.local/lib/pythonX.Y $HOME/.local/lib/pythonX.Y.backup
   ```

2. **Out of memory errors**: If you encounter GPU memory issues, try reducing the batch size or increasing gradient accumulation steps.

3. **Module loading issues**: Always ensure you've properly unloaded conflicting modules with `module purge` before loading Conda.

### Getting Help

For further assistance with HPCC-specific issues, refer to:
- [ICER Documentation](https://docs.icer.msu.edu/)
- Contact HPCC support at icer@msu.edu 