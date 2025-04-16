# Project 4: Diffusion Models

This repository contains the starter code for Project 4, which focuses on implementing diffusion models for generative modeling.

## Large Files

Due to GitHub file size limitations, the following files are not included in the repository:

- `states.pt` (The main dataset file)
- `cached_data/*.pt` (Cached dataset files)
- `checkpoints/*.pt` (Model checkpoints)
- `classifier.pt` (Final trained classifier model)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/DataScienceBioLab/reptile.git
cd reptile/Project4/starter_code
```

### 2. Download Required Files

The large files can be downloaded from the course management system or generated using the provided scripts.

### 3. Directory Structure

Ensure your directory has the following structure:

```
Project4/starter_code/
├── cached_data/             # Will be created when running the code
├── checkpoints/             # Will be created when running the code
├── plots/                   # Will be created when running the code
├── classifier.py            # Classifier implementation
├── cond_gen.py              # Conditional generation script
├── data.py                  # Dataset handling
├── models.py                # Model architectures
├── uncond_gen.py            # Unconditional generation script
└── states.pt                # Dataset file (download separately)
```

### 4. CUDA Optimizations

The code includes several optimizations for running on CUDA devices:
- Memory-efficient data loading
- Gradient accumulation for larger effective batch sizes
- Automatic Mixed Precision (AMP) training
- Proper device placement to avoid CUDA initialization errors

### 5. Training the Classifier

```bash
python classifier.py --batch_size 1024 --num_epochs 200 --num_workers 0
```

This will train the classifier model and save:
- `checkpoints/classifier_checkpoint.pt`: Latest checkpoint
- `checkpoints/classifier_best.pt`: Best model based on validation accuracy
- `classifier.pt`: Final model
- `plots/classifier_training.png`: Training curves
- `plots/classifier_prediction_map.png`: Visualization of classifier predictions

### 6. Running on HPCC

See `README_HPCC.md` for instructions on running on High-Performance Computing Clusters.

## Troubleshooting

If you encounter CUDA memory errors:
1. Reduce batch size (`--batch_size`)
2. Set `--num_workers 0` to avoid CUDA initialization errors with multiple workers
3. Clear CUDA cache manually if needed: `torch.cuda.empty_cache()` 