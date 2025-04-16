import torch
from torch import nn
from models import MLP
from data import States
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.amp import autocast, GradScaler  # Updated import path
import os  # Add import for file operations
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts  # Import better schedulers
import argparse

# Add command-line argument parsing for HPCC flexibility
parser = argparse.ArgumentParser(description='Train a classifier for diffusion model')
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
parser.add_argument('--lr', type=float, default=1.0e-03, help='Base learning rate')
parser.add_argument('--max_lr', type=float, default=5.0e-02, help='Maximum learning rate')
parser.add_argument('--cache_dir', type=str, default='cached_data', help='Directory to cache mixed data')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--validate_every', type=int, default=5, help='Validate every N epochs')
parser.add_argument('--reset', action='store_true', help='Reset training (ignore existing checkpoints)')

args = parser.parse_args()

# Set device based on arguments or availability
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters from args
num_steps = 1000
batch_size = args.batch_size
# Reduce batch size if using CUDA to avoid memory issues
if device.type == "cuda":
    batch_size = min(batch_size, 1024)  # Cap batch size on GPU
num_epochs = args.num_epochs
lr = args.lr
max_lr = args.max_lr
weight_decay = 0.005  # Increased weight decay for better regularization
refresh_epochs = 50   # Refresh data more frequently
use_amp = True
accumulation_steps = 8 if device.type == "cuda" else 4  # More accumulation steps on GPU
grad_clip_norm = 1.0  # Lower gradient clipping for more stable training
validate_every = args.validate_every
effective_batch_size = batch_size * accumulation_steps

# Enable TensorFloat32 for much faster matrix multiplications on Ampere+ GPUs
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# Enable asynchronous CUDA operations if using CUDA
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

print(f"Running on device: {device}")
print(f"Using effective batch size of {effective_batch_size} (batch_size={batch_size}, accumulation_steps={accumulation_steps})")
print(f"Using OneCycleLR with min_lr={lr}, max_lr={max_lr}, gradient clipping={grad_clip_norm}")
print(f"Training for {num_epochs} epochs, validating every {validate_every} epochs")
print(f"Dataset will be refreshed every {refresh_epochs} epochs")
print(f"Using AMP: {use_amp}")

# Clear GPU cache to ensure we have enough memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Create cache directory
cache_dir = args.cache_dir
os.makedirs(cache_dir, exist_ok=True)

# Create checkpoint directory
checkpoint_dir = args.checkpoint_dir
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "classifier_checkpoint.pt")
best_model_path = os.path.join(checkpoint_dir, "classifier_best.pt")

# Check if we should reset training
if args.reset and os.path.exists(checkpoint_path):
    print(f"Removing existing checkpoint at {checkpoint_path}")
    os.remove(checkpoint_path)

# Get the original data from states.pt
states_data = torch.load("states.pt")
clean_data = states_data["data"].to(device)
clean_labels = states_data["labels"].to(device)

# Split the clean data and labels
indices = torch.randperm(len(clean_data))
train_size = int(0.8 * len(clean_data))
val_size = len(clean_data) - train_size

train_indices = indices[:train_size]
val_indices = indices[train_size:]

print(f"Original dataset: {len(clean_data)} samples")
print(f"Split into: {len(train_indices)} training samples, {len(val_indices)} validation samples")

# We'll only create the diffusion schedule once
beta = torch.linspace(1e-4, 0.02, num_steps, device=device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# Create separate datasets for training and validation
class StatesSubset(torch.utils.data.Dataset):
    def __init__(self, data, labels, num_steps, beta=None, alpha=None, alpha_bar=None, cache_dir=None, subset_name=None):
        """
        Create a subset of States dataset with proper diffusion sampling.
        
        Args:
            data: Clean data points
            labels: Labels for data points 
            num_steps: Total possible timesteps
            beta, alpha, alpha_bar: Pre-computed diffusion parameters
            cache_dir: Directory to cache mixed data
            subset_name: Name of this subset (e.g., 'train', 'val')
        """
        self.data = data
        self.labels = labels
        self.num_steps = num_steps
        self.n_points = len(data)
        self.device = data.device
        
        # Use provided diffusion parameters or create new ones
        if beta is not None and alpha is not None and alpha_bar is not None:
            self.beta = beta
            self.alpha = alpha
            self.alpha_bar = alpha_bar
        else:
            self.beta = torch.linspace(1e-4, 0.02, num_steps, device=self.device)
            self.alpha = 1.0 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Set cache path if provided
        self.cache_path = None
        if cache_dir is not None and subset_name is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_path = os.path.join(cache_dir, f"{subset_name}_mixed_data.pt")
        
        # Try to load cached data, otherwise mix data
        if self.cache_path and os.path.exists(self.cache_path):
            self.load_mixed_data()
        else:
            # Mix our own data
            self.custom_mix_data()
            # Save to cache if path provided
            if self.cache_path:
                self.save_mixed_data()
    
    def custom_mix_data(self):
        """Create our own implementation that stores tensors differently"""
        print(f"Mixing data for subset with {len(self.data)} samples, generating all {self.num_steps} timesteps per sample...")
        
        # Create storage for mixed samples
        self.x_t_all = []  # Noisy samples (x_t)
        self.t_all = []    # Timesteps (t)
        self.eps_all = []  # Noise (epsilon)
        self.x_all = []    # Original samples (x_0)
        self.y_all = []    # Labels (y)
        
        # For each clean sample, create noisy versions at all timesteps
        for idx in tqdm(range(len(self.data)), desc="Mixing Data"):
            x = self.data[idx]
            y = self.labels[idx]
            
            # Create noisy versions at all timesteps as per project requirements
            for t in range(1, self.num_steps + 1):
                # Calculate alpha_bar_t
                alpha_bar_t = self.alpha_bar[t-1]
                
                # Sample random noise
                eps = torch.randn_like(x)
                
                # Create noisy sample using diffusion equation
                x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps
                
                # Store all information
                self.x_t_all.append(x_t)
                self.t_all.append(t)
                self.eps_all.append(eps)
                self.x_all.append(x)
                self.y_all.append(y)
        
        # Convert lists to tensors
        self.x_t_all = torch.stack(self.x_t_all)
        self.t_all = torch.tensor(self.t_all)
        self.eps_all = torch.stack(self.eps_all)
        self.x_all = torch.stack(self.x_all)
        self.y_all = torch.stack(self.y_all)
        
        print(f"Mixed dataset contains {len(self.x_t_all)} samples")
    
    def save_mixed_data(self):
        """Save mixed data to cache file"""
        print(f"Saving mixed data to {self.cache_path}")
        torch.save({
            'x_t_all': self.x_t_all,
            't_all': self.t_all,
            'eps_all': self.eps_all,
            'x_all': self.x_all,
            'y_all': self.y_all
        }, self.cache_path)
        print("Save complete")
    
    def load_mixed_data(self):
        """Load mixed data from cache file"""
        print(f"Loading mixed data from {self.cache_path}")
        cached_data = torch.load(self.cache_path)
        self.x_t_all = cached_data['x_t_all']
        self.t_all = cached_data['t_all']
        self.eps_all = cached_data['eps_all']
        self.x_all = cached_data['x_all']
        self.y_all = cached_data['y_all']
        print(f"Loaded {len(self.x_t_all)} mixed samples")
    
    def __len__(self):
        """Return the number of mixed samples"""
        return len(self.x_t_all)
    
    def __getitem__(self, idx):
        """Get a single mixed sample by index"""
        # Return tensors on CPU to avoid CUDA memory issues during transfer
        return (
            self.x_t_all[idx].cpu() if self.x_t_all[idx].device.type == "cuda" else self.x_t_all[idx], 
            self.t_all[idx].cpu() if self.t_all[idx].device.type == "cuda" else self.t_all[idx], 
            self.eps_all[idx].cpu() if self.eps_all[idx].device.type == "cuda" else self.eps_all[idx], 
            self.x_all[idx].cpu() if self.x_all[idx].device.type == "cuda" else self.x_all[idx], 
            self.y_all[idx].cpu() if self.y_all[idx].device.type == "cuda" else self.y_all[idx]
        )

# Create train and validation datasets with separate data samples (reusing diffusion parameters)
print("Creating/loading training dataset...")
train_dataset = StatesSubset(clean_data[train_indices], clean_labels[train_indices], 
                           num_steps, beta, alpha, alpha_bar,
                           cache_dir=cache_dir, subset_name="train")
print("Creating/loading validation dataset...")
val_dataset = StatesSubset(clean_data[val_indices], clean_labels[val_indices], 
                          num_steps, beta, alpha, alpha_bar,
                          cache_dir=cache_dir, subset_name="val")

print(f"Dataset split: {len(train_indices)} training samples, {len(val_indices)} validation samples")

# Create separate dataloaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0,  # Force 0 workers to prevent CUDA initialization issues
                             pin_memory=True if device.type == "cuda" else False,
                             persistent_workers=False)  # Disable persistent workers to avoid memory leaks
val_dataloader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, 
                           num_workers=0,  # Force 0 workers to prevent CUDA initialization issues
                           pin_memory=True if device.type == "cuda" else False,
                           persistent_workers=False)  # Disable persistent workers to avoid memory leaks

print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

# TODO: create the architecture with the hidden size layers from the
# PDF.
# Input: noisy sample xt (2D) + timestep t (1D) = 3D
# Output: 5-class logits
hidden_layers = [100, 200, 500]
classifier = MLP(input_dim=3, output_dim=5, hidden_layers=hidden_layers).to(device)

# Use torch.compile for PyTorch 2.0+ to speed up training
if hasattr(torch, 'compile'):
    # Use dynamic=True for faster compilation
    classifier = torch.compile(classifier, mode='reduce-overhead', dynamic=True)
    print("Using torch.compile() for acceleration with reduced overhead")

# TODO: Create loss function, optimizer, and scheduler. 
ce_loss = nn.CrossEntropyLoss()
# Switch to AdamW with weight decay
optimizer = torch.optim.AdamW(
    classifier.parameters(), 
    lr=lr,
    weight_decay=weight_decay,
    betas=(0.9, 0.99)  # Higher beta2 for better handling of sparse gradients
)

# Use OneCycleLR scheduler with LR test results
scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,  # Peak at steepest LR
    total_steps=num_epochs * len(train_dataloader),  # Correct total steps = epochs * batches per epoch
    pct_start=0.3,  # Spend 30% of training warming up
    div_factor=max_lr/lr,  # Start with base lr
    final_div_factor=10,  # End with base_lr/10
    anneal_strategy='cos'  # Cosine annealing
)

# Create AMP GradScaler
scaler = GradScaler(enabled=use_amp)

# Create training and validation trackers
train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_acc = 0.0
best_epoch = 0

# Check for checkpoint and load it
start_epoch = 0
if os.path.exists(checkpoint_path) and not args.reset:
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_accs = checkpoint.get('val_accs', [])
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    best_epoch = checkpoint.get('best_epoch', 0)
    print(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.2f}% at epoch {best_epoch}")
else:
    print("Starting training from scratch")

# Define training and validation functions
def train_epoch(model, dataloader, criterion, optimizer, scaler):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar for batches
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    # Zero gradients once at the beginning
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        x_t, t, eps, x, y = batch
        
        # Move data to the correct device
        x_t = x_t.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Normalize timestep
        t_normalized = (2 * (t.float() - num_steps / 2) / num_steps).unsqueeze(1)
        inp = torch.cat([x_t, t_normalized], dim=1)
        
        # Forward pass with AMP
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(inp)
            loss = criterion(logits, y)
            loss = loss / accumulation_steps  # Normalize loss for gradient accumulation
        
        # Backward pass with AMP
        if use_amp:
            scaler.scale(loss).backward()
            
            # Step every accumulation_steps batches or at the end
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Apply gradient clipping
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            
            # Step every accumulation_steps batches or at the end
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                
                optimizer.step()
                optimizer.zero_grad()
        
        # Update statistics
        total_loss += loss.item() * accumulation_steps  # Multiply back to get actual loss
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item() * accumulation_steps, "acc": 100. * correct / total})
    
    # Clean up progress bar
    pbar.close()
    
    # Calculate epoch statistics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Validation")
    for batch_idx, (x_t, t, _, _, y) in enumerate(pbar):
        x_t = x_t.to(device)
        t = t.to(device)
        y = y.to(device)
        
        # Normalize timestep
        t_normalized = (2 * (t.float() - num_steps / 2) / num_steps).unsqueeze(1)
        
        # Combine inputs
        inputs = torch.cat([x_t, t_normalized], dim=1)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, y)
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1), 
            'acc': 100. * correct / total
        })
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# Training loop
print(f"Starting classifier training for {num_epochs} epochs...")
for epoch in range(start_epoch, num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Train for one epoch
    train_loss, train_acc = train_epoch(classifier, train_dataloader, ce_loss, optimizer, scaler)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Print current learning rate
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1} completed. Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, LR: {current_lr:.6f}")
    
    # Step the scheduler
    scheduler.step()
    
    # Validate periodically
    if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
        val_loss, val_acc = validate(classifier, val_dataloader, ce_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"New best model with validation accuracy: {val_acc:.2f}%")
            torch.save(classifier.state_dict(), best_model_path)
    else:
        # Add placeholder for plotting
        if val_losses:
            val_losses.append(val_losses[-1])
            val_accs.append(val_accs[-1])
        else:
            val_losses.append(float('nan'))
            val_accs.append(float('nan'))
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
    }, checkpoint_path)
    
    # Plot training curves every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join('plots', 'classifier_training.png'))
        plt.close()
    
    # Refresh dataset
    if (epoch + 1) % refresh_epochs == 0:
        print(f"Refreshing dataset at epoch {epoch+1}")
        # Free up CUDA memory before creating new dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Create new training dataset with fresh noise
        train_dataset = StatesSubset(clean_data[train_indices], clean_labels[train_indices], 
                                  num_steps, beta, alpha, alpha_bar,
                                  cache_dir=cache_dir, subset_name=f"train_refresh_{epoch+1}")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                    num_workers=0,  # Force 0 workers to prevent CUDA initialization issues
                                    pin_memory=True if device.type == "cuda" else False,
                                    persistent_workers=False)  # Disable persistent workers to avoid memory leaks

print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

# Load best model for prediction map generation
if os.path.exists(best_model_path):
    print(f"Loading best model from epoch {best_epoch}")
    classifier.load_state_dict(torch.load(best_model_path))
else:
    print("Using final model for prediction map")

# Save final model
torch.save(classifier.state_dict(), "classifier.pt")

# Add state names and colors just before prediction map generation
label_to_states = {
    0: "Michigan",
    1: "Idaho",
    2: "Ohio", 
    3: "Oklahoma",
    4: "Wisconsin"
}
colors = ["red", "blue", "green", "orange", "purple"]
cmap = ListedColormap(colors)

# Generate a classifier prediction map
print("Generating classifier prediction map...")
classifier.eval()

# Create a grid of points to visualize the classifier's decision boundaries
x_min, x_max = -3, 3
y_min, y_max = -3, 3
step = 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_tensor = torch.from_numpy(grid_points).float().to(device)

# Generate predictions for all timesteps
print("Generating predictions for all timesteps...")
prediction_maps = []

# Predict for several timesteps
timesteps = [1, 100, 250, 500, 750, 999]  # Sample at different noise levels
for timestep in tqdm(timesteps):
    # Prepare inputs with normalized timestep
    t_normalized = torch.full((grid_points_tensor.size(0), 1), 
                             2 * (timestep - num_steps / 2) / num_steps, 
                             device=device)
    inputs = torch.cat([grid_points_tensor, t_normalized], dim=1)
    
    # Process in batches to avoid OOM
    batch_size = 4096
    predictions = []
    
    for i in range(0, inputs.size(0), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        with torch.no_grad():
            outputs = classifier(batch_inputs)
            batch_predictions = outputs.argmax(dim=1).cpu().numpy()
        predictions.append(batch_predictions)
    
    # Combine batch predictions
    grid_predictions = np.concatenate(predictions)
    prediction_map = grid_predictions.reshape(xx.shape)
    prediction_maps.append((timestep, prediction_map))

# Create a multi-panel plot
n_timesteps = len(timesteps)
fig, axs = plt.subplots(1, n_timesteps, figsize=(n_timesteps*5, 5))

# Create plots for each timestep
for i, (timestep, prediction_map) in enumerate(prediction_maps):
    ax = axs[i]
    im = ax.imshow(prediction_map, extent=[x_min, x_max, y_min, y_max], 
                 origin='lower', cmap=cmap, alpha=0.7)
    
    # Plot the actual state data
    for label in range(5):
        mask = clean_labels.cpu().numpy() == label
        ax.scatter(clean_data[mask, 0].cpu().numpy(), 
                 clean_data[mask, 1].cpu().numpy(), 
                 c=colors[label], 
                 label=label_to_states[label],
                 edgecolors='k', s=40, alpha=0.6)
    
    ax.set_title(f'Timestep {timestep}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# Add a common legend
handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()
plt.savefig('plots/classifier_prediction_map.png', bbox_inches='tight')
print("Classifier prediction map saved to plots/classifier_prediction_map.png")

def train_classifier(
    dataset_root,
    csv_path,
    num_epochs=30,
    batch_size=64,
    lr=0.0085,  # Min loss LR
    max_lr=0.059,  # Steepest LR
    weight_decay=1e-4,
    device=None,
    use_amp=True,
    refresh_epochs=5,
    gradient_accumulation_steps=1,
    save_path="classifier.pt"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Adjust batch size and accumulation steps for CUDA
    if device.type == "cuda":
        batch_size = min(batch_size, 1024)  # Cap batch size on GPU
        gradient_accumulation_steps = max(gradient_accumulation_steps, 4)  # Ensure enough accumulation steps
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"OneCycleLR parameters: base_lr={lr}, max_lr={max_lr}, epochs={num_epochs}")
    print(f"Weight decay: {weight_decay}, AMP enabled: {use_amp}")
    print(f"Dataset will be refreshed every {refresh_epochs} epochs")
    
    # Create dataset and dataloader
    dataset = States(dataset_root=dataset_root, csv_path=csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0)  # Force 0 workers to prevent CUDA initialization issues
    
    # Create model architecture
    hidden_layers = [100, 200, 500]
    classifier = MLP(input_dim=3, output_dim=5, hidden_layers=hidden_layers).to(device)
    
    # Loss function, optimizer, and scheduler
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use OneCycleLR scheduler with LR test results
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,  # Peak at steepest LR
        total_steps=num_epochs * len(dataloader),  # Correct total steps = epochs * batches per epoch
        pct_start=0.3,  # Spend 30% of training warming up
        div_factor=max_lr/lr,  # Start with base lr
        final_div_factor=10,  # End with base_lr/10
        anneal_strategy='cos'  # Cosine annealing
    )
    
    # Create AMP GradScaler
    scaler = GradScaler(enabled=use_amp)
    
    # Gradient clipping value
    grad_clip_norm = 1.0
    
    # Training loop
    train_loss_list = []
    classifier.train()
    for epoch in trange(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(dataloader):
            x_, t, eps, x, y = batch
            
            # Move tensors to device, using non_blocking for better performance
            x_ = x_.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Normalize timestep
            t_normalized = (2 * (t.float() - num_steps / 2) / num_steps).unsqueeze(1)
            inp = torch.cat([x_, t_normalized], dim=1)
            
            # Forward pass with AMP
            with autocast(device_type=device.type, enabled=use_amp):
                logits = classifier(inp)
                loss = ce_loss(logits, y)
                # Normalize loss by accumulation steps
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with AMP and gradient accumulation
            if use_amp:
                scaler.scale(loss).backward()
                
                # Step every accumulation_steps batches
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    # Apply gradient clipping
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                # Step every accumulation_steps batches
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    # Apply gradient clipping
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps  # Multiply back to get actual loss
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        train_loss_list.append(avg_epoch_loss)
        
        # Step the scheduler once per epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}")
        
        # Refresh data periodically
        if (epoch + 1) % refresh_epochs == 0:
            print(f"Refreshing dataset noise at epoch {epoch+1}")
            # Clear GPU cache before dataset refresh
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
            dataset.mix_data()
            # Recreate dataloader as dataset content changed
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                   num_workers=0)  # Force 0 workers to prevent CUDA initialization issues
    
    # Save final model
    torch.save(classifier.state_dict(), save_path)
    print(f"Classifier model saved to {save_path}")
    
    return classifier, train_loss_list

if __name__ == "__main__":
    # Training will start here when script is executed
    pass

