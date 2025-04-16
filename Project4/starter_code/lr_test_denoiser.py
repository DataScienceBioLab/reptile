# lr_test_denoiser.py - Learning Rate Finder for denoiser model
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models import MLP
from data import States

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_steps = 500
batch_size = 10000
test_epochs = 5  # Short test
use_amp = True   # Enable AMP
start_lr = 1e-7  # Starting learning rate 
end_lr = 1.0     # Ending learning rate

# Prepare dataset (smaller for quicker testing)
print("Loading dataset...")
dataset = States(num_steps=num_steps)
# Use a subset for faster testing
subset_size = len(dataset) // 5  # 20% of the data
indices = torch.randperm(len(dataset))[:subset_size]
subset_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, indices),
    batch_size=batch_size, 
    shuffle=True
)

# Create denoiser model (MLP with 4 hidden layers of 256 units each)
print("Creating denoiser model...")
hidden_layers_denoiser = [256, 256, 256, 256]
model = MLP(input_dim=3, output_dim=2, hidden_layers=hidden_layers_denoiser).to(device)
criterion = nn.MSELoss()  # Denoiser uses MSE loss to predict noise

# Setup optimizer with starting LR
optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=0.01)

# Setup AMP
scaler = GradScaler(enabled=use_amp)

# Calculate LR multiplier per batch
total_batches = test_epochs * len(subset_loader)
gamma = (end_lr / start_lr) ** (1 / total_batches)

print(f"Testing LR from {start_lr:.1e} to {end_lr:.1e} over {total_batches} batches")
print(f"LR multiplier per batch: {gamma:.4f}")

# Lists to store results
lrs = []
losses = []
batch_num = 0

# Training loop
for epoch in range(test_epochs):
    for batch in tqdm(subset_loader, desc=f"Epoch {epoch+1}/{test_epochs}"):
        # Get the learning rate for this batch
        lr = start_lr * (gamma ** batch_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get data
        x_, t, eps, x, y = batch  # eps is the noise we want to predict
        x_ = x_.to(device)
        t = t.to(device)
        eps = eps.to(device)
        
        # Normalize timestep
        t_normalized = (2 * (t.float() - num_steps / 2) / num_steps).unsqueeze(1)
        inp = torch.cat([x_, t_normalized], dim=1)
        
        # Forward pass with AMP - denoiser predicts the noise (eps)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            pred_noise = model(inp)
            loss = criterion(pred_noise, eps)
        
        # Record the loss and learning rate
        lrs.append(lr)
        losses.append(loss.item())
        
        # Backward pass with AMP
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        batch_num += 1
        
        # Stop if loss is exploding
        if not np.isfinite(loss.item()) or loss.item() > 10:
            print(f"Loss exploded at lr={lr:.6f}. Stopping early.")
            break
    
    # Check if we stopped early
    if not np.isfinite(loss.item()) or loss.item() > 10:
        break

# Plot results
plt.figure(figsize=(10, 6))
plt.semilogx(lrs, losses)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder - Denoiser Model')
plt.grid(True)
plt.axhline(min(losses), ls='--', color='r')  # Horizontal line at min loss
min_loss_idx = losses.index(min(losses))
min_lr = lrs[min_loss_idx]
plt.axvline(min_lr, ls='--', color='r')  # Vertical line at min loss LR

# Find steepest point (where loss decreases most rapidly)
smoothed_losses = np.array(losses)
gradients = np.gradient(smoothed_losses)
steepest_idx = np.argmin(gradients)
steepest_lr = lrs[steepest_idx]
plt.axvline(steepest_lr, ls='--', color='g')  # Vertical line at steepest point

plt.legend([f'Min Loss LR: {min_lr:.1e}', f'Steepest LR: {steepest_lr:.1e}'])
plt.savefig('lr_finder_denoiser_result.png')
print(f"Results saved to lr_finder_denoiser_result.png")
print(f"Min Loss LR: {min_lr:.1e}")
print(f"Steepest LR: {steepest_lr:.1e}")
print(f"Suggested LR range: {steepest_lr:.1e} to {min_lr:.1e}") 