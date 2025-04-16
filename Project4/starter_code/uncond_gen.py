import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR  # Import better scheduler
plt.switch_backend("agg")

from models import MLP
from data import States

plot_dir = "plots/unconditional_generation"
os.makedirs(plot_dir, exist_ok=True)

# training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_steps = 1000
batch_size = 128
num_epochs = 200
lr = 3.0e-03  # Min Loss LR from test results
max_lr = 7.1e-02  # Steepest LR from test results
weight_decay = 0.01  # Proper weight decay for AdamW
refresh_epochs = 100 # Refresh data every 100 epochs as suggested
use_amp = True # Enable Automatic Mixed Precision
accumulation_steps = 4  # Can adjust based on memory
effective_batch_size = batch_size * accumulation_steps
print(f"Using effective batch size of {effective_batch_size} (batch_size={batch_size}, accumulation_steps={accumulation_steps})")
print(f"Using OneCycleLR with min_lr={lr}, max_lr={max_lr}")

dataset = States(num_steps=num_steps)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4 if device == "cuda" else 0, pin_memory=True if device == "cuda" else False)
dataset.show(save_to=os.path.join(plot_dir, "original_data.png"))

# Denoiser model: MLP with 4 hidden layers, 256 units each.
# Input: noisy sample xt (2D) + normalized timestep t_bar (1D) = 3D
# Output: predicted noise epsilon_hat (2D)
hidden_layers_denoiser = [256, 256, 256, 256]
mlp = MLP(input_dim=3, output_dim=2, hidden_layers=hidden_layers_denoiser) # create the denoiser model
mlp.to(device)
mse_loss = nn.MSELoss() # create the denoising (MSE) loss function

# Create your optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)

# Create a learning rate scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,  # Peak at steepest LR
    total_steps=num_epochs,  # One step per epoch
    pct_start=0.3,  # Spend 30% of training warming up
    div_factor=1,  # Start with min_lr (don't divide)
    final_div_factor=10,  # End with min_lr/10
    anneal_strategy='cos'  # Cosine annealing
)

# Create AMP GradScaler
scaler = GradScaler(enabled=use_amp)

# Precompute necessary values from dataset for sampling
beta_t = dataset.beta.to(device)
alpha_t = dataset.alpha.to(device)
alpha_bar_t = dataset.alpha_bar.to(device)

# Create checkpoint directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "denoiser_checkpoint.pt")

# Check if checkpoint exists and load it
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    mlp.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_loss_list = checkpoint['train_loss_list']
    nll_list = checkpoint['nll_list']
    print(f"Resuming from epoch {start_epoch}")
else:
    train_loss_list = []
    nll_list = []
    print("No checkpoint found, starting from scratch")

def train_one_epoch(current_dataloader):
    mlp.train()
    avg_loss = 0
    num_batches = 0
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    pbar = tqdm(current_dataloader, leave=False, desc="Training Epoch")
    for batch_idx, batch in enumerate(pbar):
        x_, t, eps, x, y = batch 
        x_ = x_.to(device)
        t = t.to(device)
        eps = eps.to(device)

        # Normalize timestep
        t_normalized = (2 * (t.float() - num_steps / 2) / num_steps).unsqueeze(1)
        inp = torch.cat([x_, t_normalized], dim=1)

        # Forward pass: predict noise with AMP
        with autocast(enabled=use_amp):
            eps_hat = mlp(inp)
            loss = mse_loss(eps_hat, eps)
            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps

        # Backward pass and optimization with AMP and gradient accumulation
        if use_amp:
            scaler.scale(loss).backward()
            
            # Step every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(current_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            
            # Step every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(current_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        avg_loss += loss.item() * accumulation_steps  # Multiply back to get actual loss
        num_batches += 1
        pbar.set_postfix(loss=loss.item() * accumulation_steps)

    avg_loss /= num_batches
    return avg_loss

@torch.no_grad()
def sample(num_samples=2000):
    mlp.eval()
    # Start with random noise N(0, I)
    z = torch.randn(num_samples, 2, device=device) # start with noise xT

    for i in trange(num_steps - 1, -1, -1, leave=False, desc="Sampling"):
        t_idx = torch.tensor([i] * num_samples, device=device) # Timestep index t
        
        # Normalize timestep t: t_bar = 2*(t - T/2)/T
        t_normalized = (2 * (t_idx.float() - num_steps / 2) / num_steps).unsqueeze(1)
        
        # Concatenate current sample z (xt) and normalized timestep
        inp = torch.cat([z, t_normalized], dim=1)
        
        # Predict noise using the MLP
        eps_hat = mlp(inp)
        
        # Sample noise zt for the sampling step
        zt = torch.randn_like(z)
        if i == 0:
            zt = torch.zeros_like(z) # No noise added at the last step
        
        # Retrieve precomputed alpha, alpha_bar, beta for timestep i
        alpha_ti = alpha_t[i]
        alpha_bar_ti = alpha_bar_t[i]
        beta_ti = beta_t[i]
        
        # Apply the reverse diffusion step formula from Ho et al. / Project PDF:
        # xt-1 = (1/sqrt(alpha_t)) * (xt - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta(xt, t)) + sqrt(beta_t) * zt
        term1 = 1.0 / torch.sqrt(alpha_ti)
        term2 = (beta_ti / torch.sqrt(1.0 - alpha_bar_ti)) * eps_hat
        term3 = torch.sqrt(beta_ti) * zt
        
        z = term1 * (z - term2) + term3 # z becomes x_{t-1}
    
    z = z.cpu().numpy()
    nll = dataset.calc_nll(z)

    return nll, z

print("Starting unconditional generator training...")
for e in trange(start_epoch, num_epochs):
    epoch_loss = train_one_epoch(dataloader)
    train_loss_list.append(epoch_loss)
    
    # Sample and calculate NLL periodically
    if (e + 1) % 100 == 0 or e == num_epochs - 1:
        nll, z = sample(2000) # Use 2000 for intermediate checks
        nll_list.append(nll)
        dataset.show(z, os.path.join(plot_dir, f"epoch_{e+1}.png"))
        print(f"Epoch {e+1}/{num_epochs}, Loss: {epoch_loss:.4f}, NLL: {nll:.4f}")
    else:
        nll_list.append(nll_list[-1] if nll_list else 0) # Placeholder NLL
        if (e + 1) % 10 == 0: # Print loss more frequently
             print(f"Epoch {e+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    dataset.show(z, os.path.join(plot_dir, f"latest.png")) # Show latest sample
    
    # Save checkpoint every 10 epochs
    if (e + 1) % 10 == 0:
        print(f"Saving checkpoint at epoch {e+1}")
        torch.save({
            'epoch': e,
            'model_state_dict': mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss_list': train_loss_list,
            'nll_list': nll_list,
        }, checkpoint_path)
    
    # Update scheduler once per epoch
    if (e + 1) % 1 == 0:
        scheduler.step()
        print(f"Epoch {(e + 1)}, Current LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Refresh data periodically
    if (e + 1) % refresh_epochs == 0:
        print(f"Refreshing dataset noise at epoch {e+1}")
        dataset.mix_data()
        # Recreate dataloader as dataset content changed
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4 if device == "cuda" else 0, pin_memory=True if device == "cuda" else False)

print("Training finished.")

# Final sampling with 5000 samples
print("Generating final 5000 samples...")
# If memory is an issue, sample in batches
final_samples = []
num_final_samples = 5000
batch_sample_size = 1000 # Adjust if needed
num_sample_batches = (num_final_samples + batch_sample_size - 1) // batch_sample_size
for _ in range(num_sample_batches):
    nll_batch, z_batch = sample(batch_sample_size)
    final_samples.append(z_batch)

final_z = np.concatenate(final_samples, axis=0)[:num_final_samples] # Ensure exact count
final_nll = dataset.calc_nll(final_z)
print(f"Final NLL on {num_final_samples} samples: {final_nll:.4f}")

dataset.show(final_z, os.path.join(plot_dir, "final.png"))
np.save(os.path.join(plot_dir, "uncond_gen_samples.npy"), final_z) # Save as .npy

# Save the final model state_dict for inference
torch.save(mlp.state_dict(), "denoiser.pt")

# Also save the final checkpoint (complete training state)
torch.save({
    'epoch': num_epochs - 1,
    'model_state_dict': mlp.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss_list': train_loss_list,
    'nll_list': nll_list,
}, checkpoint_path)
print(f"Final checkpoint saved to {checkpoint_path}")

print(f"Saved final samples to uncond_gen_samples.npy and model to denoiser.pt")

print("Plotting training logs...")
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(train_loss_list)
axs[0].set_title("Training Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].set_yscale("log")

axs[1].plot(nll_list)
axs[1].set_title("Negative Log Likelihood")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("NLL")
axs[1].set_yscale("log")

fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "train_logs.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)
