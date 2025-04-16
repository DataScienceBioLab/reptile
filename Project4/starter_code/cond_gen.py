import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")

from models import MLP
from data import States

plot_dir = "plots/conditional_diffusion"
os.makedirs(plot_dir, exist_ok=True)

# training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch_size = None # Not needed for sampling directly
# num_epochs = None # No training here
num_steps = 500 # Must match the training setting

# create the same classifier as in classifier.py and load the weights.
# Set it to eval mode.
hidden_layers_classifier = [100, 200, 500]
classifier = MLP(input_dim=3, output_dim=5, hidden_layers=hidden_layers_classifier).to(device)
classifier.load_state_dict(torch.load("classifier.pt", map_location=device))
classifier.eval()
logsoftmax = nn.LogSoftmax(dim=1).to(device) # create log-softmax

# create your denoiser model architecture and load the weights from
# uncond_gen.py
hidden_layers_denoiser = [256, 256, 256, 256]
mlp = MLP(input_dim=3, output_dim=2, hidden_layers=hidden_layers_denoiser).to(device)
mlp.load_state_dict(torch.load("denoiser.pt", map_location=device))
mlp.eval()

dataset = States(num_steps=num_steps)
dataset.show(save_to=os.path.join(plot_dir, "original_data.png"))

# Precompute necessary values from dataset for sampling
beta_t = dataset.beta.to(device)
alpha_t = dataset.alpha.to(device)
alpha_bar_t = dataset.alpha_bar.to(device)

def sample(label, num_samples=1000, guidance_strength=1.0):
    mlp.eval()
    classifier.eval()
    
    z = torch.randn(num_samples, 2, device=device) # start with random noise xT

    for i in trange(num_steps-1, -1, -1, leave=False, desc=f"Sampling Label {label}"):
        t_idx = torch.tensor([i] * num_samples, device=device) # Timestep index t
        # Normalize timestep t: t_bar = 2*(t - T/2)/T
        t_normalized = (2 * (t_idx.float() - num_steps / 2) / num_steps).unsqueeze(1)
        
        # --- Classifier Guidance --- 
        # Requires gradients for the input z (xt)
        z_with_grad = z.detach().requires_grad_(True) 
        inp_with_grad = torch.cat([z_with_grad, t_normalized], dim=1)
        
        logits = classifier(inp_with_grad)
        log_probs = logsoftmax(logits)
        selected_log_prob = log_probs[:, label].sum() # Sum over the batch for gradient calculation
        
        # Compute the gradient w.r.t. the input z
        cls_grad = torch.autograd.grad(selected_log_prob, z_with_grad)[0]
        
        # --- Denoising Step --- 
        # No gradients needed for the denoiser part itself during sampling
        with torch.no_grad():
            inp = torch.cat([z, t_normalized], dim=1) # Use detached z
            eps_theta = mlp(inp) # Unconditional noise prediction
            
            # Retrieve precomputed alpha, alpha_bar, beta for timestep i
            alpha_ti = alpha_t[i]
            alpha_bar_ti = alpha_bar_t[i]
            beta_ti = beta_t[i]
            sqrt_one_minus_alpha_bar_ti = torch.sqrt(1.0 - alpha_bar_ti)

            # Compute conditional noise prediction eps_hat
            # eps_hat = eps_theta - sqrt(1 - alpha_bar_t) * grad(log f_phi(y | xt))
            eps_hat = eps_theta - guidance_strength * sqrt_one_minus_alpha_bar_ti * cls_grad
            
            # Sample noise zt for the sampling step
            zt = torch.randn_like(z)
            if i == 0:
                zt = torch.zeros_like(z) # No noise added at the last step
            
            # Apply the reverse diffusion step formula using eps_hat:
            # xt-1 = (1/sqrt(alpha_t)) * (xt - (beta_t / sqrt(1 - alpha_bar_t)) * eps_hat) + sqrt(beta_t) * zt
            term1 = 1.0 / torch.sqrt(alpha_ti)
            term2 = (beta_ti / sqrt_one_minus_alpha_bar_ti) * eps_hat
            term3 = torch.sqrt(beta_ti) * zt
            
            z = term1 * (z - term2) + term3 # z becomes x_{t-1}
 
    z = z.detach().cpu().numpy()
    # NLL calculation is done outside on the full 5k samples
    # nll = dataset.calc_nll(z) 
    return z # Return samples, NLL calc outside

print("Starting conditional generation...")
num_samples_per_label = 5000
batch_sample_size = 1000 # Adjust based on GPU memory

for label in range(5):
    full_z = []
    num_sample_batches = (num_samples_per_label + batch_sample_size - 1) // batch_sample_size
    print(f"Generating {num_samples_per_label} samples for label {label}...")
    for i in range(num_sample_batches):
        print(f"  Batch {i+1}/{num_sample_batches}")
        z_batch = sample(label, num_samples=batch_sample_size)
        full_z.append(z_batch)
        
    full_z = np.concatenate(full_z, axis=0)[:num_samples_per_label] # Ensure exact count
    nll = dataset.calc_nll(full_z)
    print(f"Label {label}, NLL: {nll:.4f}")
    dataset.show(full_z, os.path.join(plot_dir, f"label_{label}.png"))
    np.save(os.path.join(plot_dir, f"cond_gen_samples_{label}.npy"), full_z) # Save as .npy

print("Conditional generation finished.")

