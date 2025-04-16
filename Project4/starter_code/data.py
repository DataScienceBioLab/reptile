from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm # Import tqdm
plt.switch_backend("agg") # this is to avoid a Matplotlib issue.

class States(Dataset):
    def __init__(self, dataset_root=None, csv_path=None, num_steps=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load states.pt using torch.load. It contains the raw data
        # under the key "data" and the US state labels under the key
        # "labels". Load the raw data to self.data and the labels to
        # self.labels.
        states_data = torch.load("states.pt") # Load your data here
        
        # Check if we need to reduce memory usage
        # Store original data on CPU by default, only move to GPU when needed
        # This prevents OOM errors when data loading
        self.use_cuda_tensors = self.device.type == "cuda"
        store_device = self.device if self.use_cuda_tensors else torch.device("cpu")
        
        self.data = states_data["data"].to(store_device) # Load your actual 2D data here
        self.labels = states_data["labels"].to(store_device) # Load your labels here
        n_points = self.data.shape[0]
        self.n_points = n_points
        self.num_steps = num_steps
        # The diffusion process adds noise over T steps. Index t goes from 0 to T-1.
        # self.steps = torch.linspace(0, num_steps - 1, num_steps, device=self.device) # Create the steps using linspace between -1 and 1 - Not needed directly, use indices
        self.beta = torch.linspace(1e-4, 0.02, num_steps, device=store_device) # Create beta according to the schedule in PDF
        self.alpha = (1.0 - self.beta).to(store_device) # Compute alpha from beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(store_device) # Compute alpha_bar from alpha
        
        # Initialize cache pins for efficient CPU->GPU transfer if needed
        self._pin_memory = self.device.type == "cuda" and not self.use_cuda_tensors
        
        self.mix_data()
    
    def refresh_eps(self):
        total_samples = len(self)
        data_dim = self.data.shape[1]
        
        # Generate random noise on CPU to save CUDA memory
        # Only move to GPU if we're using CUDA tensors
        device = self.device if self.use_cuda_tensors else torch.device("cpu")
        self.eps = torch.randn(total_samples, data_dim, device=device) # Get a fresh set of epsilons

    def mix_data(self):
        # Clear previous data if it exists to free memory
        if hasattr(self, 'all_data'):
            del self.all_data
            del self.all_labels
            del self.all_times
            del self.all_steps
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        self.all_data = []
        self.all_labels = []
        self.all_times = []
        self.refresh_eps()
        total_samples = len(self)
        
        # Determine what device to use for storage
        store_device = self.device if self.use_cuda_tensors else torch.device("cpu")

        print(f"Mixing data for {total_samples} samples...") # Add print statement
        # Process in batches to save memory
        batch_size = 10000  # Adjust based on available memory
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            
            for i in tqdm(range(batch_start, batch_end), desc=f"Mixing Data Batch {batch_start//batch_size + 1}/{(total_samples+batch_size-1)//batch_size}"):
                data_idx = i % self.n_points
                step = i // self.n_points
                x = self.data[data_idx]
                # t corresponds to the step index
                alpha_bar_t = self.alpha_bar[step]
                e = self.eps[i]
                # Using Eq (2): xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
                x_ = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1.0 - alpha_bar_t) * e # Create the noisy data from x, t, and e
                if self.labels is None:
                    y = 0
                else:
                    y = self.labels[data_idx]

                self.all_data.append(x_)
                self.all_times.append(step)
                self.all_labels.append(y)
            
            # Free some memory after each batch
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Stack tensors but store on the appropriate device to save memory
        self.all_data = torch.stack(self.all_data).to(store_device)
        self.all_labels = torch.tensor(self.all_labels).to(store_device)
        self.all_steps = torch.tensor(self.all_times).to(store_device)
        
        # Pin memory for efficient transfer if needed
        if self._pin_memory:
            if isinstance(self.all_data, torch.Tensor) and not self.all_data.is_pinned():
                self.all_data = self.all_data.pin_memory()
            if isinstance(self.all_labels, torch.Tensor) and not self.all_labels.is_pinned():
                self.all_labels = self.all_labels.pin_memory()
            if isinstance(self.all_steps, torch.Tensor) and not self.all_steps.is_pinned():
                self.all_steps = self.all_steps.pin_memory()

    def __len__(self):
        return self.n_points * self.num_steps

    def __getitem__(self, idx):
        # Option 1: Use precomputed data (faster)
        if hasattr(self, 'all_data') and len(self.all_data) > idx:
            # Return precomputed data, but move to CPU if stored on GPU
            # This avoids CUDA initialization errors in DataLoader workers
            if self.use_cuda_tensors:
                return (
                    self.all_data[idx].cpu(),
                    self.all_steps[idx].cpu(),
                    self.eps[idx].cpu() if hasattr(self, 'eps') and len(self.eps) > idx else torch.randn_like(self.all_data[idx]).cpu(),
                    self.data[idx % self.n_points].cpu(),
                    self.all_labels[idx].cpu()
                )
            else:
                # Already on CPU
                return (
                    self.all_data[idx],
                    self.all_steps[idx],
                    self.eps[idx] if hasattr(self, 'eps') and len(self.eps) > idx else torch.randn_like(self.all_data[idx]),
                    self.data[idx % self.n_points],
                    self.all_labels[idx]
                )
        
        # Option 2: Compute on-the-fly (backup, should not happen with precomputed data)
        data_idx = idx % self.n_points
        step = idx // self.n_points
        x = self.data[data_idx]
        # t corresponds to the step index
        alpha_bar_t = self.alpha_bar[step]
        eps = torch.randn_like(x)
        # Using Eq (2): xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        x_ = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1.0 - alpha_bar_t) * eps # create the noisy data from x, t, and eps
        if self.labels is None:
            y = 0
        else:
            y = self.labels[data_idx]
            
        # Always return CPU tensors for DataLoader
        if self.use_cuda_tensors:
            return x_.cpu(), torch.tensor(step).cpu(), eps.cpu(), x.cpu(), y.cpu()
        else:
            return x_, torch.tensor(step), eps, x, y

    def show(self, samples=None, save_to=None):
        if samples is None:
            samples = self.data

        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        plt.axis('equal')
        if save_to is not None:
            plt.savefig(save_to)
        plt.close()
        plt.clf()

    def calc_nll(self, generated):
        data_ = self.data.numpy()

        kde = gaussian_kde(data_.T)
        nll = -kde.logpdf(generated.T)

        return nll.mean()
