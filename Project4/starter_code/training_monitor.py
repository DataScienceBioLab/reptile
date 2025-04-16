# training_monitor.py - Real-time visualization of training progress
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output
import json
from datetime import datetime

class TrainingMonitor:
    def __init__(self, checkpoint_paths, plot_dir="monitor_plots", update_interval=30):
        """
        Monitor training progress by watching checkpoints and updating visualizations.
        
        Args:
            checkpoint_paths: Dict of {name: path} for checkpoint files to monitor
            plot_dir: Directory to save visualization plots
            update_interval: How often to check for updates (seconds)
        """
        self.checkpoint_paths = checkpoint_paths
        self.plot_dir = plot_dir
        self.update_interval = update_interval
        self.running = True
        self.last_modified_times = {name: 0 for name in checkpoint_paths}
        self.history = {name: {} for name in checkpoint_paths}
        
        os.makedirs(plot_dir, exist_ok=True)
        
    def load_checkpoint_data(self):
        """Load data from checkpoints if they've been modified"""
        updates = False
        
        for name, path in self.checkpoint_paths.items():
            if not os.path.exists(path):
                continue
                
            mtime = os.path.getmtime(path)
            if mtime > self.last_modified_times[name]:
                try:
                    checkpoint = torch.load(path, map_location='cpu')
                    self.history[name]['epoch'] = checkpoint.get('epoch', 0)
                    
                    # Extract metric histories
                    if 'train_loss_list' in checkpoint:
                        self.history[name]['train_loss'] = checkpoint['train_loss_list']
                    if 'nll_list' in checkpoint: 
                        self.history[name]['nll'] = checkpoint['nll_list']
                    
                    self.last_modified_times[name] = mtime
                    updates = True
                    print(f"Loaded checkpoint for {name}: epoch {self.history[name]['epoch']}")
                except Exception as e:
                    print(f"Error loading checkpoint for {name}: {e}")
        
        return updates
    
    def plot_training_curves(self):
        """Plot training curves for all models"""
        plt.figure(figsize=(15, 8))
        plt.clf()
        
        # Create one plot per metric
        metrics = set()
        for name in self.history:
            metrics.update(set(self.history[name].keys()) - {'epoch'})
        
        n_metrics = len(metrics)
        if n_metrics == 0:
            return
            
        for i, metric in enumerate(metrics):
            plt.subplot(1, n_metrics, i+1)
            
            for name, data in self.history.items():
                if metric in data and len(data[metric]) > 0:
                    plt.plot(data[metric], label=f"{name} {metric}")
                    
            plt.title(metric.replace('_', ' ').title())
            plt.xlabel('Epoch')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)
            plt.legend()
            
            # Use log scale for loss curves
            if 'loss' in metric.lower() or 'nll' in metric.lower():
                plt.yscale('log')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plot_dir, f"training_curves_{timestamp}.png"))
        
        # Save the latest version with a fixed name for easy viewing
        plt.savefig(os.path.join(self.plot_dir, "latest_training_curves.png"))
        
    def check_latest_samples(self):
        """Look for and display the latest generated samples"""
        sample_paths = [
            "plots/unconditional_generation/latest.png",  # Unconditional samples
            *[f"plots/conditional_diffusion/label_{i}.png" for i in range(5)]  # Conditional samples
        ]
        
        plt.figure(figsize=(15, 8))
        plt.clf()
        
        found_samples = False
        for i, path in enumerate(sample_paths):
            if os.path.exists(path):
                found_samples = True
                img = plt.imread(path)
                plt.subplot(2, 3, i+1)
                plt.imshow(img)
                plt.title(os.path.basename(path).replace('.png', ''))
                plt.axis('off')
        
        if found_samples:
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.plot_dir, f"latest_samples_{timestamp}.png"))
            # Save with fixed name
            plt.savefig(os.path.join(self.plot_dir, "latest_samples.png"))
    
    def run(self):
        """Main monitoring loop"""
        print(f"Starting training monitor. Checking for updates every {self.update_interval} seconds.")
        print(f"Monitoring checkpoints: {list(self.checkpoint_paths.keys())}")
        print(f"Plots will be saved to: {self.plot_dir}")
        
        while self.running:
            updates = self.load_checkpoint_data()
            
            if updates:
                self.plot_training_curves()
                self.check_latest_samples()
                print(f"Updated visualizations at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            time.sleep(self.update_interval)
    
    def stop(self):
        """Stop the monitoring loop"""
        self.running = False

if __name__ == "__main__":
    # Define checkpoints to monitor
    checkpoint_paths = {
        "classifier": "checkpoints/classifier_checkpoint.pt",
        "denoiser": "checkpoints/denoiser_checkpoint.pt"
    }
    
    # Create and run monitor
    monitor = TrainingMonitor(
        checkpoint_paths=checkpoint_paths,
        update_interval=60  # Check every 60 seconds
    )
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
        monitor.stop() 