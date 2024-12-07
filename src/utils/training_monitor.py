import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import psutil
import logging
from torch.utils.tensorboard import SummaryWriter
import json

logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, log_dir="logs", model_name="level_generator"):
        self.log_dir = log_dir
        self.model_name = model_name
        self.start_time = datetime.now()
        
        # Create tensorboard writer
        self.writer = SummaryWriter(
            os.path.join(log_dir, "tensorboard", model_name, self.start_time.strftime("%Y%m%d_%H%M%S"))
        )
        
        # Initialize metrics
        self.losses = []
        self.epoch_times = []
        self.memory_usage = []
        self.learning_rates = []
        
        # Create plots directory
        self.plots_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        logger.info(f"Training monitor initialized at {self.log_dir}")
    
    def log_epoch(self, epoch, loss, lr, epoch_time, memory_used):
        """Log metrics for current epoch"""
        # Store metrics
        self.losses.append(loss)
        self.epoch_times.append(epoch_time)
        self.memory_usage.append(memory_used)
        self.learning_rates.append(lr)
        
        # Log to tensorboard
        self.writer.add_scalar("Loss/train", loss, epoch)
        self.writer.add_scalar("Time/epoch", epoch_time, epoch)
        self.writer.add_scalar("System/memory_gb", memory_used, epoch)
        self.writer.add_scalar("Training/learning_rate", lr, epoch)
        
        # Generate plots every 10 epochs
        if (epoch + 1) % 10 == 0:
            self.generate_plots(epoch + 1)
    
    def log_batch(self, epoch, batch_idx, loss, lr, num_batches):
        """Log metrics for current batch"""
        step = epoch * num_batches + batch_idx
        self.writer.add_scalar("Loss/batch", loss, step)
        self.writer.add_scalar("Training/learning_rate_batch", lr, step)
    
    def generate_plots(self, epoch):
        """Generate and save training visualization plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        ax1.plot(self.losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        
        # Plot epoch times
        ax2.plot(self.epoch_times)
        ax2.set_title("Epoch Times")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Time (s)")
        ax2.grid(True)
        
        # Plot memory usage
        ax3.plot(self.memory_usage)
        ax3.set_title("Memory Usage")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Memory (GB)")
        ax3.grid(True)
        
        # Plot learning rate
        ax4.plot(self.learning_rates)
        ax4.set_title("Learning Rate")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate")
        ax4.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f"training_plots_epoch_{epoch}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training plots saved to {plot_path}")
    
    def log_system_metrics(self):
        """Log current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3)
        }
        
        # Log to tensorboard
        for name, value in metrics.items():
            self.writer.add_scalar(f"System/{name}", value, len(self.losses))
        
        return metrics
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            "losses": self.losses,
            "epoch_times": self.epoch_times,
            "memory_usage": self.memory_usage,
            "learning_rates": self.learning_rates,
            "training_duration": str(datetime.now() - self.start_time)
        }
        
        history_path = os.path.join(
            self.log_dir,
            f"training_history_{self.model_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def close(self):
        """Clean up and save final metrics"""
        self.save_training_history()
        self.generate_plots(len(self.losses))
        self.writer.close()
        logger.info("Training monitor closed") 