import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
import shutil

class EarlyStopping:
    """Early stopping handler with validation monitoring"""
    def __init__(self, patience=10, min_delta=0.001, metric_window=5):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_window = metric_window
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.val_losses = []
        
    def __call__(self, val_loss):
        # Add current validation loss
        self.val_losses.append(val_loss)
        
        # Calculate smoothed loss over window
        if len(self.val_losses) >= self.metric_window:
            smoothed_loss = np.mean(self.val_losses[-self.metric_window:])
            
            # Check if improvement is significant
            if smoothed_loss < self.best_loss - self.min_delta:
                self.best_loss = smoothed_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        
        return self.early_stop

class TrainingMonitor:
    """
    Enhanced training monitor with validation tracking and early stopping
    """
    def __init__(self, log_dir='logs', model_name='enhanced_cvae'):
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.log_dir / f"{model_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        
        # Initialize metric history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'beta': [],
            'spatial_metrics': [],
            'latent_stats': [],
            'gradient_norms': []
        }
        
        # Training start time
        self.start_time = time.time()
        
        # Create metric plots directory
        self.plot_dir = self.run_dir / 'plots'
        self.plot_dir.mkdir(exist_ok=True)
        
        print(f"Monitoring training at: {self.run_dir}")
        
        # Initialize best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Initialize step counter
        self.current_step = 0
    
    def log_metrics(self, metrics, phase='train'):
        """Log training or validation metrics"""
        # Increment step counter
        self.current_step += 1
        
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{key}', value, self.current_step)
        
        # Update history
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        # Log elapsed time
        elapsed = time.time() - self.start_time
        self.writer.add_scalar('time/elapsed_hours', elapsed / 3600, self.current_step)
        
        # Save history
        self._save_history()
        
        # Update best model if validation loss improved
        if phase == 'val' and metrics.get('total_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = metrics['total_loss']
            self.best_model_path = self.run_dir / f'best_model_step_{self.current_step}.pt'
            return True
        return False
    
    def log_attention_weights(self, name, weights):
        """Log attention weights"""
        if weights is not None:
            self.writer.add_histogram(
                f'attention/{name}',
                weights.flatten(),
                self.current_step
            )
    
    def save_metrics(self, filepath):
        """Save metrics to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _save_history(self):
        """Save training history to a JSON file"""
        history_path = self.run_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def close(self):
        """Clean up and save final results"""
        # Save final history
        self._save_history()
        
        # Close tensorboard writer
        self.writer.close()
        
        # Log training duration
        duration = time.time() - self.start_time
        print(f"\nTraining completed in {duration/3600:.2f} hours")
        print(f"Results saved to: {self.run_dir}")
        if self.best_model_path:
            print(f"Best model saved to: {self.best_model_path}")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
 