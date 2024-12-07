import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from models.cvae import ChainedVAE
import logging
import os
from datetime import datetime
import psutil
import platform
import time
from utils.training_monitor import TrainingMonitor
from tqdm import tqdm

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LevelDataset(Dataset):
    def __init__(self, data_path):
        logger.info(f"Initializing dataset from {data_path}")
        start_time = time.time()
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} levels from JSON in {time.time() - start_time:.2f}s")
        
        # Process data into layers
        self.processed_data = []
        self.conditions = []
        
        logger.info("Processing levels into layers...")
        progress_bar = tqdm(enumerate(self.data), total=len(self.data), desc="Processing levels")
        for idx, level in progress_bar:
            # Sort objects by y-position to determine layers
            objects = sorted(level['objects'], key=lambda x: x['position']['y'])
            
            # Group into 3 layers
            num_objects = len(objects)
            layer_size = max(1, num_objects // 3)
            
            layers = [
                objects[i:i + layer_size]
                for i in range(0, num_objects, layer_size)
            ]
            
            # Pad or truncate to exactly 3 layers
            while len(layers) < 3:
                layers.append([])
            if len(layers) > 3:
                layers = layers[:3]
            
            # Convert each layer to fixed-size feature vector
            layer_vectors = []
            for layer in layers:
                layer_vec = []
                for obj in layer:
                    obj_type = obj.get('type', 0)
                    size = max(obj['scale']['x'], obj['scale']['y'], obj['scale']['z'])
                    shape = self.determine_shape(obj['scale'])
                    layer_vec.extend([obj_type, size, shape])
                
                # Pad layer vector
                while len(layer_vec) < 30:
                    layer_vec.extend([0, 0, 0])
                
                layer_vectors.append(layer_vec)
            
            # Combine layers
            level_vector = []
            for vec in layer_vectors:
                level_vector.extend(vec)
            
            # Create condition vector
            condition = [
                level['difficulty'],
                level['time_limit'] / 300.0,
                len(objects) / 20.0
            ]
            
            self.processed_data.append(level_vector)
            self.conditions.append(condition)
            
            # Update progress description
            if idx % 100 == 0:
                progress_bar.set_description(f"Processing levels ({idx}/{len(self.data)})")
        
        logger.info(f"Dataset initialization completed in {time.time() - start_time:.2f}s")
        logger.info(f"Total samples: {len(self.processed_data)}")
    
    def determine_shape(self, scale):
        """Determine object shape based on scale ratios"""
        x, y, z = scale['x'], scale['y'], scale['z']
        total = x + y + z
        if total == 0:
            return 0
        
        x, y, z = x/total, y/total, z/total
        
        if abs(x - y) < 0.1 and abs(y - z) < 0.1:
            return 0  # Cube
        elif y > max(x, z) * 2:
            return 1  # Vertical
        elif x > max(y, z) * 2:
            return 2  # Horizontal
        elif z > max(x, y) * 2:
            return 3  # Deep
        elif abs(x - z) < 0.1 and y < min(x, z):
            return 4  # Flat
        elif max(x, y, z) / min(x, y, z) > 3:
            return 5  # Long
        else:
            return 6  # Irregular
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.processed_data[idx]),
            torch.FloatTensor(self.conditions[idx])
        )

def train_model(data_path, output_path, num_epochs=100, batch_size=64):
    # Log system information
    device = log_system_info()
    
    # Initialize training monitor
    monitor = TrainingMonitor()
    
    # Initialize dataset and dataloader
    logger.info("Initializing dataset and dataloader...")
    dataset = LevelDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # Optimize for M3 Max
        pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    input_dim = 30
    condition_dim = 3
    model = ChainedVAE(input_dim, condition_dim).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0
            
            # Log system metrics at start of epoch
            system_metrics = monitor.log_system_metrics()
            logger.info(f"System metrics: {system_metrics}")
            
            model.train()
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                              desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (batch_data, batch_conditions) in progress_bar:
                batch_data = batch_data.to(device)
                batch_conditions = batch_conditions.to(device)
                
                # Forward pass
                output, loss = model(batch_data, batch_conditions)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/num_batches:.4f}"
                })
                
                # Log batch metrics
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    monitor.log_batch(epoch, batch_idx, loss.item(), current_lr, len(dataloader))
            
            # Calculate epoch metrics
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            memory_used = psutil.Process().memory_info().rss / (1024**3)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log epoch metrics
            monitor.log_epoch(epoch, avg_loss, current_lr, epoch_time, memory_used)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Average Loss: {avg_loss:.4f}")
            logger.info(f"Epoch Time: {epoch_time:.2f}s")
            logger.info(f"Memory Usage: {memory_used:.2f} GB")
            
            # Save checkpoint if best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss
                }, checkpoint_path)
                logger.info(f"Saved best model checkpoint to {checkpoint_path}")
            
            # Regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Save final model and clean up
        torch.save(model.state_dict(), output_path)
        monitor.close()
        
        # Log training summary
        total_time = time.time() - start_time
        logger.info("\nTraining Summary:")
        logger.info(f"Total training time: {total_time/3600:.2f} hours")
        logger.info(f"Best loss achieved: {best_loss:.4f}")
        logger.info(f"Final model saved to: {output_path}")
        logger.info(f"Training logs saved to: {log_file}")

def log_system_info():
    """Log system information for debugging"""
    logger.info("System Information:")
    logger.info(f"OS: {platform.system()} {platform.version()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # Check if MPS is available (Apple Silicon)
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS is available")
        device = torch.device("mps")
    else:
        logger.info("MPS not available, using CPU")
        device = torch.device("cpu")
    
    return device

if __name__ == '__main__':
    try:
        train_model('data/processed_levels.json', 'models/level_generator.pt')
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise 