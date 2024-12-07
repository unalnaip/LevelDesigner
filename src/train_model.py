import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import json

from models.cvae import EnhancedCVAE
from utils.level_dataset import LevelDataset
from utils.training_monitor import TrainingMonitor
from utils.generate_sample_data import save_sample_data

def setup_logger():
    """Configure logging for training"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.setup_directories()
        
        # Initialize dataset and model
        self.setup_data()
        self.setup_model()
        
        # Initialize training monitor
        self.monitor = TrainingMonitor(
            log_dir=str(self.run_dir),
            model_name='enhanced_cvae'
        )
        
        # Initialize early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def setup_directories(self):
        """Create necessary directories for outputs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("runs") / timestamp
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.sample_dir = self.run_dir / "samples"
        
        self.run_dir.mkdir(parents=True)
        self.checkpoint_dir.mkdir()
        self.sample_dir.mkdir()
        
        # Save configuration
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
            
    def setup_data(self):
        """Initialize datasets and dataloaders"""
        # Load dataset
        self.dataset = LevelDataset(
            data_path=self.config['data']['path'],
            grid_size=self.config['data']['grid_size'],
            max_objects=self.config['data']['max_objects']
        )
        
        # Split dataset
        val_size = int(len(self.dataset) * self.config['training']['val_split'])
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, 
            [train_size, val_size]
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Dataset loaded - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
    def setup_model(self):
        """Initialize model, optimizer, and scheduler"""
        # Create model
        self.model = EnhancedCVAE(
            input_dim=self.dataset.get_input_dim(),
            condition_dim=self.dataset.get_condition_dim(),
            latent_dim=self.config['model']['latent_dim'],
            hidden_dims=self.config['model']['hidden_dims'],
            attention_heads=self.config['model']['attention_heads'],
            dropout=self.config['model']['dropout'],
            beta_start=self.config['model']['beta_start'],
            beta_end=self.config['model']['beta_end'],
            beta_steps=self.config['model']['beta_steps']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']
    
    def generate_samples(self, epoch, num_samples=5):
        """Generate and save sample levels"""
        self.model.eval()
        with torch.no_grad():
            # Generate samples with varying conditions
            conditions = []
            for _ in range(num_samples):
                # Create different types of conditions
                conditions.append([
                    np.random.uniform(0, 1),  # Difficulty
                    np.random.uniform(30, 300) / 300,  # Time limit
                    np.random.uniform(5, 30) / 30  # Object count
                ])
            conditions = torch.FloatTensor(conditions).to(self.device)
            
            # Generate samples one by one to avoid batch size issues
            all_properties = []
            all_positions = []
            
            for i in range(num_samples):
                z = torch.randn(1, self.config['model']['latent_dim']).to(self.device)
                properties, positions = self.model.decode(z, conditions[i:i+1])
                
                all_properties.append(properties.cpu().numpy())
                all_positions.append(positions.cpu().numpy())
            
            # Save samples
            samples = {
                'epoch': epoch,
                'conditions': conditions.cpu().numpy().tolist(),
                'properties': np.concatenate(all_properties, axis=0).tolist(),
                'positions': np.concatenate(all_positions, axis=0).tolist()
            }
            
            sample_path = self.sample_dir / f"samples_epoch_{epoch}.json"
            with open(sample_path, 'w') as f:
                json.dump(samples, f, indent=2)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (level_data, conditions, spatial_features) in enumerate(pbar):
            # Move data to device
            level_data = level_data.to(self.device)
            conditions = conditions.to(self.device)
            spatial_features = spatial_features.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(level_data, conditions, spatial_features=spatial_features)
            
            # Calculate loss
            recon_loss = output['recon_loss']
            kl_loss = output['kl_loss']
            beta = output['beta']
            
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'kl': f"{kl_loss.item():.4f}",
                'β': f"{beta:.4f}"
            })
            
            # Log batch metrics
            self.monitor.log_metrics({
                'total_loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'beta': beta
            })
            
            # Log attention weights periodically
            if batch_idx % 100 == 0:
                attention_weights = self.model.get_attention_weights()
                if attention_weights['encoder'] is not None:
                    self.monitor.log_attention_weights('encoder', attention_weights['encoder'])
                if attention_weights['decoder'] is not None:
                    self.monitor.log_attention_weights('decoder', attention_weights['decoder'])
        
        # Calculate epoch metrics
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def validate(self, epoch):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for level_data, conditions, spatial_features in self.val_loader:
                # Move data to device
                level_data = level_data.to(self.device)
                conditions = conditions.to(self.device)
                spatial_features = spatial_features.to(self.device)
                
                # Forward pass
                output = self.model(level_data, conditions, spatial_features=spatial_features)
                
                # Calculate loss
                recon_loss = output['recon_loss']
                kl_loss = output['kl_loss']
                beta = output['beta']
                
                loss = recon_loss + beta * kl_loss
                
                # Update metrics
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        # Calculate validation metrics
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        # Log validation metrics
        self.monitor.log_metrics({
            'val_total_loss': avg_loss,
            'val_recon_loss': avg_recon_loss,
            'val_kl_loss': avg_kl_loss
        }, phase='val')
        
        return avg_loss
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training")
        
        for epoch in range(self.config['training']['num_epochs']):
            # Training phase
            train_loss, train_recon, train_kl = self.train_epoch(epoch)
            logger.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f} "
                f"(Recon: {train_recon:.4f}, KL: {train_kl:.4f})"
            )
            
            # Validation phase
            val_loss = self.validate(epoch)
            logger.info(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch)
            
            # Generate samples
            if (epoch + 1) % self.config['training']['sample_interval'] == 0:
                self.generate_samples(epoch)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['training']['patience']:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        logger.info("Training completed")
        self.monitor.close()

def main():
    # Configuration
    config = {
        'data': {
            'path': 'data/processed/training_data.npz',
            'grid_size': (6, 7),
            'max_objects': 30
        },
        'model': {
            'latent_dim': 32,
            'hidden_dims': [256, 128],
            'attention_heads': 4,
            'dropout': 0.1,
            'beta_start': 0.0,
            'beta_end': 1.5,
            'beta_steps': 2000
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'val_split': 0.1,
            'patience': 15,
            'checkpoint_interval': 5,
            'sample_interval': 10,
            'num_workers': 0  # Set to 0 for MPS
        }
    }
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()