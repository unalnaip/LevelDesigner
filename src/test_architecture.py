import torch
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

from models.cvae import EnhancedCVAE
from utils.level_dataset import LevelDataset
from utils.training_monitor import TrainingMonitor

def setup_logger():
    """Configure logging for the test script"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_architecture_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()  # Create logger at module level

def test_training_loop(model, dataloader, optimizer, monitor, device, num_batches=10):
    """Run a short training loop to test the architecture"""
    model.train()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches)):
        if batch_idx >= num_batches:
            break
            
        # Move data to device
        level_data = batch[0].to(device)  # First element is level data
        conditions = batch[1].to(device)  # Second element is conditions
        spatial_features = batch[2].to(device)  # Third element is spatial features
        
        # Forward pass
        optimizer.zero_grad()
        
        output = model(
            level_data,
            conditions,
            spatial_features=spatial_features
        )
        
        # Calculate loss
        recon_loss = output['recon_loss']
        kl_loss = output['kl_loss']
        beta = output['beta']
        
        total_loss = recon_loss + beta * kl_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Log metrics
        monitor.log_metrics({
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'beta': beta
        })
        
        # Get attention weights if available
        attention_weights = model.get_attention_weights()
        if attention_weights['encoder'] is not None:
            monitor.log_attention_weights('encoder', attention_weights['encoder'])
        if attention_weights['decoder'] is not None:
            monitor.log_attention_weights('decoder', attention_weights['decoder'])
            
        # Print progress
        if batch_idx % 2 == 0:
            logger.info(
                f"Batch {batch_idx}: "
                f"Loss = {total_loss.item():.4f} "
                f"(Recon = {recon_loss.item():.4f}, "
                f"KL = {kl_loss.item():.4f}, "
                f"β = {beta:.4f})"
            )

def main():
    logger.info("Starting architecture test")
    
    # Configuration
    config = {
        'model': {
            'latent_dim': 32,
            'hidden_dims': [256, 128],
            'attention_heads': 4,
            'dropout': 0.1,
            'beta_start': 0.0,
            'beta_end': 1.5,
            'beta_steps': 1000
        },
        'training': {
            'batch_size': 4,  # Small batch size for testing
            'learning_rate': 0.0001,
            'num_test_batches': 10
        }
    }
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load a small subset of data
        dataset = LevelDataset(
            data_path='data/processed/training_data.npz',  # Updated path
            grid_size=(6, 7),
            max_objects=30
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        
        # Get dimensions from dataset
        input_dim = dataset.get_input_dim()
        condition_dim = dataset.get_condition_dim()
        
        logger.info(f"Dataset dimensions - Input: {input_dim}, Condition: {condition_dim}")
        
        # Initialize model
        model = EnhancedCVAE(
            input_dim=input_dim,
            condition_dim=condition_dim,
            latent_dim=config['model']['latent_dim'],
            hidden_dims=config['model']['hidden_dims'],
            attention_heads=config['model']['attention_heads'],
            dropout=config['model']['dropout'],
            beta_start=config['model']['beta_start'],
            beta_end=config['model']['beta_end'],
            beta_steps=config['model']['beta_steps']
        ).to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Initialize training monitor
        monitor = TrainingMonitor()
        
        # Run test training loop
        logger.info("Starting test training loop")
        test_training_loop(
            model,
            dataloader,
            optimizer,
            monitor,
            device,
            num_batches=config['training']['num_test_batches']
        )
        
        # Generate a test sample
        logger.info("Generating test sample")
        with torch.no_grad():
            test_conditions = torch.randn(1, condition_dim).to(device)  # Random conditions
            properties, positions = model.generate(test_conditions)
            
            logger.info(f"Generated properties shape: {properties.shape}")
            logger.info(f"Generated positions shape: {positions.shape}")
        
        # Save training metrics
        monitor.save_metrics('logs/test_metrics.json')
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    main()