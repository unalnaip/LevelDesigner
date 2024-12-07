import torch
from pathlib import Path
import numpy as np
from models.cvae import ConditionalVAE
from utils.level_dataset import LevelDataset
from utils.data_preprocessing import load_and_preprocess_levels, normalize_level_data, save_preprocessed_data
from training.train_cvae import train_cvae
from torch.utils.data import DataLoader, random_split

def main():
    # Data paths
    raw_data_path = Path('data/raw/levelsdesign.txt')
    processed_data_dir = Path('data/processed')
    
    # Load and preprocess data if not already processed
    if not (processed_data_dir / 'level_data.npy').exists():
        print("Preprocessing raw level data...")
        level_data, difficulty_labels = load_and_preprocess_levels(raw_data_path)
        level_data = normalize_level_data(level_data)
        save_preprocessed_data(level_data, difficulty_labels, processed_data_dir)
    else:
        print("Loading preprocessed data...")
        level_data = np.load(processed_data_dir / 'level_data.npy')
        difficulty_labels = np.load(processed_data_dir / 'difficulty_labels.npy')
    
    # Create dataset
    dataset = LevelDataset(level_data, difficulty_labels)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model parameters
    input_dim = level_data.shape[1]  # Dimension of flattened level
    condition_dim = difficulty_labels.shape[1]  # Number of difficulty classes
    latent_dim = 32
    hidden_dims = [256, 128]
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    )
    
    # Training parameters
    training_params = {
        'num_epochs': 100,
        'learning_rate': 0.001,
        'device': device
    }
    
    # Train model
    print(f"Training model on {device}...")
    train_cvae(model, train_loader, val_loader, **training_params)
    
    print("Training complete!")

if __name__ == '__main__':
    main() 