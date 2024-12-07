import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from config.model_config import TRAINING_CONFIG, MODEL_CONFIG

def compute_loss(recon_data, data, mu, log_var, spatial_features=MODEL_CONFIG['spatial_features']):
    """
    Compute VAE loss with separate terms for spatial and non-spatial features
    
    Args:
        recon_data (torch.Tensor): Reconstructed data
        data (torch.Tensor): Original data
        mu (torch.Tensor): Mean of latent distribution
        log_var (torch.Tensor): Log variance of latent distribution
        spatial_features (int): Number of spatial features
    """
    # Split features
    basic_features = data.shape[1] - spatial_features
    
    # Compute reconstruction loss for basic features
    recon_loss_basic = F.binary_cross_entropy(
        recon_data[:, :basic_features],
        data[:, :basic_features],
        reduction='sum'
    )
    
    # Compute reconstruction loss for spatial features
    recon_loss_spatial = F.binary_cross_entropy(
        recon_data[:, -spatial_features:],
        data[:, -spatial_features:],
        reduction='sum'
    )
    
    # Weight the spatial loss
    recon_loss = recon_loss_basic + TRAINING_CONFIG['spatial_loss_weight'] * recon_loss_spatial
    
    # KL divergence loss with numerical stability
    kl_loss = -0.5 * torch.sum(1 + torch.clamp(log_var, min=-10, max=10) - mu.pow(2) - log_var.exp())
    
    # Total loss with beta weighting
    total_loss = recon_loss + TRAINING_CONFIG['beta'] * kl_loss
    
    return total_loss, recon_loss, kl_loss

def train_cvae(model, train_loader, val_loader, num_epochs=TRAINING_CONFIG['num_epochs'], 
               learning_rate=TRAINING_CONFIG['learning_rate'], device='cpu'):
    """
    Train the Conditional VAE model
    
    Args:
        model (ConditionalVAE): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        device (str): Device to train on ('cpu' or 'cuda')
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Create directory for model checkpoints
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        # Training loop with progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, condition) in enumerate(progress_bar):
            try:
                data = data.float().to(device)
                condition = condition.float().to(device)
                
                # Ensure data is normalized between 0 and 1
                data = torch.clamp(data, 0, 1)
                
                optimizer.zero_grad()
                
                # Forward pass
                recon_data, mu, log_var = model(data, condition)
                
                # Compute loss
                loss, recon_loss, kl_loss = compute_loss(recon_data, data, mu, log_var)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item()/len(data):.4f}',
                    'recon': f'{recon_loss.item()/len(data):.4f}',
                    'kl': f'{kl_loss.item()/len(data):.4f}'
                })
                
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        with torch.no_grad():
            for data, condition in val_loader:
                try:
                    data = data.float().to(device)
                    condition = condition.float().to(device)
                    
                    # Ensure data is normalized between 0 and 1
                    data = torch.clamp(data, 0, 1)
                    
                    recon_data, mu, log_var = model(data, condition)
                    loss, recon_loss, kl_loss = compute_loss(recon_data, data, mu, log_var)
                    
                    val_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    
                except RuntimeError as e:
                    print(f"Error in validation: {str(e)}")
                    continue
        
        # Average losses
        train_loss /= len(train_loader.dataset)
        train_recon_loss /= len(train_loader.dataset)
        train_kl_loss /= len(train_loader.dataset)
        
        val_loss /= len(val_loader.dataset)
        val_recon_loss /= len(val_loader.dataset)
        val_kl_loss /= len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f})')
        print(f'Val Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f})')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')
        
        # Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_dir / "cvae_model.pt")
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    print("Training completed!") 