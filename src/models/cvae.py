import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple, Optional

class CVAE(pl.LightningModule):
    def __init__(self, 
                 input_dim: int = 2,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 condition_dim: int = 32,
                 num_layers: int = 3,
                 learning_rate: float = 1e-4):
        super().__init__()
        
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim + condition_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim * 2
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim + condition_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Tanh()
            ])
            in_dim = out_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(3, condition_dim // 2),  # difficulty, time_limit, object_count
            nn.ReLU(),
            nn.Linear(condition_dim // 2, condition_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input and condition to latent space"""
        c_encoded = self.condition_encoder(c)
        x_c = torch.cat([x, c_encoded], dim=-1)
        h = self.encoder(x_c)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar
    
    def decode(self, z: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latent vector and condition to output space"""
        if c is not None:
            c_encoded = self.condition_encoder(c)
            z_c = torch.cat([z, c_encoded], dim=-1)
        else:
            # For unconditional generation, use neutral conditions
            batch_size = z.size(0)
            c_encoded = torch.zeros(batch_size, self.condition_dim).to(z.device)
            z_c = torch.cat([z, c_encoded], dim=-1)
        
        return self.decoder(z_c)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, c = batch['positions'], batch['conditions']
        
        # Forward pass
        recon, mu, logvar = self(x, c)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + 0.1 * kl_loss
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('recon_loss', recon_loss)
        self.log('kl_loss', kl_loss)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        x, c = batch['positions'], batch['conditions']
        recon, mu, logvar = self(x, c)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + 0.1 * kl_loss
        
        # Log metrics
        self.log('val_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> 'CVAE':
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = cls(**checkpoint['hyper_parameters'])
        model.load_state_dict(checkpoint['state_dict'])
        return model