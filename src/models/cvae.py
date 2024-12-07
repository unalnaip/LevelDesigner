import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_encoder import ObjectEncoder, PhysicsGNN, SpatialAttention

class FlexibleConditionEncoder(nn.Module):
    """
    Flexible condition encoder that can handle varying condition dimensions
    and is easily extensible for new condition types
    """
    def __init__(self, condition_dim, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.layers = nn.Sequential(
            nn.Linear(condition_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, c):
        return self.layers(c)

class AttentionEncoder(nn.Module):
    """
    Attention-based encoder for capturing spatial relationships and object interactions
    """
    def __init__(self, input_dim, condition_dim, latent_dim=32, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Object embedding
        self.obj_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Condition embedding
        self.condition_encoder = FlexibleConditionEncoder(condition_dim, hidden_dim)
        
        # Multi-head attention for modeling interactions
        self.attention = SpatialAttention(hidden_dim, num_heads=num_heads)
        
        # Output layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, c):
        # Embed input and condition
        x_embed = self.obj_embedding(x)
        c_embed = self.condition_encoder(c)
        
        # Apply attention
        x_attended = self.attention(x_embed.unsqueeze(0)).squeeze(0)
        
        # Combine with condition
        combined = x_attended + c_embed
        
        # Output parameters
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar

class SpatialDecoder(nn.Module):
    """
    Decoder with spatial awareness and physics constraints
    """
    def __init__(self, latent_dim, condition_dim, output_dim, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Condition embedding
        self.condition_encoder = FlexibleConditionEncoder(condition_dim, hidden_dim)
        
        # Initial projection
        self.initial_projection = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Spatial attention for decoding
        self.attention = SpatialAttention(hidden_dim, num_heads=num_heads)
        
        # Separate decoders for properties and positions
        self.property_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.position_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim // 3 * 2),  # x,y coordinates for each object
            nn.Tanh()
        )
        
        # Physics-aware refinement
        self.physics_gnn = PhysicsGNN(node_dim=5)  # 3 properties + 2 coordinates
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z, c):
        # Get batch size from z
        batch_size = z.size(0) if len(z.shape) > 1 else 1
        
        # Reshape inputs if needed
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        if len(c.shape) == 1:
            c = c.unsqueeze(0)
            
        # Repeat condition if needed
        if c.size(0) == 1 and batch_size > 1:
            c = c.repeat(batch_size, 1)
        elif c.size(0) != batch_size:
            raise ValueError(f"Condition batch size ({c.size(0)}) does not match latent batch size ({batch_size})")
        
        # Embed condition
        c_embed = self.condition_encoder(c)
        
        # Combine latent and condition
        combined = torch.cat([z, c_embed], dim=1)
        hidden = self.initial_projection(combined)
        
        # Apply spatial attention
        hidden = self.attention(hidden.unsqueeze(0)).squeeze(0)
        
        # Decode properties and positions
        properties = self.property_decoder(hidden)
        positions = self.position_decoder(hidden)
        
        # Reshape for physics processing
        num_objects = properties.size(1) // 3
        properties_reshaped = properties.view(batch_size, num_objects, 3)
        positions_reshaped = positions.view(batch_size, num_objects, 2)
        
        # Combine features for physics processing
        combined_features = torch.cat([properties_reshaped, positions_reshaped], dim=-1)
        
        # Apply physics refinement
        refined_features = self.physics_gnn(combined_features)
        
        # Split refined features
        refined_properties = refined_features[..., :3].reshape(batch_size, -1)
        refined_positions = refined_features[..., 3:].reshape(batch_size, -1)
        
        return refined_properties, refined_positions

class EnhancedCVAE(nn.Module):
    """
    Enhanced Conditional VAE with spatial awareness, β-VAE properties,
    and flexible conditioning for puzzle game level generation
    """
    def __init__(
        self,
        input_dim,
        condition_dim,
        latent_dim=32,
        hidden_dims=[256, 128],
        attention_heads=4,
        dropout=0.1,
        beta_start=0.0,
        beta_end=1.5,
        beta_steps=1000
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.current_step = 0
        
        # Encoder
        self.encoder = AttentionEncoder(
            input_dim=input_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dims[0],
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = SpatialDecoder(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dims[0],
            num_heads=attention_heads,
            dropout=dropout
        )
        
    def encode(self, x, c):
        return self.encoder(x, c)
    
    def decode(self, z, c):
        return self.decoder(z, c)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x, c, spatial_features=None):
        # Encode
        mu, logvar = self.encode(x, c)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        properties, positions = self.decode(z, c)
        
        # Compute losses
        recon_loss = F.mse_loss(properties, x)
        if spatial_features is not None:
            recon_loss += F.mse_loss(positions, spatial_features)
            
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Get current beta value
        beta = self.get_beta()
        
        # Return losses separately for monitoring
        return {
            'properties': properties,
            'positions': positions,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': beta
        }
    
    def generate(self, c, num_samples=1):
        """Generate new levels given conditions"""
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim).to(c.device)
            
            # Decode
            properties, positions = self.decode(z, c)
            
            return properties, positions
    
    def get_beta(self):
        """Get current β value based on annealing schedule"""
        if not self.training:
            return self.beta_end
            
        self.current_step = min(self.current_step + 1, self.beta_steps)
        return self.beta_start + (self.beta_end - self.beta_start) * (self.current_step / self.beta_steps)
    
    def get_attention_weights(self):
        """Get attention weights from the encoder and decoder"""
        attention_weights = {
            'encoder': None,
            'decoder': None
        }
        
        # Get encoder attention weights
        if hasattr(self.encoder, 'attention'):
            attention_weights['encoder'] = self.encoder.attention.get_attention_weights()
            
        # Get decoder attention weights
        if hasattr(self.decoder, 'attention'):
            attention_weights['decoder'] = self.decoder.attention.get_attention_weights()
            
        return attention_weights