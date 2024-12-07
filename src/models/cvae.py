import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimplePhysicsValidator:
    """Simple physics validation based on object properties"""
    def __init__(self):
        # Basic stability rules based on shape combinations
        self.stability_matrix = {
            # shape_above vs shape_below
            (0, 0): 0.9,  # Similar shapes stack well
            (1, 1): 0.8,
            (2, 2): 0.9,  # Flat shapes are stable
            (3, 3): 0.7,  # Round shapes less stable
            (4, 4): 0.6,
            (5, 5): 0.7,
            (6, 6): 0.5   # Complex shapes least stable
        }
        
        # Default stability for non-matching shapes
        self.default_stability = 0.4
    
    def can_stack(self, upper_obj, lower_obj):
        """Check if objects can be stacked"""
        # Basic size rule: can't stack bigger on smaller
        if upper_obj['size'] > lower_obj['size']:
            return False, 0.0
            
        # Get stability score
        shape_pair = (upper_obj['shape'], lower_obj['shape'])
        stability = self.stability_matrix.get(shape_pair, self.default_stability)
        
        # Size difference bonus
        size_diff = lower_obj['size'] - upper_obj['size']
        stability += min(0.2, size_diff * 0.1)  # Small bonus for better size ratios
        
        return True, min(1.0, stability)

class LayerVAE(nn.Module):
    """VAE for generating a single layer of objects"""
    def __init__(self, input_dim, condition_dim, latent_dim=32):
        super().__init__()
        
        # Dimensions
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, c):
        combined = torch.cat([x, c], dim=1)
        hidden = self.encoder(combined)
        return self.fc_mu(hidden), self.fc_var(hidden)
    
    def decode(self, z, c):
        combined = torch.cat([z, c], dim=1)
        return self.decoder(combined)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, c), mu, log_var

class ChainedVAE(nn.Module):
    """
    Chained VAE implementation for layer-by-layer level generation
    with physics validation and designer intervention support
    """
    def __init__(self, input_dim, condition_dim, num_layers=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.num_layers = num_layers
        
        # Create VAEs for each layer
        self.layer_vaes = nn.ModuleList([
            LayerVAE(input_dim, condition_dim + (input_dim if i > 0 else 0))
            for i in range(num_layers)
        ])
        
        # Physics validator
        self.physics = SimplePhysicsValidator()
        
        # Layer-specific condition encoders
        self.layer_conditions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(condition_dim, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, condition_dim)
            ) for _ in range(num_layers)
        ])
    
    def generate_layer(self, condition, previous_layer=None, layer_idx=0, manual_input=None):
        """
        Generate a single layer with optional manual input
        
        Args:
            condition: Base condition (difficulty, etc.)
            previous_layer: Output from previous layer if any
            layer_idx: Which layer we're generating
            manual_input: Designer-provided input to influence generation
        """
        # Encode layer-specific condition
        layer_condition = self.layer_conditions[layer_idx](condition)
        
        # Combine with previous layer if exists
        if previous_layer is not None:
            layer_condition = torch.cat([layer_condition, previous_layer], dim=1)
        
        # Use manual input if provided
        if manual_input is not None:
            # Blend manual input with generated
            vae = self.layer_vaes[layer_idx]
            with torch.no_grad():
                z = torch.randn(1, vae.latent_dim).to(condition.device)
                generated = vae.decode(z, layer_condition)
                # Mix manual and generated (70-30 ratio)
                output = 0.7 * manual_input + 0.3 * generated
        else:
            # Generate from scratch
            vae = self.layer_vaes[layer_idx]
            with torch.no_grad():
                z = torch.randn(1, vae.latent_dim).to(condition.device)
                output = vae.decode(z, layer_condition)
        
        return output
    
    def validate_physics(self, current_layer, previous_layer=None):
        """Validate physics constraints between layers"""
        if previous_layer is None:
            return True, 1.0  # Base layer always valid
            
        # Extract object properties from layers
        current_objects = self.decode_layer_objects(current_layer)
        previous_objects = self.decode_layer_objects(previous_layer)
        
        # Check stacking validity
        valid_count = 0
        total_stability = 0.0
        
        for curr_obj in current_objects:
            obj_valid = False
            max_stability = 0.0
            
            for prev_obj in previous_objects:
                can_stack, stability = self.physics.can_stack(curr_obj, prev_obj)
                if can_stack:
                    obj_valid = True
                    max_stability = max(max_stability, stability)
            
            if obj_valid:
                valid_count += 1
                total_stability += max_stability
        
        # Layer is valid if most objects can be stacked
        validity_ratio = valid_count / len(current_objects)
        avg_stability = total_stability / len(current_objects) if current_objects else 0
        
        return validity_ratio > 0.7, avg_stability
    
    def decode_layer_objects(self, layer_data):
        """Convert layer tensor to list of object properties"""
        # This is a simplified version - in practice would need proper decoding
        objects = []
        # Assume layer_data contains [type, size, shape] for each object
        for i in range(0, layer_data.size(1), 3):
            obj = {
                'type': layer_data[0, i].item(),
                'size': layer_data[0, i+1].item(),
                'shape': layer_data[0, i+2].item()
            }
            objects.append(obj)
        return objects
    
    def generate_level(self, condition, manual_inputs=None):
        """
        Generate complete level layer by layer
        
        Args:
            condition: Base generation condition
            manual_inputs: List of manual inputs for each layer (optional)
        """
        layers = []
        previous_layer = None
        
        for i in range(self.num_layers):
            # Get manual input for this layer if provided
            manual_input = manual_inputs[i] if manual_inputs is not None else None
            
            # Generate layer
            current_layer = self.generate_layer(
                condition, previous_layer, i, manual_input
            )
            
            # Validate physics if not first layer
            if i > 0:
                valid, stability = self.validate_physics(current_layer, previous_layer)
                if not valid:
                    # Regenerate with adjusted condition to improve stability
                    adjusted_condition = condition * (1.0 + (1.0 - stability))
                    current_layer = self.generate_layer(
                        adjusted_condition, previous_layer, i, manual_input
                    )
            
            layers.append(current_layer)
            previous_layer = current_layer
        
        return layers
    
    def forward(self, x, condition):
        """Forward pass for training"""
        # Split input into layers
        layer_size = x.size(1) // self.num_layers
        layers = torch.split(x, layer_size, dim=1)
        
        # Process each layer
        outputs = []
        previous_layer = None
        total_loss = 0
        
        for i, layer in enumerate(layers):
            vae = self.layer_vaes[i]
            
            # Prepare condition
            layer_condition = self.layer_conditions[i](condition)
            if previous_layer is not None:
                layer_condition = torch.cat([layer_condition, previous_layer], dim=1)
            
            # VAE forward pass
            output, mu, log_var = vae(layer, layer_condition)
            
            # Accumulate loss
            recon_loss = F.mse_loss(output, layer)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            total_loss += recon_loss + 0.1 * kl_loss
            
            outputs.append(output)
            previous_layer = output
        
        return torch.cat(outputs, dim=1), total_loss