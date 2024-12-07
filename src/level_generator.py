import torch
import json
import numpy as np
from models.cvae import ChainedVAE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LevelGenerator:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Initialize model
        self.model = ChainedVAE(
            input_dim=30,  # 10 objects per layer * 3 features per object
            condition_dim=3  # difficulty, time, object count
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def decode_layer_to_objects(self, layer_data):
        """Convert layer tensor to list of objects"""
        objects = []
        
        # Process each object in the layer (3 features per object)
        for i in range(0, len(layer_data), 3):
            obj_type = int(layer_data[i])
            size = float(layer_data[i + 1])
            shape = int(layer_data[i + 2])
            
            # Skip if all features are 0 (padding)
            if obj_type == 0 and size == 0 and shape == 0:
                continue
            
            # Create object with properties
            obj = {
                'type': obj_type,
                'scale': self.get_scale_from_shape(shape, size),
                'position': {'x': 0, 'y': 0, 'z': 0},  # Will be set later
                'rotation': {'x': 0, 'y': 0, 'z': 0}
            }
            
            objects.append(obj)
        
        return objects
    
    def get_scale_from_shape(self, shape, base_size):
        """Convert shape category and size to actual scale values"""
        if shape == 0:  # Cube
            return {'x': base_size, 'y': base_size, 'z': base_size}
        elif shape == 1:  # Vertical
            return {'x': base_size * 0.7, 'y': base_size * 1.4, 'z': base_size * 0.7}
        elif shape == 2:  # Horizontal
            return {'x': base_size * 1.4, 'y': base_size * 0.7, 'z': base_size * 0.7}
        elif shape == 3:  # Deep
            return {'x': base_size * 0.7, 'y': base_size * 0.7, 'z': base_size * 1.4}
        elif shape == 4:  # Flat
            return {'x': base_size * 1.2, 'y': base_size * 0.5, 'z': base_size * 1.2}
        elif shape == 5:  # Long
            return {'x': base_size * 1.5, 'y': base_size * 0.6, 'z': base_size * 0.6}
        else:  # Irregular
            return {'x': base_size * 0.8, 'y': base_size * 1.1, 'z': base_size * 0.9}
    
    def position_objects(self, objects, layer_idx, total_layers):
        """Position objects in a layer with physics considerations"""
        if not objects:
            return
        
        # Calculate base Y position for this layer
        base_y = 2.0 * layer_idx  # 2 units between layers
        
        # Create grid for object placement
        grid_size = int(np.ceil(np.sqrt(len(objects))))
        grid_spacing = 2.0  # Units between objects
        
        # Place objects in grid
        for i, obj in enumerate(objects):
            row = i // grid_size
            col = i % grid_size
            
            # Calculate position with slight randomization
            x = (col - grid_size/2) * grid_spacing + np.random.uniform(-0.2, 0.2)
            z = (row - grid_size/2) * grid_spacing + np.random.uniform(-0.2, 0.2)
            y = base_y + obj['scale']['y']/2  # Place on ground
            
            obj['position'] = {'x': x, 'y': y, 'z': z}
            
            # Add slight random rotation for variety
            if layer_idx == 0:  # Keep base layer stable
                obj['rotation'] = {'x': 0, 'y': np.random.uniform(0, 360), 'z': 0}
            else:
                obj['rotation'] = {
                    'x': np.random.uniform(-5, 5),
                    'y': np.random.uniform(0, 360),
                    'z': np.random.uniform(-5, 5)
                }
    
    def generate_level(self, difficulty, time_limit, num_objects):
        """Generate a complete level with the given parameters"""
        # Normalize conditions
        condition = torch.tensor([
            difficulty,  # Assumed to be already normalized 0-1
            time_limit / 300.0,  # Normalize time
            num_objects / 20.0  # Normalize object count
        ], device=self.device).unsqueeze(0)
        
        # Generate level using model
        with torch.no_grad():
            layers = self.model.generate_level(condition)
        
        # Process each layer
        all_objects = []
        for i, layer_tensor in enumerate(layers):
            # Convert layer tensor to objects
            layer_objects = self.decode_layer_to_objects(layer_tensor.cpu().numpy().flatten())
            
            # Position objects in layer
            self.position_objects(layer_objects, i, len(layers))
            
            all_objects.extend(layer_objects)
        
        # Create level data
        level_data = {
            'metadata': {
                'difficulty': difficulty,
                'time_limit': int(time_limit),
                'object_count': len(all_objects)
            },
            'objects': all_objects
        }
        
        return level_data
    
    def generate_batch(self, num_levels, difficulty_range=(0.3, 0.7),
                      time_range=(120, 300), object_range=(10, 20)):
        """Generate multiple levels with varying parameters"""
        levels = []
        
        for _ in range(num_levels):
            # Sample random parameters
            difficulty = np.random.uniform(*difficulty_range)
            time_limit = np.random.randint(*time_range)
            num_objects = np.random.randint(*object_range)
            
            # Generate level
            level = self.generate_level(difficulty, time_limit, num_objects)
            levels.append(level)
        
        return levels

def main():
    # Initialize generator
    generator = LevelGenerator('models/level_generator.pt')
    
    # Generate a batch of levels
    levels = generator.generate_batch(
        num_levels=10,
        difficulty_range=(0.3, 0.7),
        time_range=(120, 300),
        object_range=(10, 20)
    )
    
    # Save generated levels
    with open('data/generated_levels.json', 'w') as f:
        json.dump(levels, f, indent=2)
    
    logger.info(f"Generated {len(levels)} levels and saved to data/generated_levels.json")

if __name__ == '__main__':
    main() 