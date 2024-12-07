import json
import numpy as np
import os
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"sample_data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Object categories and properties
OBJECT_CATEGORIES = {
    'fruits': [
        'apple_red', 'banana_yellow', 'cherry_red',
        'grape_green', 'grape_purple', 'pear_green'
    ],
    'school': [
        'pencil', 'ruler', 'eraser', 'crayon',
        'calculator', 'notebook', 'textbook'
    ],
    'sports': [
        'baseball', 'basketball', 'tennis_ball',
        'volleyball', 'soccer_ball'
    ]
}

# Shape categories
SHAPES = {
    'sphere': {'scale_range': (0.8, 1.2)},
    'cube': {'scale_range': (0.9, 1.1)},
    'cylinder': {'scale_range': (0.7, 1.3)},
    'flat': {'scale_range': (1.2, 1.8)},
    'curved': {'scale_range': (0.6, 1.4)},
    'sharp': {'scale_range': (0.5, 1.5)},
    'mix': {'scale_range': (0.7, 1.3)}
}

def get_object_type():
    """Get random object type and its properties"""
    # Select random category
    category = np.random.choice(list(OBJECT_CATEGORIES.keys()))
    # Select random object from category
    obj_type = np.random.choice(OBJECT_CATEGORIES[category])
    # Get shape based on object type
    if 'ball' in obj_type or 'apple' in obj_type:
        shape = 'sphere'
    elif 'book' in obj_type or 'ruler' in obj_type:
        shape = 'flat'
    elif 'pencil' in obj_type or 'crayon' in obj_type:
        shape = 'cylinder'
    else:
        shape = np.random.choice(list(SHAPES.keys()))
    
    return obj_type, shape

def generate_random_object(difficulty, layer_idx=0):
    """Generate a random object with properties based on difficulty"""
    obj_type, shape = get_object_type()
    
    # Base size affected by difficulty
    scale_range = SHAPES[shape]['scale_range']
    base_size = float(np.random.uniform(*scale_range) * (1 + difficulty * 0.5))
    
    # Position with more spread for higher difficulty
    spread = float(5 + difficulty * 3)
    position = {
        'x': float(np.random.uniform(-spread, spread)),
        'y': float(2.0 * layer_idx + np.random.uniform(0, 1)),
        'z': float(np.random.uniform(-spread, spread))
    }
    
    # Scale based on shape
    if shape == 'sphere':
        scale = {'x': base_size, 'y': base_size, 'z': base_size}
    elif shape == 'cylinder':
        scale = {
            'x': float(base_size * 0.7),
            'y': float(base_size * 1.4),
            'z': float(base_size * 0.7)
        }
    elif shape == 'flat':
        scale = {
            'x': float(base_size * 1.2),
            'y': float(base_size * 0.5),
            'z': float(base_size * 1.2)
        }
    else:
        scale_variation = float(0.3 + difficulty * 0.2)
        scale = {
            'x': float(base_size * np.random.uniform(1 - scale_variation, 1 + scale_variation)),
            'y': float(base_size * np.random.uniform(1 - scale_variation, 1 + scale_variation)),
            'z': float(base_size * np.random.uniform(1 - scale_variation, 1 + scale_variation))
        }
    
    # Rotation with more variation for higher difficulty
    rot_range = float(45 + difficulty * 45)
    rotation = {
        'x': float(np.random.uniform(-rot_range, rot_range) if layer_idx > 0 else 0),
        'y': float(np.random.uniform(0, 360)),
        'z': float(np.random.uniform(-rot_range, rot_range) if layer_idx > 0 else 0)
    }
    
    # Special flags more likely with higher difficulty
    flags = {
        't': bool(np.random.random() < (0.2 + difficulty * 0.3)),
        'a': bool(np.random.random() < (0.1 + difficulty * 0.2)),
        'r': bool(np.random.random() < (0.3 + difficulty * 0.4))
    }
    
    return {
        'type': obj_type,
        'position': position,
        'scale': scale,
        'rotation': rotation,
        't': flags['t'],
        'a': flags['a'],
        'r': flags['r']
    }

def generate_level(difficulty):
    """Generate a single level with given difficulty"""
    # Number of objects per layer based on difficulty
    objects_per_layer = int(np.random.uniform(2, 4) * (1 + difficulty))
    num_layers = 3
    
    # Time limit decreases with difficulty
    time_limit = int(300 - difficulty * 120 + np.random.uniform(-30, 30))
    
    # Generate objects layer by layer
    objects = []
    for layer_idx in range(num_layers):
        layer_objects = [
            generate_random_object(difficulty, layer_idx)
            for _ in range(objects_per_layer)
        ]
        objects.extend(layer_objects)
    
    return {
        'metadata': {
            'difficulty': float(difficulty),
            'time_limit': int(time_limit)
        },
        'objects': objects
    }

def generate_dataset(num_levels, output_path):
    """Generate a dataset of random levels"""
    logger.info(f"Generating {num_levels} sample levels...")
    
    levels = []
    
    # Generate levels with varying difficulty
    difficulties = np.linspace(0.2, 0.8, num_levels)  # Range from easy to hard
    np.random.shuffle(difficulties)  # Randomize order
    
    for i, difficulty in enumerate(difficulties):
        if i % 100 == 0:
            logger.info(f"Generating level {i}/{num_levels}")
        
        level = generate_level(float(difficulty))
        levels.append(level)
    
    # Save to file
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(levels, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Generated {len(levels)} levels and saved to {output_path}")
    
    # Log some statistics
    total_objects = sum(len(level['objects']) for level in levels)
    avg_objects = total_objects / len(levels)
    avg_time = float(np.mean([level['metadata']['time_limit'] for level in levels]))
    
    logger.info(f"Dataset statistics:")
    logger.info(f"- Average objects per level: {avg_objects:.2f}")
    logger.info(f"- Average time limit: {avg_time:.2f}")
    logger.info(f"- Difficulty range: {float(min(difficulties)):.2f} to {float(max(difficulties)):.2f}")

def main():
    """Generate sample dataset"""
    output_path = 'data/raw_levels.json'
    num_levels = 1000  # Generate 1000 sample levels
    
    logger.info("Starting sample data generation")
    generate_dataset(num_levels, output_path)
    logger.info("Sample data generation completed")

if __name__ == '__main__':
    main() 