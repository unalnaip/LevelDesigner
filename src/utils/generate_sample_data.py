import numpy as np
from pathlib import Path
import json

def generate_sample_data(num_samples=100, max_objects=30, grid_size=(6, 7)):
    """
    Generate sample data for testing the architecture
    
    Args:
        num_samples (int): Number of samples to generate
        max_objects (int): Maximum number of objects per level
        grid_size (tuple): Size of the grid (rows, cols)
    """
    # Generate level data
    level_data = []
    difficulty_labels = []
    positions = []
    
    for _ in range(num_samples):
        # Random number of objects
        num_objects = np.random.randint(5, max_objects)
        
        # Generate object properties (type, size, shape)
        objects = []
        for _ in range(num_objects):
            obj_type = np.random.randint(0, 5)  # 5 different types
            size = np.random.uniform(0.5, 2.0)
            shape = np.random.randint(0, 7)  # 7 different shapes
            objects.extend([obj_type, size, shape])
        
        # Pad to max objects
        while len(objects) < max_objects * 3:
            objects.extend([0, 0, 0])
        
        # Generate conditions (difficulty, time_limit, object_count)
        difficulty = np.random.uniform(0, 1)
        time_limit = np.random.randint(30, 300) / 300.0  # Normalize to [0,1]
        obj_count = num_objects / float(max_objects)  # Normalize to [0,1]
        
        # Generate positions
        pos = []
        rows, cols = grid_size
        for i in range(num_objects):
            row = i // cols
            col = i % cols
            # Normalize to [-1, 1]
            norm_row = 2 * (row / (rows - 1)) - 1 if rows > 1 else 0
            norm_col = 2 * (col / (cols - 1)) - 1 if cols > 1 else 0
            pos.extend([norm_row, norm_col])
        
        # Pad positions
        while len(pos) < max_objects * 2:
            pos.extend([0, 0])
        
        level_data.append(objects)
        difficulty_labels.append([difficulty, time_limit, obj_count])
        positions.append(pos)
    
    # Convert to numpy arrays
    level_data = np.array(level_data, dtype=np.float32)
    difficulty_labels = np.array(difficulty_labels, dtype=np.float32)
    positions = np.array(positions, dtype=np.float32)
    
    return level_data, difficulty_labels, positions

def save_sample_data(output_dir='data/processed', filename='training_data.npz'):
    """
    Generate and save sample data
    
    Args:
        output_dir (str): Output directory
        filename (str): Output filename
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    level_data, difficulty_labels, positions = generate_sample_data()
    
    # Save as npz
    np.savez(
        output_dir / filename,
        level_data=level_data,
        difficulty_labels=difficulty_labels,
        positions=positions
    )
    
    print(f"Sample data saved to {output_dir / filename}")
    print(f"Generated {len(level_data)} samples")
    print(f"Level data shape: {level_data.shape}")
    print(f"Difficulty labels shape: {difficulty_labels.shape}")
    print(f"Positions shape: {positions.shape}")

if __name__ == "__main__":
    save_sample_data() 