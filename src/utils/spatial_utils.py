import numpy as np
from config.model_config import (
    GRID_SIZE_DEFAULTS,
    NORMALIZATION,
    SPAWN_BOUNDARIES
)

def calculate_grid_size(num_items, difficulty):
    """
    Calculate optimal grid size based on number of items and difficulty
    
    Args:
        num_items (int): Total number of items to place
        difficulty (str): Difficulty level ('Normal', 'Hard', 'Super Hard')
    
    Returns:
        tuple: (rows, columns) for the grid
    """
    # Get base grid size from defaults
    base_config = GRID_SIZE_DEFAULTS[difficulty]
    row = base_config['base_rows']
    col = base_config['base_cols']
    
    # Scale up for more goals
    if num_items > 35:
        row = min(NORMALIZATION['grid']['row_max'], row + 1)
        col = min(NORMALIZATION['grid']['col_max'], col + 1)
    elif num_items < 20:
        row = max(NORMALIZATION['grid']['row_min'], row - 1)
        col = max(NORMALIZATION['grid']['col_min'], col - 1)
    
    # Add some randomness while respecting bounds
    row = min(NORMALIZATION['grid']['row_max'], 
              max(NORMALIZATION['grid']['row_min'], 
                  row + np.random.randint(-1, 2)))
    col = min(NORMALIZATION['grid']['col_max'], 
              max(NORMALIZATION['grid']['col_min'], 
                  col + np.random.randint(-1, 2)))
    
    return row, col

def assign_grid_positions(num_items, difficulty, row=None, col=None):
    """
    Assign grid-based positions for items
    
    Args:
        num_items (int): Total number of items to place
        difficulty (str): Difficulty level
        row (int, optional): Number of rows in grid
        col (int, optional): Number of columns in grid
    
    Returns:
        np.array: Array of normalized positions [x, z, row, col]
    """
    if row is None or col is None:
        row, col = calculate_grid_size(num_items, difficulty)
    
    # Calculate spacing
    x_min, _, z_min = SPAWN_BOUNDARIES['min']
    x_max, _, z_max = SPAWN_BOUNDARIES['max']
    x_spacing = (x_max - x_min) / row
    z_spacing = (z_max - z_min) / col
    
    positions = []
    for i in range(num_items):
        x = i % row
        z = i // row
        
        # Calculate position with spacing and random offset
        pos_x = x_min + (x + 0.5) * x_spacing + np.random.uniform(-0.1, 0.1) * x_spacing
        pos_z = z_min + (z + 0.5) * z_spacing + np.random.uniform(-0.1, 0.1) * z_spacing
        
        # Normalize positions to [0,1] range for VAE
        norm_x = (pos_x - x_min) / (x_max - x_min)
        norm_z = (pos_z - z_min) / (z_max - z_min)
        norm_row = row / NORMALIZATION['grid']['row_max']
        norm_col = col / NORMALIZATION['grid']['col_max']
        
        positions.append([norm_x, norm_z, norm_row, norm_col])
    
    return np.array(positions)

def denormalize_positions(positions):
    """
    Convert normalized positions back to Unity coordinates
    
    Args:
        positions (np.array): Array of normalized positions [x, z, row, col]
    
    Returns:
        np.array: Array of Unity coordinates [x, y, z]
    """
    x_min, y, z_min = SPAWN_BOUNDARIES['min']
    x_max, _, z_max = SPAWN_BOUNDARIES['max']
    
    unity_positions = []
    for pos in positions:
        x = pos[0] * (x_max - x_min) + x_min
        z = pos[1] * (z_max - z_min) + z_min
        unity_positions.append([x, y, z])
    
    return np.array(unity_positions)

def validate_grid_parameters(row, col, num_items):
    """
    Validate grid parameters
    
    Args:
        row (int): Number of rows
        col (int): Number of columns
        num_items (int): Total number of items
    
    Returns:
        bool: True if parameters are valid
    """
    if row < NORMALIZATION['grid']['row_min'] or row > NORMALIZATION['grid']['row_max']:
        return False
    if col < NORMALIZATION['grid']['col_min'] or col > NORMALIZATION['grid']['col_max']:
        return False
    if row * col < num_items:
        return False
    return True 