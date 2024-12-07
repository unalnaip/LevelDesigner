import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class LevelDataset(Dataset):
    """
    Dataset class for level data with enhanced spatial awareness and grid-based positioning
    """
    def __init__(self, data_path, grid_size=(6, 7), max_objects=30):
        """
        Initialize the dataset with spatial features
        
        Args:
            data_path (str): Path to the processed level data (.npz format)
            grid_size (tuple): Size of the grid (rows, cols)
            max_objects (int): Maximum number of objects per level
        """
        self.grid_size = grid_size
        self.max_objects = max_objects
        
        # Load data
        data = np.load(data_path)
        self.level_data = torch.FloatTensor(data['level_data'])
        self.conditions = torch.FloatTensor(data['difficulty_labels'])
        self.spatial_features = torch.FloatTensor(data['positions'])
        
        # Calculate dimensions
        self.num_objects = self.level_data.shape[1] // 3  # Divide by 3 for (type, size, shape)
        self.input_dim = self.level_data.shape[1]
        self.condition_dim = self.conditions.shape[1]
    
    def get_input_dim(self):
        """Get input dimension"""
        return self.input_dim
    
    def get_condition_dim(self):
        """Get condition dimension"""
        return self.condition_dim
    
    def get_grid_size(self):
        """Get grid size"""
        return self.grid_size
    
    def get_spatial_dim(self):
        """Get spatial feature dimension"""
        return 2  # x, y coordinates per object
        
    def __len__(self):
        return len(self.level_data)
        
    def __getitem__(self, idx):
        """
        Get a level with its conditions and spatial features
        
        Returns:
            tuple: (level_data, conditions, spatial_features)
        """
        return (
            self.level_data[idx],
            self.conditions[idx],
            self.spatial_features[idx]
        ) 