import torch
from torch.utils.data import Dataset
import numpy as np

class LevelDataset(Dataset):
    """
    Dataset class for level data
    """
    def __init__(self, level_data, difficulty_labels):
        """
        Args:
            level_data (np.array): Array of vectorized level representations
            difficulty_labels (np.array): Array of difficulty labels
        """
        self.level_data = torch.FloatTensor(level_data)
        self.difficulty_labels = torch.FloatTensor(difficulty_labels)
        
    def __len__(self):
        return len(self.level_data)
        
    def __getitem__(self, idx):
        return self.level_data[idx], self.difficulty_labels[idx] 