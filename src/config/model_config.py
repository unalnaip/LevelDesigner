import torch

# Model architecture parameters
MODEL_CONFIG = {
    'latent_dim': 32,
    'hidden_dims': [256, 128],
    'embedding_dim': 32,
    'attention_heads': 4,
    'dropout': 0.1,  # Added dropout for regularization
    'min_grid_size': 4,
    'max_grid_rows': 6,
    'max_grid_cols': 7,
    'spatial_features': 4,  # x, z, row, col
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 8,  # Reduced for stability
    'learning_rate': 0.0001,  # Reduced from 0.001
    'num_epochs': 100,
    'beta': 1.5,  # Increased for better KL balance
    'kl_anneal_rate': 0.005,  # Reduced for smoother annealing
    'gradient_clip': 0.5,  # Added gradient clipping
    'weight_decay': 1e-5,  # L2 regularization
    'spatial_loss_weight': 1.0,
    'early_stopping_patience': 10,
    'validation_interval': 5,
    'checkpoint_interval': 10,
    'num_workers': 0 if torch.backends.mps.is_available() else 4,  # Adjusted for MPS
}

# Validation parameters
VALIDATION_CONFIG = {
    'val_split': 0.2,
    'min_delta': 0.001,  # Minimum change for improvement
    'metric_window': 5,  # Window for smoothing validation metrics
}

# Generation parameters
GENERATION_CONFIG = {
    'num_samples': 50,
    'temperature': 1.0,
    'min_time_limit': 30,
    'max_time_limit': 300,
    'max_goals_per_type': 30,
}

# Normalization parameters
NORMALIZATION = {
    'time_limit': {
        'min': 30,
        'max': 300
    },
    'level_id': {
        'min': 0,
        'max': 250
    },
    'goals': {
        'min': 0,
        'max': 30
    },
    'grid': {
        'row_min': 4,
        'row_max': 6,
        'col_min': 4,
        'col_max': 7
    }
}

# Grid size defaults per difficulty
GRID_SIZE_DEFAULTS = {
    'Normal': {
        'base_rows': 5,
        'base_cols': 5
    },
    'Hard': {
        'base_rows': 5,
        'base_cols': 6
    },
    'Super Hard': {
        'base_rows': 5,
        'base_cols': 6
    }
}

# Spawn area boundaries
SPAWN_BOUNDARIES = {
    'min': [-5, 0, -5],
    'max': [5, 12, 5]
} 