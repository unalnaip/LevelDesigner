# Model architecture parameters
MODEL_CONFIG = {
    'latent_dim': 32,
    'hidden_dims': [256, 128],
    'embedding_dim': 32,
    'min_grid_size': 4,
    'max_grid_rows': 6,
    'max_grid_cols': 7,
    'spatial_features': 4,  # x, z, row, col
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'beta': 1.0,  # KL loss weight
    'spatial_loss_weight': 1.0,  # Weight for spatial features loss
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