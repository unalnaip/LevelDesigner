from pathlib import Path
from typing import Dict, List, Any

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data directories
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Input files
LEVEL_DATA_FILE = RAW_DATA_DIR / "levelsdesign.txt"
OBJECT_DATA_FILE = RAW_DATA_DIR / "objectlistbig.txt"

# Output files
PROCESSED_LEVELS_FILE = PROCESSED_DATA_DIR / "processed_levels.json"
DATASET_STATS_FILE = PROCESSED_DATA_DIR / "dataset_statistics.json"
SEQUENCE_DATA_FILE = PROCESSED_DATA_DIR / "sequence_data.json"

# Model input format configuration
SEQUENCE_CONFIG = {
    "max_objects_per_level": 20,  # Maximum number of objects in a sequence
    "object_feature_dim": 8,      # Dimension of object feature embeddings
    "position_encoding_dim": 4,   # Dimension of positional encodings
    "grid_size": (8, 8),         # Size of discretized grid for spatial positions
}

# Flexible conditioning configuration
CONDITION_CONFIG = {
    "numerical_features": {
        "time_limit": {
            "min": 30,
            "max": 300,
            "normalization": "minmax"
        },
        "total_objects": {
            "min": 2,
            "max": 20,
            "normalization": "minmax"
        }
    },
    "categorical_features": {
        "difficulty": {
            "values": ["none", "hard", "superhard"],
            "encoding": "onehot"
        },
        "level_parameter": {
            "values": list(range(6)),  # 0-5
            "encoding": "onehot"
        }
    },
    "spatial_features": {
        "grid_position": {
            "type": "discrete",
            "resolution": SEQUENCE_CONFIG["grid_size"]
        }
    }
}

# Object features configuration
OBJECT_FEATURES = {
    "categorical": [
        "object_type",
        "is_goal"
    ],
    "numerical": [
        "amount",
        "grid_x",
        "grid_y"
    ],
    "embeddings": {
        "object_type": 4,  # Embedding dimension for object types
        "position": 4      # Embedding dimension for positions
    }
}

# Unity prototype format
UNITY_LEVEL_FORMAT = {
    "i": "level_index",
    "d": "duration_seconds",
    "t": "level_type",
    "l": "objects"  # List of objects
}

UNITY_OBJECT_FORMAT = {
    "t": "object_type",  # e.g. "pumpkin_orange"
    "a": "amount",       # Required amount (2-9)
    "r": "is_goal",      # true for goal, false for blocker
    "p": {              # Optional position information
        "x": "float",   # Normalized x position (0-1)
        "y": "float"    # Normalized y position (0-1)
    }
}

# Object type categories with metadata
OBJECT_CATEGORIES = {
    "fruits": {
        "types": [
            "apple_red",
            "banana_yellow",
            "cherry_red",
            "grape_green",
            "grape_black",
            "grape_purple",
            "pear_green",
            "pineapple_orange",
            "pumpkin_orange"
        ],
        "properties": {
            "can_stack": False,
            "default_size": 1.0
        }
    }
}

# Data processing parameters
DIFFICULTY_MAPPING = {
    "none": 0,     # Normal difficulty
    "hard": 1,     # Hard difficulty
    "superhard": 2 # Super hard difficulty
}

# Level parameter mapping (column 3)
LEVEL_PARAMETER_MAPPING = {
    0: "normal",        # Standard level
    1: "variant_1",     # First variant
    2: "variant_2",     # Second variant
    3: "variant_3",     # Third variant
    4: "variant_super", # Super variant
    5: "variant_max"    # Maximum complexity variant
}

# Data structure parameters
MAX_GOALS_PER_LEVEL = 6
MIN_TIME_LIMIT = 30  # seconds
MAX_TIME_LIMIT = 300  # seconds

# Column indices for levelsdesign.txt
LEVEL_COLUMNS = {
    'level_id': 0,
    'time_seconds': 1,
    'level_parameter': 2,
    'difficulty': 3,
    'goals_start': 4,
    'goals_end': 9,  # Up to 6 goals (columns 4-9)
    'blockers_start': 9  # Blockers start after goals
}

# Additional markers in the data
MARKERS = {
    'f': 'section_separator',  # Separates different sections in the level data
    't': 'special_blocker'     # Might indicate special blocker types
}

# Positional encoding configuration
POSITIONAL_ENCODING = {
    "type": "sinusoidal",      # Type of positional encoding
    "max_sequence_length": 20,  # Maximum sequence length
    "encoding_dim": 4,         # Dimension of positional encoding
    "dropout": 0.1             # Dropout rate for positional encodings
}

# Model architecture parameters
MODEL_CONFIG = {
    "attention": {
        "num_heads": 4,
        "head_dim": 64,
        "dropout": 0.1
    },
    "transformer": {
        "num_layers": 3,
        "hidden_dim": 256,
        "feedforward_dim": 512
    },
    "condition_encoder": {
        "embedding_dim": 64,
        "num_layers": 2
    }
}