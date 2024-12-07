from pathlib import Path

# Project root directory (going up one more level since we're in src/config)
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
    "r": "is_goal"      # true for goal, false for blocker
}

# Object type categories
OBJECT_CATEGORIES = {
    "fruits": [
        "apple_red",
        "banana_yellow",
        "cherry_red",
        "grape_green",
        "grape_black",
        "grape_purple",
        "pear_green",
        "pineapple_orange",
        "pumpkin_orange"
    ]
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
    5: "variant_max"    # Maximum complexity variant (found in superhard levels)
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