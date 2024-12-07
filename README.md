# Level Generation System - Phase 0: Data Preparation

This phase focuses on cleaning, encoding, and formatting the existing level data and object lists into a structured dataset suitable for training models.

## Project Structure
```
leveldesigner/
├── data/                      # Data files
│   ├── raw/                  # Original data files
│   └── processed/            # Processed datasets
├── src/                      # Source code
│   ├── config/              # Configuration files
│   │   └── data_processing_config.py
│   ├── data_processing/     # Data processing modules
│   └── utils/               # Utility functions
├── tests/                    # Test files
├── setup.py                  # Package setup file
└── requirements.txt          # Dependencies
```

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Ensure data files are in the correct location:
```
data/raw/
├── Playliner_Match_Factory.xlsx - Level Parameteres (1).csv
└── objectlistbig.txt
```

## Data Processing

Run the data processing script:
```bash
python -m src.data_processing.processor
```

This will:
1. Load and clean the level and object data
2. Convert categorical variables to numeric encodings
3. Normalize continuous variables
4. Generate dataset statistics
5. Save processed data in JSON format

## Output

The script generates two files in `data/processed/`:

1. `processed_levels.json`: Contains the processed level data with:
   - Level metadata (difficulty, time limit, etc.)
   - Goal information
   - Encoded categorical variables

2. `dataset_statistics.json`: Contains summary statistics:
   - Total number of levels
   - Difficulty distribution
   - Time limit statistics
   - Goal count statistics

## Data Format

### Level Data Structure
```json
{
    "level_id": "integer",
    "metadata": {
        "difficulty": "string",
        "difficulty_encoded": "integer",
        "time_limit": "integer",
        "total_goals": "integer",
        "unique_goals": "integer"
    },
    "goals": {
        "goal_1": "integer",
        "goal_2": "integer",
        ...
    }
}
```

### Encodings
- Difficulty: Normal (0), Hard (1), Super Hard (2)
- Time: Converted to seconds
- Goals: Raw counts preserved

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
Follow PEP 8 guidelines and use type hints where possible.

## Next Steps

The processed dataset will be used in Phase 1 to train the conditional Variational Autoencoder (cVAE) for level generation. 