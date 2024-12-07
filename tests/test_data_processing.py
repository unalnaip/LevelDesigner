import pytest
from pathlib import Path
import pandas as pd
from src.utils.time_utils import parse_time_to_seconds
from src.data_processing.processor import DataProcessor

def test_parse_time_to_seconds():
    """Test time string parsing."""
    assert parse_time_to_seconds("00:01:30") == 90
    assert parse_time_to_seconds("01:30") == 90
    assert parse_time_to_seconds("02:00:00") == 7200
    
    with pytest.raises(ValueError):
        parse_time_to_seconds("invalid")

def test_difficulty_mapping():
    """Test difficulty encoding."""
    processor = DataProcessor()
    assert processor.difficulty_mapping[""] == 0  # Normal
    assert processor.difficulty_mapping["Hard"] == 1
    assert processor.difficulty_mapping["Super Hard"] == 2

@pytest.fixture
def sample_level_data():
    """Create sample level data for testing."""
    return pd.DataFrame({
        'Level': [1, 2],
        'Time': ['00:05:00', '00:03:00'],
        'Difficulty': ['Hard', ''],
        'Goal 1': [3, 6],
        'Goal 2': [3, 0],
        'Goal 3': [3, 0]
    })

def test_level_data_processing(sample_level_data, tmp_path):
    """Test level data processing."""
    # Save sample data to temp file
    sample_file = tmp_path / "test_levels.csv"
    sample_level_data.to_csv(sample_file, index=False)
    
    processor = DataProcessor()
    df = processor.load_level_data(sample_file)
    
    assert len(df) == 2
    assert df['time_seconds'].tolist() == [300, 180]
    assert df['difficulty_encoded'].tolist() == [1, 0]
    assert df['total_goals'].tolist() == [9, 6]
    assert df['unique_goals'].tolist() == [3, 1] 