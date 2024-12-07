import pandas as pd
import numpy as np
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

from src.config.data_processing_config import (
    DIFFICULTY_MAPPING, LEVEL_PARAMETER_MAPPING,
    LEVEL_DATA_FILE, OBJECT_DATA_FILE,
    PROCESSED_LEVELS_FILE, DATASET_STATS_FILE,
    LEVEL_COLUMNS, UNITY_LEVEL_FORMAT, UNITY_OBJECT_FORMAT,
    OBJECT_CATEGORIES
)
from src.utils.time_utils import parse_time_to_seconds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.difficulty_mapping = DIFFICULTY_MAPPING
        self.level_parameter_mapping = LEVEL_PARAMETER_MAPPING
        self.object_mapping = self._initialize_object_mapping()
        
    def _initialize_object_mapping(self) -> Dict[int, str]:
        """Initialize mapping between numeric object IDs and Unity object types."""
        object_mapping = {}
        current_id = 1
        
        # Map each fruit type to an ID
        for category, objects in OBJECT_CATEGORIES.items():
            for obj_type in objects:
                object_mapping[current_id] = obj_type
                current_id += 1
                
        return object_mapping
        
    def _map_object_to_unity_type(self, obj_id: int) -> str:
        """Map a numeric object ID to its corresponding Unity object type."""
        if obj_id not in self.object_mapping:
            raise ValueError(f"Unknown object ID: {obj_id}")
        return self.object_mapping[obj_id]
        
    def clean_level_id(self, level_id):
        """Clean level ID by extracting only the numeric part."""
        if isinstance(level_id, (int, float)):
            return int(level_id)
        match = re.search(r'\d+', str(level_id))
        if match:
            return int(match.group())
        raise ValueError(f"Could not extract level ID from: {level_id}")
        
    def parse_level_line(self, line: str) -> dict:
        """Parse a single line from levelsdesign.txt into a structured format."""
        parts = line.strip().split(',')
        
        # Extract basic level info
        level_id = int(parts[LEVEL_COLUMNS['level_id']])
        time_seconds = int(parts[LEVEL_COLUMNS['time_seconds']])
        level_parameter = int(parts[LEVEL_COLUMNS['level_parameter']])
        difficulty = parts[LEVEL_COLUMNS['difficulty']].lower()
        
        # Extract goals (non-empty values between goals_start and goals_end)
        goals = [
            int(g) for g in parts[LEVEL_COLUMNS['goals_start']:LEVEL_COLUMNS['goals_end']]
            if g and g.strip()
        ]
        
        # Extract blockers (values between goals_end and first 'f' marker)
        blocker_values = []
        for value in parts[LEVEL_COLUMNS['blockers_start']:]:
            if value == 'f':
                break
            if value and value.strip():
                blocker_values.append(int(value))
        
        return {
            'level_id': level_id,
            'metadata': {
                'difficulty': difficulty,
                'difficulty_encoded': self.difficulty_mapping[difficulty],
                'level_parameter': level_parameter,
                'level_variant': self.level_parameter_mapping[level_parameter],
                'time_limit': time_seconds,
                'total_goals': sum(goals),
                'unique_goals': len(goals),
                'total_blockers': sum(blocker_values),
                'unique_blockers': len(blocker_values)
            },
            'goals': {
                f'goal_{i+1}': count
                for i, count in enumerate(goals)
            },
            'blockers': {
                f'blocker_{i+1}': count
                for i, count in enumerate(blocker_values)
            }
        }
        
    def load_level_data(self, file_path: Path) -> list:
        """Load and preprocess level data from levelsdesign.txt."""
        logger.info(f"Loading level data from {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        processed_levels = []
        for line in lines:
            try:
                level_data = self.parse_level_line(line)
                processed_levels.append(level_data)
            except Exception as e:
                logger.warning(f"Error processing line: {line.strip()}\nError: {str(e)}")
                continue
                
        logger.info(f"Processed {len(processed_levels)} levels")
        return processed_levels

    def load_object_data(self, file_path: Path) -> pd.DataFrame:
        """Load and preprocess object data."""
        logger.info(f"Loading object data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Create category encodings
        self.category1_mapping = {cat: idx for idx, cat in enumerate(df['Category1'].unique())}
        self.category2_mapping = {cat: idx for idx, cat in enumerate(df['Category2'].unique())}
        self.color1_mapping = {color: idx for idx, color in enumerate(df['Color1'].unique())}
        self.shape_mapping = {shape: idx for idx, shape in enumerate(df['Shape'].unique())}
        self.size_mapping = {size: idx for idx, size in enumerate(df['Size'].unique())}
        
        logger.info(f"Processed {len(df)} objects")
        return df

    def generate_statistics(self, processed_data: list) -> dict:
        """Generate summary statistics of the dataset."""
        stats = {
            'total_levels': len(processed_data),
            'difficulty_distribution': {},
            'level_parameter_distribution': {},
            'time_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0
            },
            'goals_stats': {
                'min_total': float('inf'),
                'max_total': 0,
                'avg_total': 0,
                'min_unique': float('inf'),
                'max_unique': 0,
                'avg_unique': 0
            },
            'blockers_stats': {
                'min_total': float('inf'),
                'max_total': 0,
                'avg_total': 0,
                'min_unique': float('inf'),
                'max_unique': 0,
                'avg_unique': 0
            }
        }
        
        for level in processed_data:
            # Difficulty distribution
            diff = level['metadata']['difficulty']
            stats['difficulty_distribution'][diff] = stats['difficulty_distribution'].get(diff, 0) + 1
            
            # Level parameter distribution
            param = level['metadata']['level_variant']
            stats['level_parameter_distribution'][param] = stats['level_parameter_distribution'].get(param, 0) + 1
            
            # Time stats
            time_limit = level['metadata']['time_limit']
            stats['time_stats']['min'] = min(stats['time_stats']['min'], time_limit)
            stats['time_stats']['max'] = max(stats['time_stats']['max'], time_limit)
            stats['time_stats']['avg'] += time_limit
            
            # Goals stats
            total_goals = level['metadata']['total_goals']
            unique_goals = level['metadata']['unique_goals']
            
            stats['goals_stats']['min_total'] = min(stats['goals_stats']['min_total'], total_goals)
            stats['goals_stats']['max_total'] = max(stats['goals_stats']['max_total'], total_goals)
            stats['goals_stats']['avg_total'] += total_goals
            
            stats['goals_stats']['min_unique'] = min(stats['goals_stats']['min_unique'], unique_goals)
            stats['goals_stats']['max_unique'] = max(stats['goals_stats']['max_unique'], unique_goals)
            stats['goals_stats']['avg_unique'] += unique_goals
            
            # Blockers stats
            total_blockers = level['metadata']['total_blockers']
            unique_blockers = level['metadata']['unique_blockers']
            
            stats['blockers_stats']['min_total'] = min(stats['blockers_stats']['min_total'], total_blockers)
            stats['blockers_stats']['max_total'] = max(stats['blockers_stats']['max_total'], total_blockers)
            stats['blockers_stats']['avg_total'] += total_blockers
            
            stats['blockers_stats']['min_unique'] = min(stats['blockers_stats']['min_unique'], unique_blockers)
            stats['blockers_stats']['max_unique'] = max(stats['blockers_stats']['max_unique'], unique_blockers)
            stats['blockers_stats']['avg_unique'] += unique_blockers
        
        # Calculate averages
        total_levels = len(processed_data)
        stats['time_stats']['avg'] /= total_levels
        stats['goals_stats']['avg_total'] /= total_levels
        stats['goals_stats']['avg_unique'] /= total_levels
        stats['blockers_stats']['avg_total'] /= total_levels
        stats['blockers_stats']['avg_unique'] /= total_levels
        
        return stats

    def convert_to_unity_format(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert processed level data to Unity-compatible format."""
        unity_levels = []
        
        for level in processed_data:
            unity_level = {
                "i": level["level_id"],
                "d": level["metadata"]["time_limit"],
                "t": level["metadata"]["level_parameter"],
                "l": []
            }
            
            # Convert goals to Unity format
            for goal_key, amount in level["goals"].items():
                goal_id = int(goal_key.split("_")[1])  # Extract numeric ID from "goal_1"
                unity_level["l"].append({
                    "t": self._map_object_to_unity_type(goal_id),
                    "a": amount,
                    "r": True  # Goals are required objects
                })
                
            # Convert blockers to Unity format
            for blocker_key, amount in level["blockers"].items():
                blocker_id = int(blocker_key.split("_")[1])
                unity_level["l"].append({
                    "t": self._map_object_to_unity_type(blocker_id),
                    "a": amount,
                    "r": False  # Blockers are not required objects
                })
                
            unity_levels.append(unity_level)
            
        return unity_levels

    def validate_unity_format(self, unity_levels: List[Dict[str, Any]]) -> bool:
        """Validate that the converted levels match Unity format requirements."""
        for level in unity_levels:
            # Check required fields
            if not all(key in level for key in ["i", "d", "t", "l"]):
                logger.error(f"Level {level.get('i', 'unknown')} missing required fields")
                return False
                
            # Validate time limit
            if not (30 <= level["d"] <= 300):
                logger.warning(f"Level {level['i']} has unusual time limit: {level['d']}s")
                
            # Validate objects
            for obj in level["l"]:
                if not all(key in obj for key in ["t", "a", "r"]):
                    logger.error(f"Level {level['i']} has object missing required fields")
                    return False
                    
                if not (2 <= obj["a"] <= 9):
                    logger.warning(f"Level {level['i']} has unusual amount: {obj['a']}")
                    
                if obj["t"] not in [item for sublist in OBJECT_CATEGORIES.values() for item in sublist]:
                    logger.error(f"Level {level['i']} has unknown object type: {obj['t']}")
                    return False
                    
        return True

def main():
    logger.info("Starting data processing")
    
    # Create output directory if it doesn't exist
    PROCESSED_LEVELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    processor = DataProcessor()
    
    try:
        # Process level data
        processed_data = processor.load_level_data(LEVEL_DATA_FILE)
        
        # Generate statistics
        stats = processor.generate_statistics(processed_data)
        
        # Convert to Unity format
        unity_levels = processor.convert_to_unity_format(processed_data)
        
        # Validate Unity format
        if not processor.validate_unity_format(unity_levels):
            raise ValueError("Generated levels failed Unity format validation")
        
        # Save processed data (both formats)
        with open(PROCESSED_LEVELS_FILE, 'w') as f:
            json.dump({
                "raw_format": processed_data,
                "unity_format": unity_levels
            }, f, indent=2)
        logger.info(f"Saved processed levels to {PROCESSED_LEVELS_FILE}")
        
        with open(DATASET_STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved dataset statistics to {DATASET_STATS_FILE}")
        
        # Log summary statistics
        logger.info(f"Processed {stats['total_levels']} levels")
        logger.info(f"Difficulty distribution: {stats['difficulty_distribution']}")
        logger.info(f"Level parameter distribution: {stats['level_parameter_distribution']}")
        
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 