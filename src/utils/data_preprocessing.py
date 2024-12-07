import json
import os
import logging
from datetime import datetime
import numpy as np
from pathlib import Path

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Unity format specifications
UNITY_LEVEL_FORMAT = {
    "i": "level_index",
    "d": "duration_seconds",
    "t": "level_type",
    "l": "objects"
}

UNITY_OBJECT_FORMAT = {
    "t": "object_type",
    "a": "amount",
    "r": "is_goal"
}

# Object categories and properties
OBJECT_CATEGORIES = {
    'fruits': [
        'apple_red', 'banana_yellow', 'cherry_red',
        'grape_green', 'grape_purple', 'pear_green'
    ],
    'school': [
        'pencil', 'ruler', 'eraser', 'crayon',
        'calculator', 'notebook', 'textbook'
    ],
    'sports': [
        'baseball', 'basketball', 'tennis_ball',
        'volleyball', 'soccer_ball'
    ]
}

class DataProcessor:
    def __init__(self):
        """Initialize data processor with mappings and configurations"""
        self.object_mapping = self._initialize_object_mapping()
        self.shape_mapping = {
            'sphere': 0,
            'cube': 1,
            'cylinder': 2,
            'flat': 3,
            'curved': 4,
            'sharp': 5,
            'mix': 6
        }
        
        logger.info("Data processor initialized")
    
    def _initialize_object_mapping(self):
        """Create mapping between object types and IDs"""
        object_mapping = {}
        reverse_mapping = {}
        current_id = 1
        
        for category, objects in OBJECT_CATEGORIES.items():
            for obj_type in objects:
                object_mapping[obj_type] = current_id
                reverse_mapping[current_id] = obj_type
                current_id += 1
        
        self.reverse_mapping = reverse_mapping  # For converting back to Unity format
        logger.info(f"Initialized object mapping with {len(object_mapping)} objects")
        return object_mapping
    
    def _map_object_to_unity_type(self, obj_type):
        """Convert object type to Unity object type name"""
        if isinstance(obj_type, int):
            if obj_type not in self.reverse_mapping:
                raise ValueError(f"Unknown object ID: {obj_type}")
            return self.reverse_mapping[obj_type]
        elif isinstance(obj_type, str):
            if obj_type not in self.object_mapping:
                raise ValueError(f"Unknown object type: {obj_type}")
            return obj_type
        else:
            raise ValueError(f"Invalid object type format: {obj_type}")
    
    def _get_object_id(self, obj_type):
        """Get numeric ID for object type"""
        if isinstance(obj_type, int):
            if obj_type not in self.reverse_mapping:
                raise ValueError(f"Unknown object ID: {obj_type}")
            return obj_type
        elif isinstance(obj_type, str):
            if obj_type not in self.object_mapping:
                raise ValueError(f"Unknown object type: {obj_type}")
            return self.object_mapping[obj_type]
        else:
            raise ValueError(f"Invalid object type format: {obj_type}")
    
    def normalize_object_properties(self, obj):
        """Normalize object properties for consistent processing"""
        # Extract and normalize position
        position = obj.get('position', {'x': 0, 'y': 0, 'z': 0})
        position = {k: float(v) for k, v in position.items()}
        
        # Extract and normalize scale
        scale = obj.get('scale', {'x': 1, 'y': 1, 'z': 1})
        scale = {k: float(v) for k, v in scale.items()}
        
        # Extract and normalize rotation
        rotation = obj.get('rotation', {'x': 0, 'y': 0, 'z': 0})
        rotation = {k: float(v) for k, v in rotation.items()}
        
        # Determine shape from scale ratios
        shape = self.determine_shape(scale)
        
        # Convert object type to ID
        obj_type = obj.get('type', 0)
        obj_id = self._get_object_id(obj_type)
        
        return {
            'type': obj_id,
            'position': position,
            'scale': scale,
            'rotation': rotation,
            'shape': shape,
            'flags': {
                't': obj.get('t', False),  # Target/goal object
                'a': obj.get('a', False),  # Active/interactive
                'r': obj.get('r', False)   # Required for completion
            }
        }
    
    def determine_shape(self, scale):
        """Determine object shape category based on scale ratios"""
        x, y, z = scale['x'], scale['y'], scale['z']
        total = x + y + z
        if total == 0:
            return self.shape_mapping['cube']
        
        x, y, z = x/total, y/total, z/total
        
        if abs(x - y) < 0.1 and abs(y - z) < 0.1:
            return self.shape_mapping['sphere']
        elif y > max(x, z) * 2:
            return self.shape_mapping['cylinder']
        elif x > max(y, z) * 2:
            return self.shape_mapping['flat']
        elif z > max(x, y) * 2:
            return self.shape_mapping['curved']
        elif max(x, y, z) / min(x, y, z) > 3:
            return self.shape_mapping['sharp']
        else:
            return self.shape_mapping['mix']
    
    def process_level(self, raw_level):
        """Process a single level's data"""
        try:
            # Extract metadata
            metadata = raw_level.get('metadata', {})
            difficulty = metadata.get('difficulty', 0.5)
            time_limit = metadata.get('time_limit', 180)
            
            # Process objects
            objects = []
            for obj in raw_level.get('objects', []):
                normalized_obj = self.normalize_object_properties(obj)
                objects.append(normalized_obj)
            
            # Sort objects by Y position for layering
            objects.sort(key=lambda x: x['position']['y'])
            
            return {
                'difficulty': difficulty,
                'time_limit': time_limit,
                'objects': objects
            }
        except Exception as e:
            logger.error(f"Error processing level: {str(e)}")
            return None
    
    def convert_to_unity_format(self, processed_data):
        """Convert processed data to Unity-compatible format"""
        unity_levels = []
        
        for level in processed_data:
            unity_level = {
                "i": len(unity_levels) + 1,  # Level index
                "d": level["time_limit"],
                "t": int(level["difficulty"] * 100),  # Convert to percentage
                "l": []
            }
            
            # Convert objects to Unity format
            for obj in level["objects"]:
                unity_obj = {
                    "t": self._map_object_to_unity_type(obj["type"]),
                    "a": 1,  # Default amount
                    "r": obj["flags"]["r"]  # Required/goal object
                }
                unity_level["l"].append(unity_obj)
            
            unity_levels.append(unity_level)
        
        return unity_levels
    
    def validate_unity_format(self, unity_levels):
        """Validate Unity format requirements"""
        for level in unity_levels:
            # Check required fields
            if not all(key in level for key in UNITY_LEVEL_FORMAT.keys()):
                logger.error(f"Level {level.get('i', 'unknown')} missing required fields")
                return False
            
            # Validate time limit
            if not (30 <= level["d"] <= 300):
                logger.warning(f"Level {level['i']} has unusual time limit: {level['d']}s")
            
            # Validate objects
            for obj in level["l"]:
                if not all(key in obj for key in UNITY_OBJECT_FORMAT.keys()):
                    logger.error(f"Level {level['i']} has object missing required fields")
                    return False
                
                if obj["t"] not in [item for sublist in OBJECT_CATEGORIES.values() for item in sublist]:
                    logger.error(f"Level {level['i']} has unknown object type: {obj['t']}")
                    return False
        
        return True
    
    def process_dataset(self, raw_data_path, output_path):
        """Process entire dataset and save in both formats"""
        try:
            # Load raw data
            logger.info(f"Loading raw data from {raw_data_path}")
            with open(raw_data_path, 'r') as f:
                raw_data = json.load(f)
            
            # Process each level
            processed_levels = []
            for idx, raw_level in enumerate(raw_data):
                if idx % 100 == 0:
                    logger.info(f"Processing level {idx}/{len(raw_data)}")
                
                processed_level = self.process_level(raw_level)
                if processed_level is not None:
                    processed_levels.append(processed_level)
            
            # Convert to Unity format
            unity_levels = self.convert_to_unity_format(processed_levels)
            
            # Validate Unity format
            if not self.validate_unity_format(unity_levels):
                logger.error("Unity format validation failed")
                return False
            
            # Save both formats
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save processed format
            with open(output_path, 'w') as f:
                json.dump(processed_levels, f, indent=2)
            
            # Save Unity format
            unity_path = output_path.replace('.json', '_unity.json')
            with open(unity_path, 'w') as f:
                json.dump(unity_levels, f, indent=2)
            
            # Calculate and save statistics
            stats = self.calculate_dataset_statistics(processed_levels)
            stats_path = os.path.join(output_dir, 'dataset_statistics.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Processed {len(processed_levels)} levels")
            logger.info(f"Saved processed data to {output_path}")
            logger.info(f"Saved Unity format to {unity_path}")
            logger.info(f"Saved statistics to {stats_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return False
    
    def calculate_dataset_statistics(self, levels):
        """Calculate statistics for the processed dataset"""
        stats = {
            'num_levels': len(levels),
            'difficulty': {
                'mean': np.mean([level['difficulty'] for level in levels]),
                'std': np.std([level['difficulty'] for level in levels]),
                'min': min([level['difficulty'] for level in levels]),
                'max': max([level['difficulty'] for level in levels])
            },
            'time_limit': {
                'mean': np.mean([level['time_limit'] for level in levels]),
                'std': np.std([level['time_limit'] for level in levels]),
                'min': min([level['time_limit'] for level in levels]),
                'max': max([level['time_limit'] for level in levels])
            },
            'objects_per_level': {
                'mean': np.mean([len(level['objects']) for level in levels]),
                'std': np.std([len(level['objects']) for level in levels]),
                'min': min([len(level['objects']) for level in levels]),
                'max': max([len(level['objects']) for level in levels])
            }
        }
        
        # Calculate object type distribution
        type_counts = {}
        for level in levels:
            for obj in level['objects']:
                obj_type = obj['type']
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        stats['object_type_distribution'] = type_counts
        
        return stats

def main():
    """Main preprocessing function"""
    raw_data_path = 'data/raw_levels.json'
    output_path = 'data/processed_levels.json'
    
    logger.info("Starting data preprocessing")
    processor = DataProcessor()
    success = processor.process_dataset(raw_data_path, output_path)
    
    if success:
        logger.info("Data preprocessing completed successfully")
    else:
        logger.error("Data preprocessing failed")

if __name__ == '__main__':
    main() 