import numpy as np
import torch
from pathlib import Path
import logging
from datetime import datetime
import json

# Configure logging
def setup_logger():
    """Configure logging for the data processor"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"data_processing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

class DataProcessor:
    """Process and normalize level data for training"""
    def __init__(self, config):
        self.config = config
        self.logger = logger
        
    def process_level_data(self, raw_data):
        """Process raw level data into training format"""
        try:
            self.logger.info("Processing level data...")
            processed_data = []
            
            for level in raw_data:
                # Extract object properties
                objects = level.get('objects', [])
                if not objects:
                    self.logger.warning(f"Empty level found, skipping")
                    continue
                
                # Process object properties and sequence
                level_data, object_sequence = self._process_objects(objects)
                
                # Process conditions
                conditions = self._process_conditions(level)
                
                # Generate spatial features and positional encodings
                spatial_features, positional_encodings = self._generate_spatial_features(objects)
                
                processed_data.append({
                    'level_data': level_data,
                    'object_sequence': object_sequence,
                    'conditions': conditions,
                    'spatial_features': spatial_features,
                    'positional_encodings': positional_encodings
                })
            
            self.logger.info(f"Processed {len(processed_data)} levels")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing level data: {str(e)}")
            raise
    
    def _process_objects(self, objects):
        """Process object properties into tensor format and sequence"""
        try:
            max_objects = self.config['max_objects']
            features_per_object = 3  # type, size, shape
            
            # Initialize tensors
            level_data = np.zeros((max_objects, features_per_object))
            object_sequence = []
            
            for i, obj in enumerate(objects[:max_objects]):
                # Extract features
                obj_type = obj.get('type', 0)
                size = max(obj['scale']['x'], obj['scale']['y'], obj['scale']['z'])
                shape = self._determine_shape(obj['scale'])
                
                # Store features
                level_data[i] = [obj_type, size, shape]
                
                # Create sequence entry
                object_sequence.append({
                    't': obj_type,
                    'a': obj.get('amount', 1),
                    'r': obj.get('is_goal', False),
                    'features': [obj_type, size, shape]
                })
            
            return level_data.flatten(), object_sequence
            
        except Exception as e:
            self.logger.error(f"Error processing objects: {str(e)}")
            raise
    
    def _process_conditions(self, level):
        """Process level conditions into normalized format"""
        try:
            # Extract conditions
            difficulty = level.get('difficulty', 0)
            time_limit = level.get('time_limit', 180)
            num_objects = len(level.get('objects', []))
            
            # Normalize
            norm_time = (time_limit - self.config['time_limit']['min']) / (
                self.config['time_limit']['max'] - self.config['time_limit']['min']
            )
            norm_objects = num_objects / self.config['max_objects']
            
            return np.array([difficulty, norm_time, norm_objects])
            
        except Exception as e:
            self.logger.error(f"Error processing conditions: {str(e)}")
            raise
    
    def _generate_spatial_features(self, objects):
        """Generate spatial features and positional encodings"""
        try:
            max_objects = self.config['max_objects']
            grid_size = self.config['grid_size']
            encoding_dim = self.config.get('encoding_dim', 32)
            
            # Initialize features
            spatial_features = np.zeros((max_objects, 2))  # x, y coordinates
            
            # Generate positional encodings
            position = torch.arange(0, max_objects).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, encoding_dim, 2) * 
                               (-np.log(10000.0) / encoding_dim))
            
            pos_encoding = torch.zeros(max_objects, encoding_dim)
            pos_encoding[:, 0::2] = torch.sin(position * div_term)
            pos_encoding[:, 1::2] = torch.cos(position * div_term)
            
            for i, obj in enumerate(objects[:max_objects]):
                # Calculate grid position
                row = i // grid_size[1]
                col = i % grid_size[1]
                
                # Normalize to [-1, 1]
                norm_row = 2 * (row / (grid_size[0] - 1)) - 1 if grid_size[0] > 1 else 0
                norm_col = 2 * (col / (grid_size[1] - 1)) - 1 if grid_size[1] > 1 else 0
                
                spatial_features[i] = [norm_row, norm_col]
            
            return spatial_features.flatten(), pos_encoding.numpy()
            
        except Exception as e:
            self.logger.error(f"Error generating spatial features: {str(e)}")
            raise
    
    def _determine_shape(self, scale):
        """Determine object shape based on scale ratios"""
        try:
            x, y, z = scale['x'], scale['y'], scale['z']
            total = x + y + z
            if total == 0:
                return 0
            
            x, y, z = x/total, y/total, z/total
            
            if abs(x - y) < 0.1 and abs(y - z) < 0.1:
                return 0  # Cube
            elif y > max(x, z) * 2:
                return 1  # Vertical
            elif x > max(y, z) * 2:
                return 2  # Horizontal
            elif z > max(x, y) * 2:
                return 3  # Deep
            elif abs(x - z) < 0.1 and y < min(x, z):
                return 4  # Flat
            elif max(x, y, z) / min(x, y, z) > 3:
                return 5  # Long
            else:
                return 6  # Irregular
                
        except Exception as e:
            self.logger.error(f"Error determining shape: {str(e)}")
            raise
    
    def save_processed_data(self, processed_data, output_path):
        """Save processed data to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = []
            for level in processed_data:
                serializable_level = {
                    'level_data': level['level_data'].tolist(),
                    'object_sequence': level['object_sequence'],
                    'conditions': level['conditions'].tolist(),
                    'spatial_features': level['spatial_features'].tolist(),
                    'positional_encodings': level['positional_encodings'].tolist()
                }
                serializable_data.append(serializable_level)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            self.logger.info(f"Saved processed data to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise

def process_data(input_path, output_path, config):
    """Process raw level data and save to file"""
    processor = DataProcessor(config)
    
    try:
        # Load raw data
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
        
        # Process data
        processed_data = processor.process_level_data(raw_data)
        
        # Save processed data
        processor.save_processed_data(processed_data, output_path)
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    config = {
        'max_objects': 30,
        'grid_size': (6, 7),
        'encoding_dim': 32,
        'time_limit': {
            'min': 30,
            'max': 300
        }
    }
    
    # Process data
    process_data(
        'data/raw/levels.json',
        'data/processed/processed_levels.json',
        config
    )