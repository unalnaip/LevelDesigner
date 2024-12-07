import torch
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from models.cvae import EnhancedCVAE

def load_item_data():
    """Load and parse all items and their variations from ItemData.json"""
    item_path = Path("/Users/naipunal/Documents/GitHub/RestPlay.MatchAges.Unity/Assets/_RestPlay/Addressables/GameData/ItemData.json")
    with open(item_path, 'r') as f:
        items = json.load(f)

    # Group variations by category
    category_items = defaultdict(list)
    all_variations = []
    
    for item in items:
        category = item['category']
        for var in item['variation']:
            var_name = var['name']
            var_data = {
                'name': var_name,
                'category': category,
                'size': item['size'],
                'shape': item['shape'],
                'scale': item['scale'],
                'color': var['color']
            }
            category_items[category].append(var_data)
            all_variations.append(var_data)
    
    return category_items, all_variations

def setup_logger():
    """Configure logging for batch generation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("generated_levels")
    output_dir.mkdir(exist_ok=True)
    
    log_file = output_dir / f"generation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(checkpoint_path, config):
    """Load the trained model from checkpoint"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = EnhancedCVAE(
        input_dim=config['data']['max_objects'] * 3,
        condition_dim=3,
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        attention_heads=config['model']['attention_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def create_level_conditions(num_levels=2000):
    """Create balanced set of level conditions"""
    conditions = []
    
    # Define difficulty ranges (in seconds)
    difficulties = {
        'Easy': (60, 120),
        'Normal': (120, 180),
        'Hard': (180, 240)
    }
    
    difficulty_counts = {
        'Easy': int(num_levels * 0.3),
        'Normal': int(num_levels * 0.4),
        'Hard': int(num_levels * 0.3)
    }
    
    # Adjust for rounding
    total = sum(difficulty_counts.values())
    if total < num_levels:
        difficulty_counts['Normal'] += num_levels - total
    
    for difficulty, count in difficulty_counts.items():
        time_range = difficulties[difficulty]
        for i in range(count):
            # Generate duration
            duration = np.random.randint(*time_range)
            
            # Generate condition values
            conditions.append({
                'index': i + 1,
                'duration': duration,
                'difficulty': difficulty,
                'difficulty_value': 0.2 if difficulty == 'Easy' else 0.5 if difficulty == 'Normal' else 0.8,
                'object_count': np.random.randint(3, 8)  # Reasonable range for number of objects
            })
    
    # Shuffle conditions
    np.random.shuffle(conditions)
    # Reassign indices after shuffling
    for i, condition in enumerate(conditions):
        condition['index'] = i + 1
        
    return conditions

def select_objects_for_level(category_items, condition):
    """Select objects for a level based on difficulty and categories"""
    num_objects = condition['object_count']
    difficulty = condition['difficulty']
    
    # Define category weights based on difficulty
    # Categories: 2=Fruits, 5=Stationery, 7=Sports, 15=Text
    category_weights = {
        'Easy': {2: 0.4, 5: 0.4, 7: 0.2, 15: 0.0},  # More fruits and stationery
        'Normal': {2: 0.3, 5: 0.3, 7: 0.3, 15: 0.1},  # Balanced
        'Hard': {2: 0.2, 5: 0.2, 7: 0.4, 15: 0.2}   # More sports and text
    }
    
    weights = category_weights[difficulty]
    
    # Calculate number of objects per category
    category_counts = {}
    remaining = num_objects
    for cat in sorted(weights.keys()):
        if cat == list(weights.keys())[-1]:
            category_counts[cat] = remaining
        else:
            count = int(num_objects * weights[cat])
            category_counts[cat] = max(1, count) if weights[cat] > 0 else 0
            remaining -= category_counts[cat]
    
    # Select objects
    selected_objects = []
    used_variations = set()
    used_shapes = set()  # Track shape variety
    used_sizes = set()   # Track size variety
    
    for category, count in category_counts.items():
        if count <= 0:
            continue
            
        # Get available variations for this category
        variations = category_items[category]
        if not variations:
            continue
            
        # Filter out already used variations
        available = [v for v in variations if v['name'] not in used_variations]
        if not available:
            available = variations  # If all used, allow reuse
            
        # Try to select objects with different shapes and sizes
        selected_indices = []
        attempts = 0
        while len(selected_indices) < min(count, len(available)) and attempts < 100:
            idx = np.random.randint(0, len(available))
            if idx in selected_indices:
                attempts += 1
                continue
                
            var = available[idx]
            # Prefer objects with new shapes/sizes when possible
            if (len(used_shapes) < 5 and var['shape'] not in used_shapes) or \
               (len(used_sizes) < 4 and var['size'] not in used_sizes) or \
               attempts > 50:  # After many attempts, be less strict
                selected_indices.append(idx)
                used_shapes.add(var['shape'])
                used_sizes.add(var['size'])
                used_variations.add(var['name'])
                selected_objects.append(var)
            attempts += 1
    
    # If we couldn't get enough objects with variety, fill remaining slots
    if len(selected_objects) < num_objects:
        remaining_count = num_objects - len(selected_objects)
        all_variations = [v for cat_vars in category_items.values() for v in cat_vars]
        additional = np.random.choice(all_variations, size=remaining_count, replace=False)
        selected_objects.extend(additional)
    
    return selected_objects

def convert_to_game_format(properties, condition, selected_objects):
    """Convert model output to game-compatible format"""
    num_objects = len(selected_objects)
    difficulty = condition['difficulty_value']
    
    # Create level objects
    level_objects = []
    
    # Determine number of goal objects based on difficulty
    num_goals = max(1, int(num_objects * (0.6 - difficulty * 0.2)))  # More goals in easier levels
    goal_indices = np.random.choice(num_objects, num_goals, replace=False)
    
    for i, obj in enumerate(selected_objects):
        # Get object properties from model output
        obj_value = float(properties[i * 3 + 1])
        
        # Convert value to amount (2-7 range, influenced by difficulty)
        base_amount = 2 + int(obj_value * 3)  # 2-5 base range
        if difficulty > 0.6:  # Hard levels get higher amounts
            base_amount += 2
        amount = min(7, base_amount)
        
        # Determine if it's a goal object
        is_goal = i in goal_indices
        
        # Create object
        game_obj = {
            "t": obj['name'],
            "a": amount,
            "r": is_goal
        }
        level_objects.append(game_obj)
    
    # Create level data
    level = {
        "i": condition['index'],
        "d": condition['duration'],
        "t": 1,  # Type 1 for all levels for now
        "l": level_objects
    }
    
    return level

def generate_levels(model, device, conditions, category_items):
    """Generate levels based on conditions"""
    levels = []
    
    # Progress bar
    pbar = tqdm(conditions, desc="Generating levels")
    
    for condition in pbar:
        # Create condition tensor
        condition_tensor = torch.FloatTensor([
            condition['difficulty_value'],
            condition['duration'] / 300,  # Normalize duration
            condition['object_count'] / 10  # Normalize object count
        ]).to(device)
        
        # Select objects for this level
        selected_objects = select_objects_for_level(category_items, condition)
        
        # Generate level
        with torch.no_grad():
            z = torch.randn(1, model.latent_dim).to(device)
            properties, _ = model.decode(z, condition_tensor.unsqueeze(0))
            
            # Convert to numpy
            properties = properties.cpu().numpy()[0]
            
            # Convert to game format
            level = convert_to_game_format(properties, condition, selected_objects)
            levels.append(level)
    
    return levels

def main():
    # Setup logging
    logger = setup_logger()
    
    # Load item data
    logger.info("Loading item data...")
    category_items, all_variations = load_item_data()
    logger.info(f"Loaded {len(all_variations)} total variations across {len(category_items)} categories")
    
    # Load configuration from best model
    run_dir = sorted(Path("runs").glob("*"))[-1]
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Load model
    logger.info("Loading model from checkpoint...")
    model, device = load_model(checkpoint_path, config)
    
    # Create conditions for 2000 levels
    logger.info("Creating level conditions...")
    conditions = create_level_conditions(num_levels=2000)
    
    # Generate levels
    logger.info("Generating levels...")
    levels = generate_levels(model, device, conditions, category_items)
    
    # Save levels
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("generated_levels") / f"levels_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(levels, f, indent=2)
    
    logger.info(f"\nGeneration complete! Levels saved to: {output_file}")
    logger.info(f"Total levels generated: {len(levels)}")
    
    # Log category distribution
    category_counts = defaultdict(int)
    shape_counts = defaultdict(int)
    size_counts = defaultdict(int)
    
    for level in levels:
        for obj in level['l']:
            obj_type = obj['t']
            for cat, items in category_items.items():
                for item in items:
                    if item['name'] == obj_type:
                        category_counts[cat] += 1
                        shape_counts[item['shape']] += 1
                        size_counts[item['size']] += 1
                        break
    
    logger.info("\nCategory distribution in generated levels:")
    total_objects = sum(category_counts.values())
    for category, count in sorted(category_counts.items()):
        percentage = (count / total_objects) * 100
        logger.info(f"Category {category}: {count} objects ({percentage:.1f}%)")
    
    logger.info("\nShape distribution in generated levels:")
    for shape, count in sorted(shape_counts.items()):
        percentage = (count / total_objects) * 100
        logger.info(f"Shape {shape}: {count} objects ({percentage:.1f}%)")
    
    logger.info("\nSize distribution in generated levels:")
    for size, count in sorted(size_counts.items()):
        percentage = (count / total_objects) * 100
        logger.info(f"Size {size}: {count} objects ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 