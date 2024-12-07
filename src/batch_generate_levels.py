import torch
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

from models.cvae import EnhancedCVAE

# Unity object types
OBJECT_TYPES = [
    "RulerBlue", "RulerRed", "RulerGreen",
    "PencilBlue", "PencilRed", "PencilGreen",
    "EraserBlue", "EraserRed", "EraserGreen"
]

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
                'difficulty_value': 0.2 if difficulty == 'Easy' else 0.5 if difficulty == 'Normal' else 0.8,
                'object_count': np.random.randint(3, 8)  # Reasonable range for number of objects
            })
    
    # Shuffle conditions
    np.random.shuffle(conditions)
    # Reassign indices after shuffling
    for i, condition in enumerate(conditions):
        condition['index'] = i + 1
        
    return conditions

def convert_to_game_format(properties, condition):
    """Convert model output to game-compatible format"""
    num_objects = condition['object_count']
    difficulty = condition['difficulty_value']
    
    # Create level objects
    level_objects = []
    
    # Determine number of goal objects based on difficulty
    num_goals = max(1, int(num_objects * (0.6 - difficulty * 0.2)))  # More goals in easier levels
    goal_indices = np.random.choice(num_objects, num_goals, replace=False)
    
    # Track used types to ensure variety
    used_types = set()
    
    for i in range(num_objects):
        # Get object properties
        obj_type = int(properties[i * 3]) % len(OBJECT_TYPES)
        obj_value = float(properties[i * 3 + 1])
        
        # Ensure type variety
        while OBJECT_TYPES[obj_type] in used_types and len(used_types) < len(OBJECT_TYPES):
            obj_type = (obj_type + 1) % len(OBJECT_TYPES)
        used_types.add(OBJECT_TYPES[obj_type])
        
        # Convert value to amount (2-7 range, influenced by difficulty)
        base_amount = 2 + int(obj_value * 3)  # 2-5 base range
        if difficulty > 0.6:  # Hard levels get higher amounts
            base_amount += 2
        amount = min(7, base_amount)
        
        # Determine if it's a goal object
        is_goal = i in goal_indices
        
        # Create object
        game_obj = {
            "t": OBJECT_TYPES[obj_type],
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

def generate_levels(model, device, conditions):
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
        
        # Generate level
        with torch.no_grad():
            z = torch.randn(1, model.latent_dim).to(device)
            properties, _ = model.decode(z, condition_tensor.unsqueeze(0))
            
            # Convert to numpy
            properties = properties.cpu().numpy()[0]
            
            # Convert to game format
            level = convert_to_game_format(properties, condition)
            levels.append(level)
    
    return levels

def main():
    # Setup logging
    logger = setup_logger()
    
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
    levels = generate_levels(model, device, conditions)
    
    # Save levels
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("generated_levels") / f"levels_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(levels, f, indent=2)
    
    logger.info(f"\nGeneration complete! Levels saved to: {output_file}")
    logger.info(f"Total levels generated: {len(levels)}")

if __name__ == "__main__":
    main() 