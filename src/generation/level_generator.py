import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Thematic object groups
THEMES = {
    'basic_school': {
        'objects': ['pencil', 'ruler', 'rubbereraser', 'crayon', 'markerpen'],
        'min_amount': 2,
        'max_amount': 3
    },
    'fruits_veggies': {
        'objects': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'pear'],
        'min_amount': 2,
        'max_amount': 4
    },
    'sports': {
        'objects': ['soccerball', 'basketball', 'tennisball', 'volleyball', 'baseballbat'],
        'min_amount': 3,
        'max_amount': 4
    },
    'advanced_school': {
        'objects': ['calculator', 'mathcompass', 'earthglobe', 'notebook', 'textbook'],
        'min_amount': 3,
        'max_amount': 5
    }
}

def load_item_data():
    """Load and parse the item data from Unity"""
    with open('/Users/naipunal/Documents/GitHub/RestPlay.MatchAges.Unity/Assets/_RestPlay/Addressables/GameData/ItemData.json', 'r') as f:
        items = json.load(f)
    
    # Create lookup of all variations
    variations = {}
    for item in items:
        for var in item['variation']:
            variations[var['name']] = {
                'category': item['category'],
                'size': item['size'],
                'shape': item['shape']
            }
    
    return variations

def select_theme_objects(theme_name, num_objects, variations):
    """Select objects from a theme with their variations"""
    theme = THEMES[theme_name]
    selected_objects = []
    
    # Get all possible variations for the theme's objects
    theme_variations = []
    for obj in theme['objects']:
        # Get all variations that start with this object name
        obj_variations = [name for name in variations.keys() if name.startswith(obj + '_')]
        theme_variations.extend(obj_variations)
    
    # Randomly select variations
    selected_variations = np.random.choice(theme_variations, num_objects, replace=False)
    
    for var_name in selected_variations:
        amount = np.random.randint(theme['min_amount'], theme['max_amount'] + 1)
        selected_objects.append({
            't': var_name,
            'a': amount,
            'r': np.random.random() > 0.3  # 70% chance of being a collectible
        })
    
    return selected_objects

def generate_playtest_batch(num_levels=2000):
    """Generate a large batch of levels for playtesting"""
    print(f"\nGenerating {num_levels} levels for playtesting...")
    
    # Load item data
    variations = load_item_data()
    
    # Calculate levels per difficulty
    levels_per_difficulty = num_levels // 3
    extra_levels = num_levels % 3
    
    difficulty_counts = {
        'Normal': levels_per_difficulty + (1 if extra_levels > 0 else 0),
        'Hard': levels_per_difficulty + (1 if extra_levels > 1 else 0),
        'Super Hard': levels_per_difficulty
    }
    
    # Time limits per difficulty
    time_limits = {
        'Normal': (120, 130),
        'Hard': (130, 140),
        'Super Hard': (140, 150)
    }
    
    # Objects per difficulty
    objects_per_difficulty = {
        'Normal': (3, 4),
        'Hard': (4, 5),
        'Super Hard': (5, 6)
    }
    
    # Generate levels
    levels = []
    level_id = 1
    
    for difficulty, count in difficulty_counts.items():
        print(f"\nGenerating {count} {difficulty} levels...")
        
        for _ in range(count):
            # Select theme
            theme = np.random.choice(list(THEMES.keys()))
            
            # Generate time limit
            time_min, time_max = time_limits[difficulty]
            time_limit = np.random.randint(time_min, time_max + 1)
            
            # Generate number of objects
            min_obj, max_obj = objects_per_difficulty[difficulty]
            num_objects = np.random.randint(min_obj, max_obj + 1)
            
            # Generate object list
            object_list = select_theme_objects(theme, num_objects, variations)
            
            # Create level
            level = {
                'i': level_id,
                'd': time_limit,
                't': 1 if difficulty == 'Normal' else (2 if difficulty == 'Hard' else 3),
                'l': object_list
            }
            
            levels.append(level)
            level_id += 1
            
            if level_id % 100 == 0:
                print(f"Progress: {level_id}/{num_levels} levels")
    
    # Save to JSON file
    output_path = Path("data/generated/playtest_levels.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(levels, f, indent=2)
    
    print(f"\nGenerated {len(levels)} levels total")
    print(f"Levels saved to {output_path}")
    
    return output_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate game levels')
    parser.add_argument('--playtest', action='store_true', help='Generate a batch of levels for playtesting')
    parser.add_argument('--num_levels', type=int, default=2000, help='Number of levels to generate for playtesting')
    args = parser.parse_args()
    
    if args.playtest:
        generate_playtest_batch(args.num_levels)
    else:
        print("Please use --playtest flag to generate levels") 