from typing import List, Dict
import numpy as np
import json
import os
from ..agents.level_evaluator import LevelEvaluator
from ..agents.player_simulation import Strategy

def load_levels_from_file(filepath: str) -> List[Dict]:
    """Load level data from the raw levels design file"""
    print(f"Attempting to load levels from: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
        print(f"Successfully loaded {len(data)} levels")
        
        # Convert level format
        for level in data:
            level['layout'] = np.array(level['layout'])
            
        return data

def create_sample_level(size=10, difficulty=0.5):
    """Create a sample level for testing"""
    print(f"Creating sample level with size {size} and difficulty {difficulty}")
    grid = np.full((size, size), ' ')
    
    # Add start and goal
    grid[0, 0] = 'S'
    grid[-1, -1] = 'G'
    
    # Add some objects
    objects = ['A', 'B', 'C']
    num_objects = int(size * size * 0.3)  # 30% density
    
    for _ in range(num_objects):
        x, y = np.random.randint(0, size, 2)
        if grid[x, y] == ' ':
            grid[x, y] = np.random.choice(objects)
            
    return {
        'layout': grid,
        'difficulty_tag': difficulty,
        'required_objects': {'A': 3, 'B': 3, 'C': 3},
        'required_triples': 3
    }

def main():
    print("Starting level evaluation test...")
    # Try to load actual level data
    data_path = os.path.join('data', 'raw', 'levelsdesign.txt')
    try:
        levels = load_levels_from_file(data_path)
        print(f"Loaded {len(levels)} levels from file")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not load level data: {e}")
        print("Falling back to sample levels...")
        # Create sample levels as backup
        num_levels = 50
        print(f"Generating {num_levels} sample levels...")
        levels = [
            create_sample_level(
                size=10,
                difficulty=np.random.uniform(0.3, 0.8)
            )
            for _ in range(num_levels)
        ]
    
    print("\nInitializing evaluator...")
    # Initialize evaluator
    evaluator = LevelEvaluator(min_success_rate=0.3)
    
    # Run evaluation and get report
    print("\nRunning level evaluation...")
    target_difficulty = 0.5
    filtered_levels, report = evaluator.evaluate_and_report(
        levels,
        target_difficulty=target_difficulty,
        num_candidates=20
    )
    
    # Print report
    print("\n" + "="*50)
    print(report)
    
    # Print detailed statistics for first level
    if filtered_levels:
        level = filtered_levels[0]
        print("\nDetailed Analysis of Best Level:")
        print(f"Difficulty Score: {level.difficulty_score:.2f}")
        print("\nStrategy Performance:")
        
        for strategy in Strategy:
            results = level.strategy_results[strategy]
            successes = [r for r in results if r.success]
            success_rate = level.success_rates[strategy]
            avg_moves = level.average_moves[strategy]
            
            print(f"\n{strategy.value.title()} Strategy:")
            print(f"Success Rate: {success_rate*100:.1f}%")
            print(f"Average Moves: {avg_moves:.1f}")
            print(f"Average Triples: {np.mean([r.triples_formed for r in successes]):.1f}")
    else:
        print("\nNo levels passed all criteria!")
            
if __name__ == "__main__":
    print("Starting test execution...")
    main()
    print("Test execution completed.") 