import torch
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from models.cvae import CVAE
from generation.level_generator import LevelGenerator
from config.model_config import GenerationConfig
from utils.difficulty_scaling import DifficultyScaler

def analyze_level_distribution(levels):
    """Analyze the distribution of level parameters"""
    data = []
    
    for level in levels:
        data.append({
            'level_number': level['level_number'],
            'time_limit': level['time_limit'],
            'difficulty': level['difficulty'],
            'object_count': len(level['objects']),
            'goal_count': len([o for o in level['objects'] if not o['properties']['t']]),
            'blocker_count': len([o for o in level['objects'] if o['properties']['t']])
        })
    
    df = pd.DataFrame(data)
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Time limits by difficulty
    sns.boxplot(data=df, x='difficulty', y='time_limit', ax=axes[0,0])
    axes[0,0].set_title('Time Limits by Difficulty')
    
    # Object counts by difficulty
    sns.boxplot(data=df, x='difficulty', y='object_count', ax=axes[0,1])
    axes[0,1].set_title('Object Counts by Difficulty')
    
    # Goal vs Blocker ratio
    sns.scatterplot(data=df, x='goal_count', y='blocker_count', 
                   hue='difficulty', ax=axes[1,0])
    axes[1,0].set_title('Goal vs Blocker Distribution')
    
    # Progressive difficulty
    sns.lineplot(data=df, x='level_number', y='object_count', ax=axes[1,1])
    axes[1,1].set_title('Level Progression')
    
    plt.tight_layout()
    return fig, df

def test_generation_pipeline(model_path: str, num_levels: int = 100):
    """Test the level generation pipeline"""
    print("\nLoading model and configurations...")
    model = CVAE.load_from_checkpoint(model_path)
    model.eval()
    
    config = GenerationConfig()
    generator = LevelGenerator(model, config_path="config/generation_config.json")
    
    print(f"\nGenerating {num_levels} test levels...")
    levels = generator.generate_batch(
        start_level=1,
        count=num_levels,
        output_path=f"generated_levels/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    print("\nAnalyzing level distribution...")
    fig, df = analyze_level_distribution(levels)
    
    # Save analysis
    analysis_path = Path("analysis")
    analysis_path.mkdir(exist_ok=True)
    
    fig.savefig(analysis_path / f"distribution_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # Print summary statistics
    print("\nDifficulty Distribution:")
    print(df['difficulty'].value_counts())
    
    print("\nObject Count Statistics:")
    print(df['object_count'].describe())
    
    print("\nTime Limit Statistics:")
    print(df['time_limit'].describe())
    
    return levels, df

if __name__ == "__main__":
    model_path = "checkpoints/latest.ckpt"  # Update with your model path
    levels, stats = test_generation_pipeline(model_path, num_levels=100)
    
    print("\nTest generation complete. Check the analysis folder for visualizations.") 