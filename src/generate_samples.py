import torch
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime

from models.cvae import EnhancedCVAE
from utils.level_dataset import LevelDataset

def setup_logger():
    """Configure logging for sample generation"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sample_generation_{timestamp}.log"
    
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

def load_model(checkpoint_path, config):
    """Load the trained model from checkpoint"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize model with same configuration
    model = EnhancedCVAE(
        input_dim=config['data']['max_objects'] * 3,
        condition_dim=3,  # difficulty, time_limit, object_count
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        attention_heads=config['model']['attention_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def generate_samples(model, device, num_samples=20):
    """Generate samples with varying conditions"""
    samples = []
    
    # Generate samples for different difficulty levels
    difficulties = ['Easy', 'Normal', 'Hard']
    for difficulty in difficulties:
        diff_value = 0.2 if difficulty == 'Easy' else 0.5 if difficulty == 'Normal' else 0.8
        
        for _ in range(num_samples // len(difficulties)):
            # Create condition vector
            time_limit = np.random.uniform(30, 300)
            object_count = np.random.randint(5, 30)
            
            condition = torch.FloatTensor([
                diff_value,
                time_limit / 300,  # Normalize time limit
                object_count / 30  # Normalize object count
            ]).to(device)
            
            # Generate level
            with torch.no_grad():
                z = torch.randn(1, model.latent_dim).to(device)
                properties, positions = model.decode(z, condition.unsqueeze(0))
                
                # Convert to numpy for saving
                properties = properties.cpu().numpy()[0]
                positions = positions.cpu().numpy()[0]
                
                # Create sample metadata
                sample = {
                    'difficulty': difficulty,
                    'difficulty_value': diff_value,
                    'time_limit': int(time_limit),
                    'object_count': int(object_count),
                    'properties': properties.tolist(),
                    'positions': positions.tolist()
                }
                
                # Add statistics
                sample['stats'] = {
                    'avg_object_size': float(np.mean(properties[1::3])),
                    'unique_types': len(np.unique(properties[::3])),
                    'spatial_spread': float(np.std(positions))
                }
                
                samples.append(sample)
    
    return samples

def analyze_samples(samples):
    """Analyze generated samples for quality metrics"""
    analysis = {
        'Easy': {'sizes': [], 'types': [], 'spreads': []},
        'Normal': {'sizes': [], 'types': [], 'spreads': []},
        'Hard': {'sizes': [], 'types': [], 'spreads': []}
    }
    
    for sample in samples:
        diff = sample['difficulty']
        stats = sample['stats']
        
        analysis[diff]['sizes'].append(stats['avg_object_size'])
        analysis[diff]['types'].append(stats['unique_types'])
        analysis[diff]['spreads'].append(stats['spatial_spread'])
    
    # Calculate statistics per difficulty
    summary = {}
    for diff in analysis:
        summary[diff] = {
            'avg_object_size': float(np.mean(analysis[diff]['sizes'])),
            'avg_unique_types': float(np.mean(analysis[diff]['types'])),
            'avg_spatial_spread': float(np.mean(analysis[diff]['spreads'])),
            'size_std': float(np.std(analysis[diff]['sizes'])),
            'type_std': float(np.std(analysis[diff]['types'])),
            'spread_std': float(np.std(analysis[diff]['spreads']))
        }
    
    return summary

def main():
    # Load configuration from best model
    run_dir = sorted(Path("runs").glob("*"))[-1]  # Get latest run
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Create samples directory
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    # Load model
    logger.info("Loading model from checkpoint...")
    model, device = load_model(checkpoint_path, config)
    
    # Generate samples
    logger.info("Generating samples...")
    samples = generate_samples(model, device)
    
    # Analyze samples
    logger.info("Analyzing samples...")
    analysis = analyze_samples(samples)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'samples': samples,
        'analysis': analysis,
        'config': config,
        'timestamp': timestamp
    }
    
    output_path = samples_dir / f"samples_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Log analysis
    logger.info("\nSample Analysis Summary:")
    for diff in analysis:
        logger.info(f"\n{diff} Levels:")
        for metric, value in analysis[diff].items():
            logger.info(f"  {metric}: {value:.3f}")
    
    logger.info(f"\nSamples saved to: {output_path}")

if __name__ == "__main__":
    main() 