# Level Designer

An AI-powered level design system using conditional VAE with spatial awareness.

## Data Format

### Sequence Format
The system uses a sequence-based format for level representation:

```python
{
    'level_data': np.array,      # Shape: (max_objects * 3,) - [type, size, shape] for each object
    'conditions': np.array,      # Shape: (3,) - [difficulty, time_limit, object_count]
    'spatial_features': np.array # Shape: (max_objects * 2,) - [x, y] grid positions
}
```

### Positional Encoding
- Each object is assigned a grid position normalized to [-1, 1]
- Grid positions are based on row/column indices
- Positional embeddings use sinusoidal encoding for better spatial awareness

### Condition Embeddings
The system uses a flexible condition encoding mechanism:
- Difficulty: Normalized [0, 1]
- Time Limit: Normalized [30s, 300s] → [0, 1]
- Object Count: Normalized by max_objects

## Architecture

### Enhanced CVAE
- Attention-based spatial encoder
- β-VAE with KL annealing
- Flexible condition encoder
- Physics-aware GNN for object relationships

### Training Configuration
```python
MODEL_CONFIG = {
    'latent_dim': 32,
    'hidden_dims': [256, 128],
    'attention_heads': 4,
    'dropout': 0.1
}

TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 0.0001,
    'beta': 1.5,
    'kl_anneal_rate': 0.005
}
```

### Monitoring
- Loss components (reconstruction, KL)
- Attention weights visualization
- Latent space analysis
- Gradient statistics

## Unity Integration
The system maintains backward compatibility with Unity:
- Positions are mapped to Unity's coordinate system
- Object properties are preserved (type, scale, rotation)
- Special flags (t, a, r) are handled appropriately

## Usage

1. Data Processing:
```bash
python src/data_processing/processor.py
```

2. Training:
```bash
python src/train_model.py
```

3. Level Generation:
```bash
python src/generation/level_generator.py
```

## Development

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA or MPS (Apple Silicon)