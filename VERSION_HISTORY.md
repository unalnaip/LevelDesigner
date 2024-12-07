# Level Designer AI - Version History

## Version 2.0.0 (Current)

### Major Updates
1. Stacking-Aware CVAE Implementation
   - Object relationship modeling
   - Layer-based generation
   - Pairwise object interaction encoding
   - Separate decoders for different stack layers
   - Improved condition handling

2. Object System Integration
   - Unity object type support
   - Category-based object selection
   - Size and shape consideration
   - Collectible/blocker balance

3. Level Generation Improvements
   - Theme-based progression
   - Strategic object placement
   - Difficulty scaling through stacking
   - Time limit optimization

### Technical Details
1. Model Architecture
   - StackingEncoder: Handles object relationships
   - ObjectEncoder: Processes individual properties
   - LayerDecoders: Generate level in layers
   - Condition integration for difficulty

2. Generation Parameters
   - Object counts: 3-6 per level
   - Layer distribution: 3 layers (base, middle, top)
   - Time limits: 120-150 seconds
   - Theme grouping: School, Sports, Food items

### Future Considerations
1. Layer-based Generation System
   - Bottom-up object placement
   - Layer-specific density control
   - Support point calculation
   - Access path validation
   - Expected accuracy: 70-80%

2. Physics-aware Placement
   - Object stability modeling
   - Center of mass calculations
   - Natural settling simulation
   - Real-time physics validation
   - Expected accuracy: 60-70%

### Known Limitations
1. Current Implementation
   - Basic stacking relationships
   - Limited physics understanding
   - Fixed layer structure

2. Potential Improvements
   - Dynamic layer count
   - Real-time physics simulation
   - Advanced object interaction modeling

## Version 1.0.0 (Previous)

### Major Features
1. Conditional VAE Implementation
   - Latent dimension: 32
   - Hidden layers: [256, 128]
   - Position embedding for spatial features
   - Separate encoders for level and spatial features
   - Batch normalization and dropout for regularization

2. Spatial Features Integration
   - Grid-based positioning system
   - Unity coordinate system integration
   - Dynamic grid size calculation based on difficulty
   - Position normalization and denormalization

3. Training System
   - Separate loss terms for spatial and non-spatial features
   - Beta-VAE weighting for KL divergence
   - Learning rate scheduling
   - Early stopping
   - Progress tracking and validation

### Configuration Updates
1. Model Configuration
   - Centralized parameter management
   - Grid size constraints (4-6 rows, 4-7 columns)
   - Spawn area boundaries for Unity
   - Normalization ranges for all features

2. Training Parameters
   - Batch size: 32
   - Learning rate: 0.001
   - Number of epochs: 100
   - Beta (KL weight): 1.0
   - Spatial loss weight: 1.0

### Code Organization
1. Modular Structure
   - `src/models/`: Neural network architectures
   - `src/training/`: Training loops and loss functions
   - `src/generation/`: Level generation utilities
   - `src/utils/`: Helper functions and data processing
   - `src/config/`: Configuration management

2. Utility Functions
   - Grid size calculation
   - Position assignment
   - Coordinate system conversion
   - Parameter validation

### Performance Metrics
1. Level Generation
   - Time limits matching original data distribution
   - Goal counts within expected ranges
   - Grid sizes appropriate for difficulty levels
   - Spatial distribution matching Unity requirements

2. Training Metrics
   - Decreasing loss values
   - Stable KL divergence
   - Consistent validation performance
   - Early stopping prevention of overfitting

### Known Issues
1. Grid Size Variance
   - Generated grid sizes sometimes smaller than original
   - Need to adjust grid size calculation parameters

2. Position Distribution
   - Random offsets might need tuning
   - Grid cell utilization could be improved

### Future Improvements
1. Model Architecture
   - Experiment with different embedding sizes
   - Add attention mechanisms
   - Try hierarchical VAE structure

2. Training Process
   - Implement curriculum learning
   - Add data augmentation
   - Experiment with different loss weightings

3. Generation Features
   - Add more control over generation parameters
   - Implement constraint-based generation
   - Add level validation checks 