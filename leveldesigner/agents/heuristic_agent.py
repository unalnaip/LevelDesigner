from typing import List, Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from .player_simulation import PlayerSimulation

@dataclass
class LevelMetrics:
    """Stores metrics calculated for a level"""
    goal_objects_count: Dict[str, int]
    path_length: int
    obstacle_density: float
    estimated_difficulty: float

class HeuristicAgent:
    """Agent that performs heuristic checks on generated levels"""
    
    def __init__(self, difficulty_model: LinearRegression = None):
        """
        Initialize the HeuristicAgent
        
        Args:
            difficulty_model: Trained regression model for difficulty estimation
        """
        self.difficulty_model = difficulty_model
        self.player_sim = PlayerSimulation()
        
    def check_solvability(self, level_data: Dict[str, Any]) -> bool:
        """
        Check if a level contains required goal objects and is potentially solvable
        
        Args:
            level_data: Dictionary containing level layout and metadata
            
        Returns:
            bool: True if level passes basic solvability checks
        """
        # First check required objects
        grid = level_data['layout']
        required_objects = level_data.get('required_objects', {})
        
        object_counts = self._count_objects(grid)
        for obj, required_count in required_objects.items():
            if object_counts.get(obj, 0) < required_count:
                return False
        
        # Then check if level is solvable using player simulation
        strategy_results = self.player_sim.simulate_strategies(grid)
        
        # If any strategy can solve it, it's considered solvable
        for results in strategy_results.values():
            if any(r.success for r in results):
                return True
                
        return False
    
    def estimate_difficulty(self, level_data: Dict[str, Any]) -> float:
        """
        Estimate level difficulty using the regression model
        
        Args:
            level_data: Dictionary containing level layout and metadata
            
        Returns:
            float: Estimated difficulty score (0-1)
        """
        features = self._extract_features(level_data)
        
        if self.difficulty_model is None:
            # Use simple heuristic if no model is provided
            return self._basic_difficulty_estimate(features)
            
        return self.difficulty_model.predict([features])[0]
    
    def filter_levels(self, 
                     levels: List[Dict[str, Any]], 
                     target_difficulty: float,
                     difficulty_tolerance: float = 0.2) -> List[Dict[str, Any]]:
        """
        Filter levels based on solvability and difficulty alignment
        
        Args:
            levels: List of generated level dictionaries
            target_difficulty: Desired difficulty (0-1)
            difficulty_tolerance: Acceptable deviation from target difficulty
            
        Returns:
            List of levels that pass the filtering criteria
        """
        filtered_levels = []
        
        for level in levels:
            # Check basic solvability
            if not self.check_solvability(level):
                continue
                
            # Check difficulty alignment
            difficulty = self.estimate_difficulty(level)
            if abs(difficulty - target_difficulty) <= difficulty_tolerance:
                level['estimated_difficulty'] = difficulty
                filtered_levels.append(level)
                
        return filtered_levels
    
    def _count_objects(self, grid: np.ndarray) -> Dict[str, int]:
        """Count occurrences of each object type in the level grid"""
        unique, counts = np.unique(grid, return_counts=True)
        return dict(zip(unique, counts))
    
    def _extract_features(self, level_data: Dict[str, Any]) -> List[float]:
        """Extract relevant features for difficulty estimation"""
        grid = level_data['layout']
        
        features = [
            self._calculate_obstacle_density(grid),
            self._estimate_path_length(grid),
            len(self._count_objects(grid))
        ]
        
        return features
    
    def _basic_difficulty_estimate(self, features: List[float]) -> float:
        """Simple difficulty estimation when no model is available"""
        # Normalize and combine features with basic weights
        obstacle_density = features[0]
        path_length = features[1] / 100  # Normalize
        object_variety = features[2] / 10  # Normalize
        
        difficulty = (
            0.4 * obstacle_density +
            0.4 * path_length +
            0.2 * object_variety
        )
        
        return np.clip(difficulty, 0, 1)
    
    def _calculate_obstacle_density(self, grid: np.ndarray) -> float:
        """Calculate the density of obstacles in the level"""
        total_cells = grid.size
        obstacle_cells = np.sum(grid != 0)  # Assuming 0 is empty space
        return obstacle_cells / total_cells
    
    def _estimate_path_length(self, grid: np.ndarray) -> int:
        """Estimate the minimum path length from start to goal"""
        _, path = self.player_sim.is_solvable(grid)
        return len(path) if path else 0