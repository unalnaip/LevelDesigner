import torch
import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path

from ..utils.difficulty_scaling import DifficultyScaler
from ..config.model_config import GenerationConfig, ValidationConfig

class LevelGenerator:
    def __init__(self, model, config_path: str = "config/generation_config.json"):
        self.model = model
        self.config = GenerationConfig.load(config_path)
        self.validator = ValidationConfig()
        self.difficulty_scaler = DifficultyScaler()
        
    def generate_batch(self, start_level: int, count: int, 
                      output_path: Optional[str] = None) -> List[Dict]:
        """Generate a batch of levels with progressive difficulty"""
        levels = []
        
        for level_num in range(start_level, start_level + count):
            level = self.generate_single_level(level_num)
            if level is not None:
                levels.append(level)
        
        if output_path:
            self._save_levels(levels, output_path)
        
        return levels
    
    def generate_single_level(self, level_num: int) -> Optional[Dict]:
        """Generate a single level with appropriate difficulty"""
        # Get difficulty parameters for this level
        tier = self.difficulty_scaler.get_tier(level_num)
        params = self.difficulty_scaler.calculate_goal_distribution(level_num)
        
        # Generate level layout
        z = torch.randn(1, self.config.latent_dim)
        with torch.no_grad():
            layout = self.model.decode(z)
        
        # Convert layout to level data
        level_data = self._layout_to_level(layout, params, level_num)
        
        # Calculate time limit
        time_limit = self.difficulty_scaler.calculate_time_limit(
            params["goal_count"], 
            params["num_types"],
            params["blocker_count"], 
            tier
        )
        
        # Add metadata
        level_data["time_limit"] = time_limit
        level_data["level_number"] = level_num
        level_data["difficulty"] = self.difficulty_scaler.get_difficulty_label(
            params["goal_count"],
            params["blocker_count"],
            time_limit,
            tier
        )
        
        # Validate level
        if not self.validator.validate_level(level_data):
            return None
            
        return level_data
    
    def _layout_to_level(self, layout: torch.Tensor, 
                        params: Dict, level_num: int) -> Dict:
        """Convert model output to level format"""
        # Extract layout parameters
        positions = layout[0, :params["goal_count"], :2].numpy()
        
        # Select goal types based on difficulty
        available_types = self.config.goal_types[:params["num_types"]]
        type_weights = [self.config.type_weights[t] for t in available_types]
        type_weights = np.array(type_weights) / sum(type_weights)
        
        # Assign goal types
        goal_types = np.random.choice(
            available_types,
            size=params["goal_count"],
            p=type_weights
        )
        
        # Create level data
        objects = []
        for pos, obj_type in zip(positions, goal_types):
            objects.append({
                "type": obj_type,
                "position": pos.tolist(),
                "properties": {
                    "t": False,  # Can be toggled
                    "a": False,  # Is active
                    "r": False   # Is required
                }
            })
        
        # Add blockers
        if params["blocker_count"] > 0:
            blocker_positions = self._generate_blocker_positions(
                positions, 
                params["blocker_count"]
            )
            blocker_types = np.random.choice(
                self.config.blocker_types,
                size=params["blocker_count"]
            )
            
            for pos, block_type in zip(blocker_positions, blocker_types):
                objects.append({
                    "type": block_type,
                    "position": pos.tolist(),
                    "properties": {
                        "t": True,   # Blockers are typically toggleable
                        "a": True,   # Start active
                        "r": False   # Not required
                    }
                })
        
        return {
            "objects": objects,
            "metadata": {
                "generation_time": datetime.now().isoformat()
            }
        }
    
    def _generate_blocker_positions(self, 
                                  goal_positions: np.ndarray,
                                  num_blockers: int) -> np.ndarray:
        """Generate valid positions for blockers"""
        positions = []
        grid_size = 10  # Assuming 10x10 grid
        
        for _ in range(num_blockers):
            valid = False
            attempts = 0
            
            while not valid and attempts < 100:
                pos = np.random.rand(2) * grid_size
                
                # Check minimum distance from goals
                distances = np.linalg.norm(
                    goal_positions - pos.reshape(1, 2),
                    axis=1
                )
                
                if np.min(distances) > self.validator.min_goal_spacing:
                    valid = True
                    positions.append(pos)
                
                attempts += 1
        
        return np.array(positions)
    
    def _save_levels(self, levels: List[Dict], output_path: str):
        """Save generated levels to file"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "levels": levels,
                "metadata": {
                    "generator_version": "2.0",
                    "generation_time": datetime.now().isoformat(),
                    "config": self.config.__dict__
                }
            }, f, indent=2)
    
    def format_for_unity(self, levels: List[Dict]) -> List[Dict]:
        """Convert level format to Unity-compatible format"""
        unity_levels = []
        
        for level in levels:
            unity_level = {
                "i": level["level_number"],
                "d": level["time_limit"],
                "t": 1 if level["difficulty"] == "none" else (
                    2 if level["difficulty"] == "hard" else 3
                ),
                "l": []
            }
            
            # Convert objects to Unity format
            for obj in level["objects"]:
                unity_obj = {
                    "t": obj["type"],
                    "p": obj["position"],
                    "properties": obj["properties"]
                }
                unity_level["l"].append(unity_obj)
            
            unity_levels.append(unity_level)
        
        return unity_levels 