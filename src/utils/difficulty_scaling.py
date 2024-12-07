from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class DifficultyTier:
    min_level: int
    max_level: int
    goal_count_range: Tuple[int, int]
    goal_type_range: Tuple[int, int]
    blocker_probability: float
    time_multiplier: float
    label: str

class DifficultyScaler:
    def __init__(self):
        # Derived from benchmark analysis
        self.tiers = [
            DifficultyTier(1, 50, (3, 9), (1, 3), 0.2, 1.2, "none"),
            DifficultyTier(51, 150, (12, 21), (2, 4), 0.4, 1.0, "none"),
            DifficultyTier(151, 300, (15, 24), (3, 5), 0.6, 0.9, "hard"),
            DifficultyTier(301, 500, (18, 30), (3, 6), 0.7, 0.8, "hard"),
            DifficultyTier(501, 1000, (21, 36), (4, 6), 0.8, 0.7, "superhard")
        ]
    
    def get_tier(self, level_num: int) -> DifficultyTier:
        for tier in self.tiers:
            if tier.min_level <= level_num <= tier.max_level:
                return tier
        return self.tiers[-1]  # Default to highest tier
    
    def calculate_time_limit(self, goal_count: int, goal_types: int, 
                           blocker_count: int, tier: DifficultyTier) -> int:
        """Calculate time limit based on level complexity"""
        base_time = 120  # Base time from benchmark analysis
        goal_time = goal_count * 5  # 5 seconds per goal
        type_time = goal_types * 10  # 10 seconds per goal type
        blocker_penalty = blocker_count * -3  # -3 seconds per blocker
        
        total_time = (base_time + goal_time + type_time + blocker_penalty) * tier.time_multiplier
        return max(int(total_time), 90)  # Minimum 90 seconds from benchmarks
    
    def calculate_goal_distribution(self, level_num: int) -> Dict[str, int]:
        """Calculate goal counts and types based on level number"""
        tier = self.get_tier(level_num)
        
        # Progressive scaling within tier
        tier_progress = (level_num - tier.min_level) / (tier.max_level - tier.min_level)
        
        goal_count = np.random.randint(
            tier.goal_count_range[0],
            int(tier.goal_count_range[0] + (tier.goal_count_range[1] - tier.goal_count_range[0]) * tier_progress)
        )
        
        num_types = np.random.randint(tier.goal_type_range[0], tier.goal_type_range[1])
        
        return {
            "goal_count": goal_count,
            "num_types": num_types,
            "blocker_count": int(np.random.binomial(goal_count // 3, tier.blocker_probability))
        }

    def get_difficulty_label(self, goal_count: int, blocker_count: int, 
                           time_limit: int, tier: DifficultyTier) -> str:
        """Determine difficulty label based on level parameters"""
        if goal_count >= 21 and blocker_count >= 2 and time_limit <= 120:
            return "hard"
        if goal_count >= 30 and blocker_count >= 3 and time_limit <= 150:
            return "superhard"
        return tier.label 