from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from .heuristic_agent import HeuristicAgent
from .player_simulation import PlayerSimulation, Strategy, SimulationResult

@dataclass
class LevelEvaluation:
    """Complete evaluation results for a level"""
    level_data: Dict[str, Any]
    difficulty_score: float
    strategy_results: Dict[Strategy, List[SimulationResult]]
    success_rates: Dict[Strategy, float]
    average_moves: Dict[Strategy, float]
    overall_success_rate: float

class LevelEvaluator:
    """Combines heuristic checks and player simulation for complete level evaluation"""
    
    def __init__(self, difficulty_model=None, min_success_rate: float = 0.3):
        self.heuristic_agent = HeuristicAgent(difficulty_model)
        self.player_sim = PlayerSimulation()
        self.min_success_rate = min_success_rate
        
    def evaluate_levels(self, 
                       levels: List[Dict[str, Any]], 
                       target_difficulty: float,
                       num_candidates: int = 20) -> List[LevelEvaluation]:
        """
        Evaluate levels through both heuristic and simulation testing
        
        Args:
            levels: List of generated levels
            target_difficulty: Desired difficulty (0-1)
            num_candidates: Number of levels to return
            
        Returns:
            List of evaluated levels that pass all criteria
        """
        # First apply heuristic filtering
        filtered_levels = self.heuristic_agent.filter_levels(
            levels, 
            target_difficulty
        )
        
        # Then simulate gameplay for remaining levels
        evaluated_levels = []
        for level in filtered_levels:
            evaluation = self._evaluate_single_level(level)
            if evaluation is not None:  # Only add levels that pass minimum success rate for ALL strategies
                evaluated_levels.append(evaluation)
                
        # Sort by closest to target difficulty and take top candidates
        evaluated_levels.sort(
            key=lambda x: abs(x.difficulty_score - target_difficulty)
        )
        
        return evaluated_levels[:num_candidates]
        
    def _evaluate_single_level(self, level_data: Dict[str, Any]) -> LevelEvaluation:
        """Run complete evaluation on a single level"""
        # Get difficulty score
        difficulty_score = self.heuristic_agent.estimate_difficulty(level_data)
        
        # Run simulations with different strategies
        strategy_results = self.player_sim.simulate_strategies(level_data['layout'])
        
        # Calculate success rates and average moves
        success_rates = {}
        average_moves = {}
        
        for strategy, results in strategy_results.items():
            successes = [r for r in results if r.success]
            success_rates[strategy] = len(successes) / len(results)
            average_moves[strategy] = np.mean([r.moves_taken for r in successes]) if successes else float('inf')
            
        # Check if ANY strategy has less than minimum success rate
        min_strategy_rate = min(success_rates.values())
        if min_strategy_rate < self.min_success_rate:
            return None  # Level fails if any strategy has low success rate
            
        overall_success_rate = np.mean(list(success_rates.values()))
        
        return LevelEvaluation(
            level_data=level_data,
            difficulty_score=difficulty_score,
            strategy_results=strategy_results,
            success_rates=success_rates,
            average_moves=average_moves,
            overall_success_rate=overall_success_rate
        ) 
        
    def evaluate_and_report(self, 
                          levels: List[Dict[str, Any]], 
                          target_difficulty: float,
                          num_candidates: int = 20) -> Tuple[List[LevelEvaluation], str]:
        """
        Evaluate levels and generate a detailed report
        
        Returns:
            Tuple of (filtered_levels, report_string)
        """
        evaluated_levels = self.evaluate_levels(levels, target_difficulty, num_candidates)
        
        report = "Level Evaluation Report\n" + "=" * 25 + "\n\n"
        report += f"Initial Levels: {len(levels)}\n"
        report += f"Levels Passing Heuristics: {len(evaluated_levels)}\n"
        report += f"Target Difficulty: {target_difficulty}\n\n"
        
        for i, eval in enumerate(evaluated_levels, 1):
            report += f"Level {i}:\n"
            report += f"Difficulty Score: {eval.difficulty_score:.2f}\n"
            report += f"Overall Success Rate: {eval.overall_success_rate*100:.1f}%\n"
            
            # Add strategy-specific results
            for strategy, rate in eval.success_rates.items():
                avg_moves = eval.average_moves[strategy]
                report += f"  {strategy.value}: {rate*100:.1f}% success "
                report += f"({avg_moves:.1f} avg moves)\n"
            
            report += "-" * 25 + "\n"
            
        return evaluated_levels, report 