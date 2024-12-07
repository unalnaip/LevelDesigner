from enum import Enum
from typing import List, Tuple, Set, Dict
from collections import deque
import numpy as np
from dataclasses import dataclass

class Strategy(Enum):
    GREEDY = "greedy"
    BALANCED = "balanced"
    STRATEGIC = "strategic"

@dataclass
class SimulationResult:
    """Stores results of a simulation run"""
    success: bool
    moves_taken: int
    objects_collected: List[str]
    triples_formed: int
    strategy: Strategy

class PlayerSimulation:
    """Simulates player movement and strategies to test level solvability"""
    
    def __init__(self, num_simulations: int = 100):
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.num_simulations = num_simulations
        
    def simulate_strategies(self, grid: np.ndarray) -> Dict[Strategy, List[SimulationResult]]:
        """Run simulations with different strategies"""
        results = {
            Strategy.GREEDY: [],
            Strategy.BALANCED: [],
            Strategy.STRATEGIC: []
        }
        
        for strategy in Strategy:
            for _ in range(self.num_simulations):
                result = self._run_simulation(grid.copy(), strategy)
                results[strategy].append(result)
                
        return results
    
    def _run_simulation(self, grid: np.ndarray, strategy: Strategy) -> SimulationResult:
        """Run a single simulation with given strategy"""
        moves = 0
        objects_collected = []
        triples = 0
        current_pos = tuple(p[0] for p in np.where(grid == 'S'))
        
        while moves < 1000:  # Prevent infinite loops
            next_move = self._get_next_move(grid, current_pos, strategy)
            if not next_move:
                break
                
            current_pos = next_move
            moves += 1
            
            # Collect objects and form triples based on strategy
            collected = self._collect_objects(grid, current_pos, strategy)
            objects_collected.extend(collected)
            
            new_triples = self._form_triples(objects_collected, strategy)
            triples += new_triples
            
            # Check win condition
            if self._check_win_condition(grid, objects_collected, triples, current_pos):
                return SimulationResult(True, moves, objects_collected, triples, strategy)
                
        return SimulationResult(False, moves, objects_collected, triples, strategy)
    
    def _get_next_move(self, grid: np.ndarray, pos: Tuple[int, int], strategy: Strategy) -> Tuple[int, int]:
        """Determine next move based on strategy"""
        if strategy == Strategy.GREEDY:
            return self._get_greedy_move(grid, pos)
        elif strategy == Strategy.BALANCED:
            return self._get_balanced_move(grid, pos)
        else:  # STRATEGIC
            return self._get_strategic_move(grid, pos)
            
    def _get_greedy_move(self, grid: np.ndarray, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Always move toward nearest collectible object"""
        objects = self._find_nearby_objects(grid, pos)
        if not objects:
            return None
        return self._path_to_nearest(grid, pos, objects[0])
        
    def _get_balanced_move(self, grid: np.ndarray, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Try to maintain a mix of different objects"""
        objects = self._find_nearby_objects(grid, pos)
        if not objects:
            return None
            
        # Consider object variety when choosing target
        current_inventory = self._get_current_inventory()
        target = self._choose_balanced_target(objects, current_inventory, grid)
        return self._path_to_nearest(grid, pos, target)
        
    def _get_strategic_move(self, grid: np.ndarray, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Save certain objects for larger combos"""
        objects = self._find_nearby_objects(grid, pos)
        if not objects:
            return None
            
        # Look for potential combo opportunities
        combo_target = self._find_combo_opportunity(objects, grid)
        if combo_target:
            return self._path_to_nearest(grid, pos, combo_target)
            
        return self._get_balanced_move(grid, pos)  # Fall back to balanced strategy
        
    def _find_nearby_objects(self, grid: np.ndarray, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find nearby objects in the grid"""
        objects = []
        for dx, dy in self.directions:
            next_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_move(grid, next_pos):
                objects.append(next_pos)
        return objects
        
    def _path_to_nearest(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[int, int]:
        """Find path from start to goal using BFS"""
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            pos, path = queue.popleft()
            
            if pos == goal:
                return path[-1]
                
            for dx, dy in self.directions:
                next_pos = (pos[0] + dx, pos[1] + dy)
                
                if self._is_valid_move(grid, next_pos) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
                    
        return None
        
    def _is_valid_move(self, grid: np.ndarray, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid for movement"""
        if not (0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1]):
            return False
            
        return grid[pos] != '#'  # Assuming '#' represents walls/obstacles 
        
    def _collect_objects(self, grid: np.ndarray, pos: Tuple[int, int], strategy: Strategy) -> List[str]:
        """Collect objects at the current position"""
        cell_content = grid[pos]
        if cell_content in ['S', 'G', '#', ' ']:  # Skip non-collectible cells
            return []
            
        # Clear the cell after collection
        grid[pos] = ' '
        return [cell_content]
        
    def _form_triples(self, objects: List[str], strategy: Strategy) -> int:
        """
        Form triples based on strategy
        Returns number of new triples formed
        """
        if len(objects) < 3:
            return 0
            
        triples_formed = 0
        remaining_objects = objects.copy()
        
        if strategy == Strategy.GREEDY:
            # Form triples as soon as possible
            while len(remaining_objects) >= 3:
                triple = self._find_first_triple(remaining_objects)
                if not triple:
                    break
                    
                for obj in triple:
                    remaining_objects.remove(obj)
                triples_formed += 1
                
        elif strategy == Strategy.BALANCED:
            # Try to maintain variety while forming triples
            while len(remaining_objects) >= 3:
                triple = self._find_balanced_triple(remaining_objects)
                if not triple:
                    break
                    
                for obj in triple:
                    remaining_objects.remove(obj)
                triples_formed += 1
                
        else:  # STRATEGIC
            # Look for larger combo opportunities
            while len(remaining_objects) >= 3:
                triple = self._find_strategic_triple(remaining_objects)
                if not triple:
                    break
                    
                for obj in triple:
                    remaining_objects.remove(obj)
                triples_formed += 1
                
        # Update the objects list to remove used objects
        objects.clear()
        objects.extend(remaining_objects)
        
        return triples_formed
        
    def _get_current_inventory(self) -> Dict[str, int]:
        """Track current inventory of collected objects"""
        if not hasattr(self, '_inventory'):
            self._inventory = {}
        return self._inventory
        
    def _find_first_triple(self, objects: List[str]) -> List[str]:
        """Find first possible triple (for greedy strategy)"""
        counts = {}
        for obj in objects:
            counts[obj] = counts.get(obj, 0) + 1
            if counts[obj] >= 3:
                return [obj] * 3
        return []
        
    def _find_balanced_triple(self, objects: List[str]) -> List[str]:
        """Find triple while maintaining object variety"""
        counts = {}
        for obj in objects:
            counts[obj] = counts.get(obj, 0) + 1
            
        # Prefer using objects we have most of
        sorted_objects = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if sorted_objects[0][1] >= 3:
            return [sorted_objects[0][0]] * 3
        return []
        
    def _find_strategic_triple(self, objects: List[str]) -> List[str]:
        """Find triple considering potential future combos"""
        counts = {}
        for obj in objects:
            counts[obj] = counts.get(obj, 0) + 1
            
        # Look for objects that could form multiple triples
        potential_combos = {
            obj: count // 3 
            for obj, count in counts.items()
        }
        
        # If we can save objects for a larger combo, do so
        best_combo = max(potential_combos.items(), key=lambda x: x[1])
        if best_combo[1] >= 2:  # Can form at least 2 triples
            return [best_combo[0]] * 3
            
        # Fall back to balanced strategy
        return self._find_balanced_triple(objects)
        
    def _find_combo_opportunity(self, objects: List[Tuple[int, int]], grid: np.ndarray) -> Tuple[int, int]:
        """Find object position that could lead to combos"""
        object_positions = {}
        
        # Group object positions by type
        for pos in objects:
            obj_type = grid[pos]
            if obj_type not in object_positions:
                object_positions[obj_type] = []
            object_positions[obj_type].append(pos)
            
        # Look for object types with multiple instances
        for obj_type, positions in object_positions.items():
            if len(positions) >= 2:  # Potential for combo
                return positions[0]  # Return first position of potential combo
                
        return None
        
    def _check_win_condition(self, grid: np.ndarray, objects: List[str], triples: int, current_pos: Tuple[int, int]) -> bool:
        """Check if level is completed"""
        # Win conditions:
        # 1. Reached goal position
        # 2. Formed required number of triples
        # 3. Collected all required objects
        
        goal_positions = np.where(grid == 'G')
        if len(goal_positions[0]) == 0:  # No goal found
            return False
        
        goal_pos = (goal_positions[0][0], goal_positions[1][0])  # Take first goal position
        at_goal = current_pos[0] == goal_pos[0] and current_pos[1] == goal_pos[1]
        required_triples = 3  # Default to 3 if not specified
        
        return (
            at_goal and 
            triples >= required_triples and 
            not self._has_required_objects_remaining(grid)
        )
        
    def _has_required_objects_remaining(self, grid: np.ndarray) -> bool:
        """Check if there are still required objects to collect"""
        for row in grid:
            for cell in row:
                if cell not in ['S', 'G', '#', ' ', 'P']:  # If any collectible objects remain
                    return True
        return False
        
    def _choose_balanced_target(self, objects: List[Tuple[int, int]], inventory: Dict[str, int], grid: np.ndarray) -> Tuple[int, int]:
        """Choose target object to maintain a balanced inventory"""
        object_scores = {}
        
        for pos in objects:
            obj_type = grid[pos]
            current_count = inventory.get(obj_type, 0)
            
            # Score based on how much we need this object type
            # Lower score = more desirable
            score = current_count + (self._count_nearby(grid, pos, obj_type) * 0.5)
            object_scores[pos] = score
            
        return min(object_scores.items(), key=lambda x: x[1])[0]
        
    def _count_nearby(self, grid: np.ndarray, pos: Tuple[int, int], obj_type: str, radius: int = 2) -> int:
        """Count number of similar objects nearby"""
        count = 0
        x, y = pos
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                new_pos = (x + dx, y + dy)
                if self._is_valid_move(grid, new_pos) and grid[new_pos] == obj_type:
                    count += 1
                    
        return count

    def generate_strategy_report(self, results: Dict[Strategy, List[SimulationResult]]) -> str:
        """Generate detailed report of strategy performance"""
        report = "Strategy Performance Report\n" + "=" * 25 + "\n\n"
        
        for strategy, sim_results in results.items():
            successes = [r for r in sim_results if r.success]
            success_rate = len(successes) / len(sim_results) * 100
            
            if successes:
                avg_moves = sum(r.moves_taken for r in successes) / len(successes)
                avg_triples = sum(r.triples_formed for r in successes) / len(successes)
            else:
                avg_moves = float('inf')
                avg_triples = 0
                
            report += f"Strategy: {strategy.value}\n"
            report += f"Success Rate: {success_rate:.1f}%\n"
            report += f"Average Moves: {avg_moves:.1f}\n"
            report += f"Average Triples Formed: {avg_triples:.1f}\n"
            report += f"Total Simulations: {len(sim_results)}\n"
            report += "-" * 25 + "\n"
            
        return report