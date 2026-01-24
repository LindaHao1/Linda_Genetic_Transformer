"""
Parallel Simulated Annealing optimizer for pixelated device design.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from pathlib import Path
import time
from datetime import datetime
import json

from simulation_engine import SimulationEngine, SimulationJob, SimulationResult
from .cost_function import calculate_filter_cost


class SimulatedAnnealingOptimizer:
    """
    Parallel Simulated Annealing optimizer with hybrid exploration strategy.
    
    In each iteration, performs parallel simulations:
    - 1 completely random design (global exploration)
    - N-1 neighbor designs (local exploration - flip K pixels)
    
    This enables efficient parallel hardware utilization while maintaining
    good exploration-exploitation balance.
    """
    
    def __init__(
        self,
        simulation_engine: SimulationEngine,
        rows: int = 21,
        cols: int = 30,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.95,
        min_temp: float = 0.01,
        parallel_sims: int = 4,
        num_random: int = 1,
        num_flips: int = 10,
        connectivity_check: Optional[Callable] = None,
        output_dir: str = "./optimization_results"
    ):
        """
        Initialize the optimizer.
        
        Args:
            simulation_engine: Configured SimulationEngine instance
            rows, cols: Grid dimensions
            initial_temp: Starting temperature for SA
            cooling_rate: Temperature decay factor (0 < rate < 1)
            min_temp: Minimum temperature (stopping criterion)
            parallel_sims: Number of parallel simulations per iteration
            num_random: Number of random designs per iteration (rest are neighbors)
            num_flips: Number of pixels to flip for neighbor generation
            connectivity_check: Optional function to validate designs
            output_dir: Directory to save optimization results
        """
        self.engine = simulation_engine
        self.rows = rows
        self.cols = cols
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.parallel_sims = parallel_sims
        self.num_random = num_random
        self.num_neighbors = parallel_sims - num_random
        self.num_flips = num_flips
        self.connectivity_check = connectivity_check
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.current_temp = initial_temp
        self.iteration = 0
        self.best_cost = float('inf')
        self.best_matrix = None
        self.current_matrix = None
        self.current_cost = float('inf')
        
        # History
        self.history = {
            'iteration': [],
            'temperature': [],
            'current_cost': [],
            'best_cost': [],
            'accepted': [],
            'acceptance_prob': []
        }
        
        print(f"🔥 Simulated Annealing Optimizer initialized")
        print(f"   Parallel simulations: {parallel_sims} ({num_random} random + {self.num_neighbors} neighbors)")
        print(f"   Temperature: {initial_temp} → {min_temp} (cooling: {cooling_rate})")
        print(f"   Neighbor strategy: flip {num_flips} pixels")
    
    def generate_random_matrix(self) -> np.ndarray:
        """Generate a completely random design matrix."""
        return np.random.randint(0, 2, size=(self.rows, self.cols))
    
    def generate_neighbor(self, matrix: np.ndarray, num_flips: int) -> np.ndarray:
        """
        Generate a neighbor by flipping num_flips random pixels.
        
        Args:
            matrix: Current design matrix
            num_flips: Number of pixels to flip
            
        Returns:
            Neighbor design matrix
        """
        neighbor = matrix.copy()
        
        # Randomly select positions to flip
        total_pixels = self.rows * self.cols
        flip_indices = np.random.choice(total_pixels, size=num_flips, replace=False)
        
        for idx in flip_indices:
            row = idx // self.cols
            col = idx % self.cols
            neighbor[row, col] = 1 - neighbor[row, col]  # Flip bit
        
        return neighbor
    
    def validate_design(self, matrix: np.ndarray, max_attempts: int = 10) -> np.ndarray:
        """
        Validate design and regenerate if it fails connectivity check.
        
        Args:
            matrix: Design matrix to validate
            max_attempts: Maximum regeneration attempts
            
        Returns:
            Valid design matrix
        """
        if self.connectivity_check is None:
            return matrix
        
        for attempt in range(max_attempts):
            if self.connectivity_check(matrix):
                return matrix
            # Regenerate by flipping some pixels
            matrix = self.generate_neighbor(matrix, self.num_flips)
        
        # If still invalid, return anyway (will get high cost)
        print(f"  ⚠️  Warning: Could not generate valid design after {max_attempts} attempts")
        return matrix
    
    def acceptance_probability(self, current_cost: float, new_cost: float, temp: float) -> float:
        """
        Calculate acceptance probability using Metropolis criterion.
        
        Always accept if new solution is better (lower cost).
        Accept worse solutions with probability exp(-ΔE/T).
        
        Args:
            current_cost: Cost of current solution
            new_cost: Cost of candidate solution
            temp: Current temperature
            
        Returns:
            Acceptance probability [0, 1]
        """
        if new_cost < current_cost:
            return 1.0
        
        if temp <= 0:
            return 0.0
        
        delta = new_cost - current_cost
        return np.exp(-delta / temp)
    
    def evaluate_design(self, result: SimulationResult) -> Tuple[float, str]:
        """
        Evaluate a simulation result and calculate cost.
        
        Args:
            result: SimulationResult from simulation engine
            
        Returns:
            Tuple of (cost, s2p_path)
        """
        if not result.success:
            return float('inf'), None
        
        # Find the s2p file in the analysis directory
        snp_dir = Path(result.snp_path).parent / result.metadata['device_name']
        s2p_path = snp_dir / "network_2port.s2p"
        
        if not s2p_path.exists():
            print(f"  ⚠️  Warning: S2P file not found for {result.job_id}")
            return float('inf'), None
        
        cost = calculate_filter_cost(str(s2p_path))
        return cost, str(s2p_path)
    
    def run_iteration(self) -> bool:
        """
        Run one iteration of parallel simulated annealing.
        
        Returns:
            True if optimization should continue, False if converged
        """
        self.iteration += 1
        print(f"\n{'='*80}")
        print(f"🔄 Iteration {self.iteration} | Temp: {self.current_temp:.4f} | Best Cost: {self.best_cost:.4f}")
        print(f"{'='*80}")
        
        # Generate candidate designs
        jobs = []
        
        # 1. Generate random designs (global exploration)
        for i in range(self.num_random):
            matrix = self.generate_random_matrix()
            if self.connectivity_check:
                matrix = self.validate_design(matrix)
            
            job = SimulationJob(
                job_id=f"iter{self.iteration}_random{i+1}",
                design_matrix=matrix,
                metadata={
                    'iteration': self.iteration,
                    'type': 'random',
                    'temperature': self.current_temp
                }
            )
            jobs.append(job)
        
        # 2. Generate neighbor designs (local exploration)
        if self.current_matrix is not None:
            for i in range(self.num_neighbors):
                matrix = self.generate_neighbor(self.current_matrix, self.num_flips)
                if self.connectivity_check:
                    matrix = self.validate_design(matrix)
                
                job = SimulationJob(
                    job_id=f"iter{self.iteration}_neighbor{i+1}",
                    design_matrix=matrix,
                    metadata={
                        'iteration': self.iteration,
                        'type': 'neighbor',
                        'temperature': self.current_temp,
                        'num_flips': self.num_flips
                    }
                )
                jobs.append(job)
        else:
            # First iteration: all random
            for i in range(self.num_neighbors):
                matrix = self.generate_random_matrix()
                if self.connectivity_check:
                    matrix = self.validate_design(matrix)
                
                job = SimulationJob(
                    job_id=f"iter{self.iteration}_random{i+1+self.num_random}",
                    design_matrix=matrix,
                    metadata={
                        'iteration': self.iteration,
                        'type': 'random_init',
                        'temperature': self.current_temp
                    }
                )
                jobs.append(job)
        
        # Run parallel simulations
        print(f"\n🚀 Submitting {len(jobs)} parallel simulations...")
        start_time = time.time()
        results = self.engine.submit_batch(jobs)
        sim_time = time.time() - start_time
        print(f"⏱️  Simulation batch completed in {sim_time:.1f}s")
        
        # Evaluate all results
        print(f"\n📊 Evaluating results...")
        candidates = []
        for result in results:
            cost, s2p_path = self.evaluate_design(result)
            candidates.append({
                'job_id': result.job_id,
                'cost': cost,
                'matrix': result.metadata.get('design_matrix', 
                          np.load(result.matrix_path) if result.matrix_path else None),
                'type': result.metadata.get('type', 'unknown'),
                's2p_path': s2p_path,
                'result': result
            })
            print(f"  {result.job_id}: cost = {cost:.4f} (type: {result.metadata.get('type', 'unknown')})")
        
        # Find best candidate from this batch
        best_candidate = min(candidates, key=lambda x: x['cost'])
        
        # Update global best
        if best_candidate['cost'] < self.best_cost:
            self.best_cost = best_candidate['cost']
            self.best_matrix = best_candidate['matrix']
            print(f"\n🎉 New best design found! Cost: {self.best_cost:.4f}")
            
            # Save best design
            best_file = self.output_dir / f"best_iter{self.iteration}.npy"
            np.save(best_file, self.best_matrix)
        
        # Simulated Annealing acceptance decision
        if self.current_matrix is None:
            # First iteration: accept best candidate
            self.current_matrix = best_candidate['matrix']
            self.current_cost = best_candidate['cost']
            accepted = True
            accept_prob = 1.0
        else:
            accept_prob = self.acceptance_probability(
                self.current_cost, 
                best_candidate['cost'], 
                self.current_temp
            )
            accepted = np.random.random() < accept_prob
            
            if accepted:
                self.current_matrix = best_candidate['matrix']
                self.current_cost = best_candidate['cost']
                print(f"✅ Accepted new solution (p={accept_prob:.4f})")
            else:
                print(f"❌ Rejected new solution (p={accept_prob:.4f})")
        
        # Update history
        self.history['iteration'].append(self.iteration)
        self.history['temperature'].append(self.current_temp)
        self.history['current_cost'].append(self.current_cost)
        self.history['best_cost'].append(self.best_cost)
        self.history['accepted'].append(accepted)
        self.history['acceptance_prob'].append(accept_prob)
        
        # Cool down temperature
        self.current_temp *= self.cooling_rate
        
        # Check convergence
        if self.current_temp < self.min_temp:
            print(f"\n❄️  Temperature below minimum ({self.min_temp}). Optimization complete.")
            return False
        
        return True
    
    def optimize(self, max_iterations: Optional[int] = None) -> dict:
        """
        Run the optimization process.
        
        Args:
            max_iterations: Maximum number of iterations (None = run until convergence)
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n🎯 Starting Simulated Annealing Optimization")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Run optimization loop
        while True:
            if max_iterations and self.iteration >= max_iterations:
                print(f"\n🛑 Reached maximum iterations ({max_iterations})")
                break
            
            should_continue = self.run_iteration()
            
            if not should_continue:
                break
            
            # Save checkpoint
            self.save_checkpoint()
        
        total_time = time.time() - start_time
        
        # Final results
        print(f"\n{'='*80}")
        print(f"🏁 Optimization Complete")
        print(f"{'='*80}")
        print(f"Total iterations: {self.iteration}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Best cost: {self.best_cost:.4f}")
        print(f"Final temperature: {self.current_temp:.4f}")
        
        # Save final results
        self.save_results()
        
        return {
            'best_cost': self.best_cost,
            'best_matrix': self.best_matrix,
            'total_iterations': self.iteration,
            'total_time': total_time,
            'history': self.history
        }
    
    def save_checkpoint(self):
        """Save optimization checkpoint."""
        checkpoint = {
            'iteration': self.iteration,
            'temperature': self.current_temp,
            'best_cost': self.best_cost,
            'current_cost': self.current_cost,
            'history': self.history
        }
        
        checkpoint_file = self.output_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        if self.best_matrix is not None:
            np.save(self.output_dir / "best_matrix_current.npy", self.best_matrix)
        if self.current_matrix is not None:
            np.save(self.output_dir / "current_matrix.npy", self.current_matrix)
    
    def save_results(self):
        """Save final optimization results."""
        results_file = self.output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'total_iterations': self.iteration,
                'best_cost': self.best_cost,
                'final_temperature': self.current_temp,
                'history': self.history,
                'parameters': {
                    'initial_temp': self.initial_temp,
                    'cooling_rate': self.cooling_rate,
                    'min_temp': self.min_temp,
                    'parallel_sims': self.parallel_sims,
                    'num_random': self.num_random,
                    'num_flips': self.num_flips
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to {self.output_dir}")
