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
        cooling_rate: float = 0.98,
        min_temp: float = 0.01,
        parallel_sims: int = 4,
        num_random: int = 1,
        num_flips: int = 10,
        connectivity_check: Optional[Callable] = None,
        output_dir: str = "./optimization_results",
        center_freq_ghz: float = 35.0,
        bandwidth_percent: float = 0.25,
        passband_loss_target_db: float = 5.0,
        stopband_loss_target_db: float = 10.0,
        stopband_offset_ghz: float = 4.0,
        stopband_width_ghz: float = 10.0
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
            center_freq_ghz: Center frequency of filter in GHz
            bandwidth_percent: Fractional bandwidth (e.g., 0.25 = 25%)
            passband_loss_target_db: Target maximum insertion loss in passband
            stopband_loss_target_db: Target minimum rejection in stopband
            stopband_offset_ghz: Gap between passband edge and stopband region
            stopband_width_ghz: Width of each stopband region
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
        
        # Filter specifications
        self.center_freq_ghz = center_freq_ghz
        self.bandwidth_percent = bandwidth_percent
        self.passband_loss_target_db = passband_loss_target_db
        self.stopband_loss_target_db = stopband_loss_target_db
        self.stopband_offset_ghz = stopband_offset_ghz
        self.stopband_width_ghz = stopband_width_ghz
        
        # Calculate passband boundaries for reference
        bw_ghz = center_freq_ghz * bandwidth_percent
        self.passband_min = center_freq_ghz - bw_ghz / 2
        self.passband_max = center_freq_ghz + bw_ghz / 2
        
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
            'acceptance_prob': [],
            'design_name': []  # Track accepted design names
        }
        
        print(f"🔥 Simulated Annealing Optimizer initialized")
        print(f"   Parallel simulations: {parallel_sims} ({num_random} random + {self.num_neighbors} neighbors)")
        print(f"   Temperature: {initial_temp} → {min_temp} (cooling: {cooling_rate})")
        print(f"   Neighbor strategy: flip {num_flips} pixels")
        print(f"   Filter specs: {center_freq_ghz} GHz center, {bandwidth_percent*100:.0f}% BW")
        print(f"   Passband: {self.passband_min:.1f}-{self.passband_max:.1f} GHz")
        print(f"   Stopband offset: ±{stopband_offset_ghz} GHz, width: {stopband_width_ghz} GHz")
    
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
    
    def validate_design(self, matrix: np.ndarray, max_attempts: int = 100) -> np.ndarray:
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
        
        cost = calculate_filter_cost(
            str(s2p_path),
            center_freq_ghz=self.center_freq_ghz,
            bandwidth_percent=self.bandwidth_percent,
            passband_loss_target_db=self.passband_loss_target_db,
            stopband_loss_target_db=self.stopband_loss_target_db,
            stopband_offset_ghz=self.stopband_offset_ghz,
            stopband_width_ghz=self.stopband_width_ghz
        )
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
                # Randomize number of flips between 1-10 for better local refinement
                # Too many flips destroy good EM structures
                num_flips = np.random.randint(1, 11)
                matrix = self.generate_neighbor(self.current_matrix, num_flips)
                # Do not check connectivity for neighbors to increase diversity
                # if self.connectivity_check:
                #     matrix = self.validate_design(matrix)
                
                job = SimulationJob(
                    job_id=f"iter{self.iteration}_neighbor{i+1}",
                    design_matrix=matrix,
                    metadata={
                        'iteration': self.iteration,
                        'type': 'neighbor',
                        'temperature': self.current_temp,
                        'num_flips': num_flips
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
                
                # Plot S-parameters for accepted design
                if best_candidate['s2p_path']:
                    try:
                        self._plot_sparameters(best_candidate['s2p_path'], 
                                              best_candidate['job_id'],
                                              self.iteration)
                    except Exception as e:
                        print(f"  ⚠️  Failed to plot S-parameters: {e}")
            else:
                print(f"❌ Rejected new solution (p={accept_prob:.4f})")
        
        # Update history
        self.history['iteration'].append(self.iteration)
        self.history['temperature'].append(self.current_temp)
        self.history['current_cost'].append(self.current_cost)
        self.history['best_cost'].append(self.best_cost)
        self.history['accepted'].append(accepted)
        self.history['acceptance_prob'].append(accept_prob)
        self.history['design_name'].append(best_candidate['job_id'])
        
        # Cool down temperature
        self.current_temp *= self.cooling_rate
        
        # Enforce minimum temperature
        if self.current_temp < self.min_temp:
            self.current_temp = self.min_temp
        
        # Don't stop early based on temperature - let it run full max_iterations
        # This allows the algorithm to explore more even at low temperatures
        
        return True
    
    def _plot_sparameters(self, s2p_path: str, design_name: str, iteration: int):
        """Plot S-parameters for an accepted design."""
        import skrf as rf
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Load network
        net = rf.Network(s2p_path)
        freqs_ghz = net.frequency.f / 1e9
        
        # Get S-parameters in dB
        s11_db = 20 * np.log10(np.abs(net.s[:, 0, 0]))
        s21_db = 20 * np.log10(np.abs(net.s[:, 1, 0]))
        s12_db = 20 * np.log10(np.abs(net.s[:, 0, 1]))
        s22_db = 20 * np.log10(np.abs(net.s[:, 1, 1]))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Return loss (S11, S22)
        ax1.plot(freqs_ghz, s11_db, 'b-', label='S11', linewidth=2)
        ax1.plot(freqs_ghz, s22_db, 'r-', label='S22', linewidth=2)
        ax1.axhline(y=-10, color='k', linestyle='--', alpha=0.3, label='−10dB')
        ax1.set_ylabel('Magnitude (dB)', fontsize=12)
        ax1.set_title(f'S-Parameters - Iteration {iteration} - {design_name}\nCost: {self.current_cost:.4f}', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([10, 60])
        
        # Mark passband region on return loss
        ax1.axvspan(self.passband_min, self.passband_max, alpha=0.1, color='green', label='Passband')
        
        # Mark stopband regions
        lower_sb_max = self.passband_min - self.stopband_offset_ghz
        lower_sb_min = lower_sb_max - self.stopband_width_ghz
        upper_sb_min = self.passband_max + self.stopband_offset_ghz
        upper_sb_max = upper_sb_min + self.stopband_width_ghz
        ax1.axvspan(lower_sb_min, lower_sb_max, alpha=0.05, color='red', label='Stopband')
        ax1.axvspan(upper_sb_min, upper_sb_max, alpha=0.05, color='red')
        
        # Insertion loss (S21, S12)
        ax2.plot(freqs_ghz, s21_db, 'g-', label='S21', linewidth=2)
        ax2.plot(freqs_ghz, s12_db, 'm-', label='S12', linewidth=2, alpha=0.7)
        ax2.axhline(y=-3, color='orange', linestyle='--', alpha=0.5, label='−3dB')
        ax2.axhline(y=-5, color='red', linestyle='--', alpha=0.5, label='−5dB (target)')
        ax2.set_xlabel('Frequency (GHz)', fontsize=12)
        ax2.set_ylabel('Magnitude (dB)', fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([10, 60])
        
        # Mark passband and stopband regions on insertion loss
        ax2.axvspan(self.passband_min, self.passband_max, alpha=0.1, color='green')
        ax2.axvspan(lower_sb_min, lower_sb_max, alpha=0.05, color='red')
        ax2.axvspan(upper_sb_min, upper_sb_max, alpha=0.05, color='red')
        
        plt.tight_layout()
        
        # Save figure
        plot_file = self.output_dir / f"sparams_iter{iteration}_{design_name}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  📊 S-parameter plot saved: {plot_file.name}")
    
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
        # Convert numpy types to native Python types for JSON serialization
        checkpoint = {
            'iteration': int(self.iteration),
            'temperature': float(self.current_temp),
            'best_cost': float(self.best_cost) if self.best_cost != float('inf') else None,
            'current_cost': float(self.current_cost) if self.current_cost != float('inf') else None,
            'history': {
                'iteration': [int(x) for x in self.history['iteration']],
                'temperature': [float(x) for x in self.history['temperature']],
                'current_cost': [float(x) if x != float('inf') else None for x in self.history['current_cost']],
                'best_cost': [float(x) if x != float('inf') else None for x in self.history['best_cost']],
                'accepted': [bool(x) for x in self.history['accepted']],
                'acceptance_prob': [float(x) for x in self.history['acceptance_prob']]
            }
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
        
        # Convert to native Python types for JSON serialization
        results = {
            'total_iterations': int(self.iteration),
            'best_cost': float(self.best_cost) if self.best_cost != float('inf') else None,
            'final_temperature': float(self.current_temp),
            'history': {
                'iteration': [int(x) for x in self.history['iteration']],
                'temperature': [float(x) for x in self.history['temperature']],
                'current_cost': [float(x) if x != float('inf') else None for x in self.history['current_cost']],
                'best_cost': [float(x) if x != float('inf') else None for x in self.history['best_cost']],
                'accepted': [bool(x) for x in self.history['accepted']],
                'acceptance_prob': [float(x) for x in self.history['acceptance_prob']],
                'design_name': [str(x) for x in self.history['design_name']]
            },
            'parameters': {
                'initial_temp': float(self.initial_temp),
                'cooling_rate': float(self.cooling_rate),
                'min_temp': float(self.min_temp),
                'parallel_sims': int(self.parallel_sims),
                'num_random': int(self.num_random),
                'num_flips': int(self.num_flips)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {self.output_dir}")
