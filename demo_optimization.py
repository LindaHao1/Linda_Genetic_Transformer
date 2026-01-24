"""
Demo script for parallel simulated annealing optimization of pixelated filters.
"""

import numpy as np
from collections import deque

from simulation_engine import SimulationEngine
from optimization import SimulatedAnnealingOptimizer


def check_connectivity(matrix, rows, cols, pitch_y, pixel_h, tline_width, pitch_x_req):
    """Check if the matrix has a valid path from left to right."""
    # Calculate geometric Y-centers for rows to check tline overlap
    y_coords = np.arange(rows) * pitch_y
    center_y = (y_coords[-1] + y_coords[0]) / 2.0
    
    tline_min = center_y - tline_width / 2.0
    tline_max = center_y + tline_width / 2.0
    
    start_nodes = []
    end_nodes = []
    
    # Identify ports: pixels intersecting tline by at least pitch_x
    for r in range(rows):
        p_min = y_coords[r] - pixel_h / 2.0
        p_max = y_coords[r] + pixel_h / 2.0
        
        overlap_len = max(0, min(tline_max, p_max) - max(tline_min, p_min))
        
        if overlap_len >= pitch_x_req - 1e-9:
            if matrix[r, 0] == 1:
                start_nodes.append((r, 0))
            if matrix[r, cols - 1] == 1:
                end_nodes.append((r, cols - 1))
                
    if not start_nodes or not end_nodes:
        return False
        
    # BFS Pathfinding (8-way connectivity)
    queue = deque(start_nodes)
    visited = set(start_nodes)
    targets = set(end_nodes)
    
    while queue:
        r, c = queue.popleft()
        if (r, c) in targets:
            return True
        
        # 8 neighbors (diagonal included)
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
                     
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if matrix[nr, nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def main():
    """Run the optimization."""
    
    # Design parameters
    rows = 21
    cols = 30
    pitch_x = 30.0
    pitch_y = 30.0
    pixel_h = 34.0
    tl_w = 34.0
    
    # Create connectivity checker
    def connectivity_validator(matrix):
        return check_connectivity(matrix, rows, cols, pitch_y, pixel_h, tl_w, pitch_x)
    
    # Initialize simulation engine
    print("🚀 Initializing Simulation Engine...")
    engine = SimulationEngine(
        base_output_dir="./optimization_simulations",
        max_parallel=4,  # Run 4 simulations in parallel
        rows=rows,
        cols=cols,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        pixel_w=34.0,
        pixel_h=34.0,
        tl_w=tl_w,
        tl_l=60.0,
        port_stick_len=5.0
    )
    
    try:
        # Initialize optimizer
        optimizer = SimulatedAnnealingOptimizer(
            simulation_engine=engine,
            rows=rows,
            cols=cols,
            initial_temp=2.0,        # Starting temperature
            cooling_rate=0.92,       # Cool down 8% each iteration
            min_temp=0.05,           # Stop when temp < 0.05
            parallel_sims=4,         # 4 parallel simulations per iteration
            num_random=1,            # 1 random + 3 neighbors
            num_flips=10,            # Flip 10 pixels for neighbors
            connectivity_check=connectivity_validator,
            output_dir="./optimization_results"
        )
        
        # Run optimization
        # Each iteration will:
        # 1. Generate 1 random design + 3 neighbor designs (flip 10 pixels)
        # 2. Simulate all 4 in parallel
        # 3. Evaluate costs based on S21 filter performance (35 GHz, 25% BW)
        # 4. Accept/reject based on temperature and cost improvement
        # 5. Cool down temperature
        
        results = optimizer.optimize(max_iterations=20)  # Run for 20 iterations
        
        print("\n" + "="*80)
        print("📊 OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Best Cost: {results['best_cost']:.4f}")
        print(f"Total Iterations: {results['total_iterations']}")
        print(f"Total Time: {results['total_time']:.1f}s")
        print(f"\nBest design saved to: ./optimization_results/best_matrix_current.npy")
        
        # Plot convergence
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Cost over time
            ax1.plot(results['history']['iteration'], 
                    results['history']['current_cost'], 
                    'b-', alpha=0.5, label='Current Cost')
            ax1.plot(results['history']['iteration'], 
                    results['history']['best_cost'], 
                    'r-', linewidth=2, label='Best Cost')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Cost')
            ax1.set_title('Optimization Progress')
            ax1.legend()
            ax1.grid(True)
            ax1.set_yscale('log')
            
            # Temperature over time
            ax2.plot(results['history']['iteration'], 
                    results['history']['temperature'], 
                    'g-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Temperature')
            ax2.set_title('Annealing Schedule')
            ax2.grid(True)
            ax2.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig('./optimization_results/convergence.png', dpi=150)
            print(f"Convergence plot saved to: ./optimization_results/convergence.png")
            
        except Exception as e:
            print(f"Could not generate convergence plot: {e}")
        
    finally:
        # Clean shutdown
        engine.shutdown()


if __name__ == "__main__":
    main()
