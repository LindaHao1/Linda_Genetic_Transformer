"""
Demo script showing how to use the SimulationEngine.

This demonstrates the producer-consumer pattern where:
- You (the producer) submit pixel matrices as jobs
- SimulationEngine (the consumer) processes them in parallel
"""

import numpy as np
from simulation_engine.pixel_gds import generate_design_matrix
from simulation_engine import SimulationEngine, SimulationJob
from collections import deque


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
        
        if overlap_len >= tline_width - 1e-9:
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


def generate_valid_matrix(rows, cols, pitch_y, pixel_h, tl_w, pitch_x, max_attempts=100):
    """Generate a design matrix that satisfies connectivity requirements."""
    for attempt in range(max_attempts):
        matrix = generate_design_matrix(rows=rows, cols=cols)
        if check_connectivity(matrix, rows, cols, pitch_y, pixel_h, tl_w, pitch_x):
            return matrix
    raise RuntimeError(f"Failed to generate valid matrix after {max_attempts} attempts")


def result_callback(result):
    """Optional callback function called when each job completes."""
    if result.success:
        print(f"   💾 Results saved:")
        print(f"      - GDS: {result.gds_path}")
        print(f"      - SNP: {result.snp_path}")
        print(f"      - Matrix: {result.matrix_path}")
    else:
        print(f"   ⚠️  Error: {result.error_message}")


# ==============================================================================
# USAGE EXAMPLE 1: Basic Single Job
# ==============================================================================
def example_single_job():
    """Example: Submit and process a single job."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Job Processing")
    print("="*80)
    
    # Design parameters
    rows = 21
    cols = 30
    pitch_x = 30.0
    pitch_y = 30.0
    pixel_h = 34.0
    tl_w = 34.0
    
    # Initialize engine (this is the slow one-time initialization)
    with SimulationEngine(
        base_output_dir="./engine_results_single",
        max_parallel=2,
        rows=rows,
        cols=cols
    ) as engine:
        
        # Generate a valid design matrix
        print("Generating design matrix...")
        matrix = generate_valid_matrix(rows, cols, pitch_y, pixel_h, tl_w, pitch_x)
        
        # Create a job
        job = SimulationJob(
            job_id="single_001",
            design_matrix=matrix,
            metadata={"description": "Single job example"}
        )
        
        # Submit and wait for result
        future = engine.submit_job(job)
        result = future.result()  # This blocks until complete
        
        # Check result
        if result.success:
            print(f"\n✅ Simulation completed successfully!")
            print(f"   GDS: {result.gds_path}")
            print(f"   SNP: {result.snp_path}")
        else:
            print(f"\n❌ Simulation failed: {result.error_message}")


# ==============================================================================
# USAGE EXAMPLE 2: Batch Processing (Multiple Jobs in Parallel)
# ==============================================================================
def example_batch_processing():
    """Example: Submit multiple jobs and process in parallel."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Processing (Parallel)")
    print("="*80)
    
    # Design parameters
    rows = 21
    cols = 30
    pitch_x = 15.0
    pitch_y = 15.0
    pixel_h = 18.0
    tl_w = 15.8
    
    # Initialize engine ONCE
    with SimulationEngine(
        base_output_dir="./engine_results_batch",
        max_parallel=4,  # Run up to 4 simulations in parallel
        rows=rows,
        cols=cols,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        pixel_w=pixel_h,
        pixel_h=pixel_h,
        tl_w=tl_w
    ) as engine:
        
        # Generate multiple design matrices
        num_designs = 4
        print(f"Generating {num_designs} valid design matrices...")
        
        jobs = []
        for i in range(num_designs):
            print(f"  Generating matrix {i+1}/{num_designs}...")
            matrix = generate_valid_matrix(rows, cols, pitch_y, pixel_h, tl_w, pitch_x)
            
            job = SimulationJob(
                job_id=f"batch_{i+1:03d}",
                design_matrix=matrix,
                metadata={
                    "batch": "batch_001",
                    "index": i,
                    "description": f"Design variant {i+1}"
                }
            )
            jobs.append(job)
        
        # Submit all jobs at once (they will be processed in parallel)
        results = engine.submit_batch(jobs, callback=result_callback)
        
        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\n📊 Summary:")
        print(f"   Total jobs: {len(results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(failed)}")
        
        if successful:
            print(f"\n✅ Successful simulations:")
            for r in successful:
                print(f"   - {r.job_id}: {r.snp_path}")


# ==============================================================================
# USAGE EXAMPLE 3: Continuous Production (Producer-Consumer)
# ==============================================================================
def example_continuous_production():
    """Example: Continuously produce matrices and submit them."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Continuous Production Pattern")
    print("="*80)
    
    # Design parameters
    rows = 21
    cols = 30
    pitch_x = 30.0
    pitch_y = 30.0
    pixel_h = 34.0
    tl_w = 34.0
    
    # Initialize engine ONCE and keep it running
    engine = SimulationEngine(
        base_output_dir="./engine_results_continuous",
        max_parallel=3,
        rows=rows,
        cols=cols
    )
    
    try:
        futures = []
        num_batches = 2
        designs_per_batch = 3
        
        for batch_idx in range(num_batches):
            print(f"\n📦 Producing batch {batch_idx + 1}/{num_batches}")
            
            for design_idx in range(designs_per_batch):
                # Generate matrix (producer activity)
                matrix = generate_valid_matrix(rows, cols, pitch_y, pixel_h, tl_w, pitch_x)
                
                # Create and submit job
                job = SimulationJob(
                    job_id=f"continuous_b{batch_idx+1}_d{design_idx+1}",
                    design_matrix=matrix,
                    metadata={
                        "batch": batch_idx + 1,
                        "design": design_idx + 1
                    }
                )
                
                future = engine.submit_job(job)
                futures.append((job.job_id, future))
                print(f"  ✉️  Submitted job: {job.job_id}")
        
        # Wait for all jobs to complete
        print(f"\n⏳ Waiting for {len(futures)} jobs to complete...")
        results = []
        for job_id, future in futures:
            result = future.result()
            results.append(result)
            status = "✅" if result.success else "❌"
            print(f"{status} {job_id} completed")
        
        # Summary
        successful = sum(1 for r in results if r.success)
        print(f"\n📊 Final Summary: {successful}/{len(results)} succeeded")
        
    finally:
        # Clean shutdown
        engine.shutdown()


# ==============================================================================
# USAGE EXAMPLE 4: Using Your Own Matrices
# ==============================================================================
def example_custom_matrices():
    """Example: Use pre-defined or custom matrices."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Matrices")
    print("="*80)
    
    rows = 21
    cols = 30
    
    # Create custom matrices (e.g., from RL agent, optimization, etc.)
    custom_matrices = [
        np.random.randint(0, 2, size=(rows, cols)),
        np.ones((rows, cols)),  # All pixels on
        np.eye(rows, cols, dtype=int),  # Diagonal pattern
    ]
    
    with SimulationEngine(
        base_output_dir="./engine_results_custom",
        max_parallel=2,
        rows=rows,
        cols=cols
    ) as engine:
        
        jobs = []
        for i, matrix in enumerate(custom_matrices):
            job = SimulationJob(
                job_id=f"custom_{i+1}",
                design_matrix=matrix,
                metadata={"type": "custom", "index": i}
            )
            jobs.append(job)
        
        results = engine.submit_batch(jobs)
        
        print(f"\n✅ Processed {len(results)} custom designs")


# ==============================================================================
# USAGE EXAMPLE 5: Batch Processing from Folder
# ==============================================================================
def example_from_folder():
    """Example: Load matrices from a folder and simulate them."""
    import os
    from pathlib import Path
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Processing from Folder")
    print("="*80)
    
    # Folder containing .npy matrix files
    matrix_folder = "/disk/mcao53/RFIC_RL/engine_results_batch/matrices"  # Change this to your folder
    
    if not os.path.exists(matrix_folder):
        print(f"❌ Folder not found: {matrix_folder}")
        print("   Create the folder and add .npy matrix files first.")
        return
    
    # Find all .npy files in the folder
    matrix_files = list(Path(matrix_folder).glob("*.npy"))
    
    if not matrix_files:
        print(f"❌ No .npy files found in {matrix_folder}")
        print("   Add some matrix files first (e.g., from previous simulations).")
        return
    
    print(f"Found {len(matrix_files)} matrix files in {matrix_folder}")
    
    rows = 21
    cols = 30
    pitch_x = 15.0
    pitch_y = 15.0
    pixel_h = 18.0
    tl_w = 15.8
    
    # Initialize engine ONCE
    with SimulationEngine(
        base_output_dir="./engine_results_batch_from_folder",
        max_parallel=4,  # Run up to 4 simulations in parallel
        rows=rows,
        cols=cols,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        pixel_w=pixel_h,
        pixel_h=pixel_h,
        tl_w=tl_w
    ) as engine:
        
        jobs = []
        for i, matrix_file in enumerate(matrix_files):
            # Load matrix
            matrix = np.load(str(matrix_file))
            
            # Extract filename without extension as job_id
            job_id = f"mat_{i+1}"
            
            print(f"  Loaded: {matrix_file.name} (shape: {matrix.shape})")
            
            job = SimulationJob(
                job_id=job_id,
                design_matrix=matrix,
                metadata={
                    "source": "folder",
                    "original_file": str(matrix_file)
                }
            )
            jobs.append(job)
        
        # Simulate all matrices from the folder
        print(f"\n🚀 Starting simulations for {len(jobs)} designs...")
        results = engine.submit_batch(jobs, callback=result_callback)
        
        # Summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\n📊 Summary:")
        print(f"   Total jobs: {len(results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(failed)}")
        
        if successful:
            print(f"\n✅ Successful simulations:")
            for r in successful:
                print(f"   - {r.job_id}")
                print(f"     S2P: {r.metadata.get('s2p_path', 'N/A')}")


# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("SIMULATION ENGINE DEMO")
    print("="*80)
    
    examples = {
        "1": ("Single job", example_single_job),
        "2": ("Batch processing", example_batch_processing),
        "3": ("Continuous production", example_continuous_production),
        "4": ("Custom matrices", example_custom_matrices),
        "5": ("Process from folder", example_from_folder),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        print("\nUsage: python demo_simulation_engine.py [1|2|3|4|5]")
        print("Running example 2 (batch processing) by default...\n")
        choice = "2"
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}\n")
        func()
    else:
        print(f"Invalid choice: {choice}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
