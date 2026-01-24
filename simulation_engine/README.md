# Simulation Engine

A producer-consumer simulation engine for parallel EMX simulations.

## Architecture

```
Producer (You)          Consumer (SimulationEngine)
    │                            │
    │  Submit Job                │
    │  (design matrix)           │
    ├──────────────────────────> │
    │                            │
    │                        ┌───┴───┐
    │                        │ Queue │
    │                        └───┬───┘
    │                            │
    │                     ┌──────┴──────┐
    │                     │  Worker Pool │
    │                     │ (parallel)   │
    │                     └──────┬──────┘
    │                            │
    │  Return Result             │
    │  (S-parameters)            │
    │ <──────────────────────────┤
    │                            │
```

## Key Features

- **One-time initialization**: PDK, LayerMap, and EMXSimInterface initialized once
- **Parallel processing**: Configurable worker pool (default: 4 parallel jobs)
- **Structured output**: All results organized in a clean directory structure
- **Producer-consumer pattern**: Submit jobs asynchronously, get results when ready
- **Thread-safe**: Multiple jobs can be submitted from different threads

## Directory Structure

```
simulation_results/
├── gds_files/          # Generated GDS files
│   ├── device_job001_timestamp.gds
│   └── ...
├── snp_files/          # S-parameter files
│   ├── device_job001_timestamp.s4p
│   ├── device_job001_timestamp/  # Processed SNP analysis
│   └── ...
├── matrices/           # Design matrices (numpy arrays)
│   ├── device_job001_timestamp.npy
│   └── ...
├── plots/              # Generated plots
│   ├── device_job001_timestamp_binary.png
│   ├── device_job001_timestamp.png
│   └── ...
└── logs/               # Error logs (if any)
    └── job001_error.log
```

## Usage

### Basic Example

```python
from simulation_engine import SimulationEngine, SimulationJob
import numpy as np

# Initialize engine ONCE (slow initialization happens here)
with SimulationEngine(
    base_output_dir="./my_results",
    max_parallel=4
) as engine:
    
    # Create a job with your design matrix
    job = SimulationJob(
        job_id="my_design_001",
        design_matrix=my_matrix,
        metadata={"description": "Test design"}
    )
    
    # Submit and get result
    future = engine.submit_job(job)
    result = future.result()  # Blocks until complete
    
    if result.success:
        print(f"SNP file: {result.snp_path}")
```

### Batch Processing (Parallel)

```python
# Initialize engine once
with SimulationEngine(max_parallel=4) as engine:
    
    # Create multiple jobs
    jobs = [
        SimulationJob(f"job_{i}", matrix_i)
        for i, matrix_i in enumerate(my_matrices)
    ]
    
    # Process all in parallel (up to 4 at a time)
    results = engine.submit_batch(jobs)
    
    # All results are ready here
    for result in results:
        print(result.job_id, result.success)
```

### Continuous Production

```python
# Keep engine running in background
engine = SimulationEngine(max_parallel=3)

try:
    # Producer loop
    for iteration in range(100):
        matrix = generate_new_matrix()  # Your matrix generator
        
        job = SimulationJob(f"iter_{iteration}", matrix)
        future = engine.submit_job(job)
        
        # Can submit next job immediately
        # Results will be processed in background
        
finally:
    engine.shutdown()  # Clean shutdown
```

## API Reference

### SimulationEngine

**Constructor Parameters:**
- `base_output_dir`: Root directory for outputs (default: "./simulation_results")
- `max_parallel`: Max simultaneous simulations (default: 4)
- `rows`, `cols`: Grid dimensions (default: 21, 30)
- `pitch_x`, `pitch_y`: Pixel spacing (default: 30.0)
- `pixel_w`, `pixel_h`: Pixel dimensions (default: 34.0)
- `tl_w`, `tl_l`: Transmission line parameters (default: 34.0, 60.0)
- `port_stick_len`: Port stick length (default: 5.0)

**Methods:**
- `submit_job(job)`: Submit single job, returns Future
- `submit_batch(jobs, callback)`: Submit multiple jobs, returns list of results
- `shutdown(wait=True)`: Shutdown engine

### SimulationJob

**Attributes:**
- `job_id`: Unique identifier for the job
- `design_matrix`: numpy array with the pixel pattern
- `metadata`: Optional dict for custom data

### SimulationResult

**Attributes:**
- `job_id`: Job identifier
- `success`: Boolean indicating success/failure
- `gds_path`: Path to generated GDS file
- `snp_path`: Path to S-parameter file
- `matrix_path`: Path to saved numpy matrix
- `plot_paths`: Dict of plot file paths
- `error_message`: Error details (if failed)
- `timestamp`: Completion timestamp
- `metadata`: Metadata from job plus additional info

## Examples

See `demo_simulation_engine.py` for complete examples:

```bash
# Run single job example
python demo_simulation_engine.py 1

# Run batch processing example
python demo_simulation_engine.py 2

# Run continuous production example
python demo_simulation_engine.py 3

# Run custom matrices example
python demo_simulation_engine.py 4
```

## Performance Tips

1. **Parallel jobs**: Set `max_parallel` based on your CPU cores and memory
2. **Batch submission**: Use `submit_batch()` for better efficiency
3. **Callbacks**: Use callbacks to process results as they complete
4. **Context manager**: Use `with` statement for automatic cleanup

## Integration with RL/Optimization

```python
# Example: Using with reinforcement learning
engine = SimulationEngine(max_parallel=8)

try:
    for episode in range(num_episodes):
        # RL agent generates design
        design_matrix = rl_agent.generate_design()
        
        # Submit for simulation
        job = SimulationJob(f"rl_ep{episode}", design_matrix)
        future = engine.submit_job(job)
        
        # Get result
        result = future.result()
        
        # Process S-parameters for reward
        if result.success:
            reward = compute_reward(result.snp_path)
            rl_agent.update(reward)
finally:
    engine.shutdown()
```
