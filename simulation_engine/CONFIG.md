# EMX Configuration Guide

The simulation engine uses a centralized configuration file for EMX settings.

## Configuration File

Edit [emx_config.py](emx_config.py) to customize your simulation settings.

## Key Settings

### 1. Paths

```python
# EMX executable path
EMX_BIN_PATH = "emx"

# Process file from your PDK
PROC_FILE_PATH = "/path/to/your/pdk/proc_file.proc"
```

### 2. Parallel Processing

```python
# Number of parallel threads for EMX
EMX_PARALLEL_THREADS = 0  # 0 = auto-detect (1.5x CPU count)
                          # N = use N threads
```

**Recommendation:** Set to `0` for automatic detection or slightly higher than your CPU count (e.g., if you have 8 cores, try 10-12 threads).

### 3. Memory Settings

```python
EMX_MAX_MEMORY = "75%"  # Options:
                        # "XG" or "XM" - Use X gigabytes/megabytes (e.g., "8G")
                        # "-XG" - All memory except X GB (e.g., "-2G")
                        # "X%" - X percent of available memory (e.g., "75%")
                        # None - Use EMX default
```

**Memory Strategy:**
- **For parallel simulations:** Use percentage (e.g., `"75%"`) to leave headroom for system
- **For single large simulation:** Use specific amount (e.g., `"12G"`)
- **Conservative:** Use negative offset (e.g., `"-2G"`) to reserve memory

### 4. Port Mapping

```python
PORT_MAPPING = {
    "P000": "P1",       # GDS label -> S-param port name
    "P001": "P1_G",
    "P002": "P2",
    "P003": "P2_G",
}
```

Match these to your GDS port labels.

### 5. Simulation Parameters

```python
# Meshing
MESH_EDGE_LENGTH = "0.25"
MESH_THICKNESS = "0.25"
MESH_VIA_SIZE = "0.5"

# Frequency sweep (0 to 60 GHz, 500 MHz steps)
SWEEP_START = "0"
SWEEP_END = "6e+10"
SWEEP_STEPSIZE = "5e+08"
```

## Usage in Code

### Using Default Config

```python
from simulation_engine import SimulationEngine

# Uses settings from emx_config.py
engine = SimulationEngine(max_parallel=4)
```

### Override Config

```python
from simulation_engine import SimulationEngine, emx_config

# Temporarily modify config
emx_config.EMX_PARALLEL_THREADS = 8
emx_config.EMX_MAX_MEMORY = "10G"

engine = SimulationEngine(max_parallel=4)
```

### Custom EMXSimInterface

```python
from simulation_engine.emx_runner import EMXSimInterface

# Override specific settings
runner = EMXSimInterface(
    destination_folder="./results",
    parallel_threads=12,
    max_memory="16G"
)
```

## Performance Tuning

### CPU Usage

- **More threads = faster simulation** (up to a point)
- Diminishing returns after ~1.5x CPU count
- Load balancing isn't perfect, so slightly over-provisioning helps

### Memory Usage

| Scenario | Recommendation |
|----------|----------------|
| Many small devices | `"75%"` or `"-2G"` |
| Few large devices | `"90%"` or specific amount |
| Mixed workload | `"80%"` |
| Shared machine | `"50%"` or `-4G` |

### Finding Optimal Settings

Run a test batch and monitor:

```bash
# Monitor CPU usage
htop

# Monitor memory
free -h
```

Adjust `EMX_PARALLEL_THREADS` and `EMX_MAX_MEMORY` based on observations.

## Example Configurations

### High-Performance Workstation (16 cores, 64GB RAM)

```python
EMX_PARALLEL_THREADS = 0    # Auto: ~24 threads
EMX_MAX_MEMORY = "-4G"      # Use all but 4GB
```

### Shared Server (8 cores, 32GB RAM)

```python
EMX_PARALLEL_THREADS = 6    # Conservative
EMX_MAX_MEMORY = "50%"      # Leave half for others
```

### Laptop (4 cores, 16GB RAM)

```python
EMX_PARALLEL_THREADS = 0    # Auto: ~6 threads
EMX_MAX_MEMORY = "60%"      # Leave room for OS
```

## Troubleshooting

### Out of Memory Errors

- Reduce `EMX_MAX_MEMORY`
- Reduce `max_parallel` in SimulationEngine
- Process smaller batches

### Slow Simulations

- Increase `EMX_PARALLEL_THREADS`
- Increase `EMX_MAX_MEMORY`
- Check if EMX is actually using multiple threads (check htop)

### EMX Crashes

- Check `PROC_FILE_PATH` is correct
- Verify EMX license is valid
- Check log files in `simulation_results/snp_files/*_emx.log`
