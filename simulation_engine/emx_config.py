"""
Configuration file for EMX simulation settings.
"""

import os

# ==============================================================================
# EMX Binary and Process File Paths
# ==============================================================================

# Path to EMX executable
EMX_BIN_PATH = "emx"

# Path to the process file (update this to your PDK location)
PROC_FILE_PATH = "/disk/smiao23/GF22/PDK_240627_121855_281/22FDX-PLUS/V1.0_2.3/Emagnetic/EMX/10M_2Mx_5Cx_1Jx_2Qx_LBthick/22fdsoi_10M_2Mx_5Cx_1Jx_2Qx_LBthick_nominal_detailed.encrypted.proc"

# ==============================================================================
# Parallel Processing Settings
# ==============================================================================

# Number of parallel threads for EMX simulation
# - Set to 0 to auto-detect (uses 1.5x physical CPUs)
# - Set to a specific number to use that many threads
# - Generally beneficial to set slightly higher than CPU count for better load balancing
EMX_PARALLEL_THREADS = 0  # Auto-detect by default

# ==============================================================================
# Memory Settings
# ==============================================================================

# Maximum memory for EMX simulation
# Options:
#   - "XG" or "XM": Use X gigabytes/megabytes (e.g., "8G", "1024M")
#   - "-XG": Use all available memory except X gigabytes (e.g., "-2G")
#   - "X%": Use X percent of available memory (e.g., "75%")
#   - None: Use EMX default (recommended-memory flag)
EMX_MAX_MEMORY = "75%"  # Use 75% of available memory by default

# ==============================================================================
# Port Mapping
# ==============================================================================

# Port mapping for the device
# Maps GDS label to S-parameter port name
PORT_MAPPING = {
    "P000": "P1",       # Left signal
    "P001": "P1_G",     # Left ground
    "P002": "P2",       # Right signal
    "P003": "P2_G",     # Right ground
}

# ==============================================================================
# Simulation Parameters
# ==============================================================================

# Meshing parameters
MESH_EDGE_LENGTH = "1"      # -e parameter
MESH_THICKNESS = "1"        # -t parameter  
MESH_VIA_SIZE = "0.5"          # -v parameter

# Frequency sweep settings
SWEEP_START = "0"              # Start frequency (Hz)
SWEEP_END = "6e+10"            # End frequency (60 GHz)
SWEEP_STEPSIZE = "5e+08"       # Step size (500 MHz)

# Simulation flags
SUBSTRATE_LAYER = "0"          # -l parameter
VERBOSE_LEVEL = "0"            # --verbose parameter
EMX_KEY = "EMXkey"             # License key

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_memory_setting(max_parallel: int) -> str:
    """
    Calculate appropriate memory setting based on parallel simulations.
    
    Divides available memory among parallel simulations.
    E.g., if EMX_MAX_MEMORY="75%" and max_parallel=4, returns "18.75%"
    
    Args:
        max_parallel: Number of parallel simulations
        
    Returns:
        Memory setting string for EMX
    """
    if EMX_MAX_MEMORY is None:
        return None
        
    if isinstance(EMX_MAX_MEMORY, str):
        # Handle percentage
        if "%" in EMX_MAX_MEMORY:
            try:
                percent = float(EMX_MAX_MEMORY.rstrip("%"))
                per_sim_percent = percent / max_parallel
                return f"{per_sim_percent:.2f}%"
            except ValueError:
                return EMX_MAX_MEMORY
        
        # Handle negative offset (e.g., "-2G")
        # For negative offsets, we can't easily divide, so return as-is
        # and let EMX handle it (not ideal for parallel, but safe)
        elif EMX_MAX_MEMORY.startswith("-"):
            return EMX_MAX_MEMORY
            
        # Handle absolute amounts (e.g., "8G", "1024M")
        else:
            # Try to parse and divide
            try:
                if EMX_MAX_MEMORY.endswith("G"):
                    gb = float(EMX_MAX_MEMORY[:-1])
                    per_sim_gb = gb / max_parallel
                    return f"{per_sim_gb:.2f}G"
                elif EMX_MAX_MEMORY.endswith("M"):
                    mb = float(EMX_MAX_MEMORY[:-1])
                    per_sim_mb = mb / max_parallel
                    return f"{per_sim_mb:.2f}M"
            except ValueError:
                pass
    
    # Fallback: return as-is
    return EMX_MAX_MEMORY


def get_parallel_threads(max_parallel: int) -> int:
    """
    Get the number of parallel threads per EMX simulation.
    
    Divides threads among parallel simulations.
    E.g., on 16-core machine with max_parallel=4, returns ~6 threads/sim
    
    Args:
        max_parallel: Number of parallel simulations
        
    Returns:
        Number of threads per simulation (0 for auto-detect)
    """
    if EMX_PARALLEL_THREADS == 0:
        # Auto-detect: divide available cores among simulations
        # Leave it as 0 and let each simulation auto-detect,
        # OR calculate a reasonable split
        # For now, keep as 0 to let EMX auto-detect per simulation
        # (each will use 1.5x cores, which is fine when they run sequentially in thread pool)
        return 0
    else:
        # If user specified a number, divide it among parallel sims
        threads_per_sim = max(1, EMX_PARALLEL_THREADS // max_parallel)
        return threads_per_sim
