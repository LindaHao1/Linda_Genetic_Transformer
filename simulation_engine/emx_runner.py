import os
import subprocess
from typing import List, Optional
from . import emx_config

class EMXSimInterface:
    def __init__(
        self, 
        destination_folder: str,
        emx_bin_path: Optional[str] = None,
        proc_file_path: Optional[str] = None,
        parallel_threads: Optional[int] = None,
        max_memory: Optional[str] = None,
    ):
        """
        Initialize the EMX Interface with your specific environment paths.
        
        Args:
            destination_folder: Where to save simulation results
            emx_bin_path: Path to EMX binary (None = use config)
            proc_file_path: Path to process file (None = use config)
            parallel_threads: Number of parallel threads (None = use config, 0 = auto-detect)
            max_memory: Memory limit like "8G", "-2G", "75%" (None = use config)
        """
        self.emx_bin = emx_bin_path or emx_config.EMX_BIN_PATH
        self.proc_file = proc_file_path or emx_config.PROC_FILE_PATH
        self.dest_folder = os.path.abspath(destination_folder)
        
        # Parallel and memory settings
        self.parallel_threads = parallel_threads if parallel_threads is not None else emx_config.get_parallel_threads()
        self.max_memory = max_memory if max_memory is not None else emx_config.EMX_MAX_MEMORY
        
        # Ensure destination folder exists
        os.makedirs(self.dest_folder, exist_ok=True)

    def run_single_emx(self, gds_file: str, cell_name: str) -> Optional[str]:
        """
        Constructs and runs the exact EMX command for a single design.
        """
        # 1. Setup Paths
        abs_gds_path = os.path.abspath(gds_file)
        
        # Output pattern: EMX uses %d to indicate port count (e.g., .s4p)
        # We target the destination folder.
        base_output_path = os.path.join(self.dest_folder, cell_name)
        output_arg = f"{base_output_path}.s%dp"
        
        # Log file path (to keep your destination clean or organized)
        log_file = os.path.join(self.dest_folder, f"{cell_name}_emx.log")

        # 2. Define Port Mapping
        # Load port mapping from config
        port_flags = []
        for gds_label, sport_name in emx_config.PORT_MAPPING.items():
            port_flags.extend(["-p", f"{gds_label}={sport_name}"])

        # 3. Construct the Command List
        cmd = [
            self.emx_bin,
            abs_gds_path,                   # Input GDS
            cell_name,                      # Top Cell Name
            self.proc_file,                 # Process File
            
            # --- Meshing & Geometry ---
            "-e", emx_config.MESH_EDGE_LENGTH, 
            "-t", emx_config.MESH_THICKNESS, 
            "-v", emx_config.MESH_VIA_SIZE, 
            "--3d=-*",                      # Exclude 3D export for specific layers if needed
            
            # --- Ports ---
            *port_flags,
            
            # --- Sweep Settings ---
            "--sweep", emx_config.SWEEP_START, emx_config.SWEEP_END, 
            "--sweep-stepsize", emx_config.SWEEP_STEPSIZE,
            
            # --- Logging & Verbosity ---
            f"--verbose={emx_config.VERBOSE_LEVEL}",
            "--print-command-line",
            "--log-time-level=0",
            "--extended-warnings",
            f"--message-file={log_file}",
            
            # --- Simulation Flags ---
            "-l", emx_config.SUBSTRATE_LAYER,
            "--quasistatic",
            "--dump-connectivity",
            f"--parallel={self.parallel_threads}",
            "--simultaneous-frequencies=0",
            f"--key={emx_config.EMX_KEY}",
        ]
        
        # Add memory setting if specified
        if self.max_memory:
            cmd.append(f"--max-memory={self.max_memory}")
        else:
            cmd.append("--recommended-memory")
        
        # Add output format
        cmd.extend([
            "--format=touchstone2",
            "-s", output_arg                # Result saved to destination folder
        ])

        # 4. Execute
        try:
            # Check if proc file exists to avoid immediate crash
            if not os.path.exists(self.proc_file):
                print(f"Error: Process file not found at {self.proc_file}")
                return None

            subprocess.run(cmd, check=True)
            
            # Return expected filename. 
            # EMX replaces %d with the number of ports (4 in this case)
            expected_output = f"{base_output_path}.s4p"
            if os.path.exists(expected_output):
                return expected_output
            else:
                print(f"Warning: EMX finished but output file missing: {expected_output}")
                return None

        except subprocess.CalledProcessError as e:
            print(f"EMX Error for {cell_name}. Exit code: {e.returncode}")
            return None
        except Exception as e:
            print(f"Unexpected Python Error: {e}")
            return None

# ==============================================================================
# Usage Example
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Define where you want to save the results
    output_folder = "./simulation_results_batch_1"

    # 2. Instantiate the interface
    runner = EMXSimInterface(destination_folder=output_folder)

    # 3. Define your inputs
    my_gds_list = [
        "pixelated_device_ec5ebcb9.gds",
        # "another_device.gds",
    ]
    
    my_cell_names = [
        "pixelated_device_ec5ebcb9",
        # "another_device_top",
    ]

    # 4. Run simulations
    s_files = []
    for gds, cell in zip(my_gds_list, my_cell_names):
        print(f"--- Starting Simulation for: {cell} ---")
        result = runner.run_single_emx(gds, cell)
        if result:
            s_files.append(result)
            print(f"Success: {result}")
        else:
            print(f"Failed: {cell}")
    
    print("\n--- Complete ---")
    print("Generated S-parameter files:")
    for f in s_files:
        print(f)