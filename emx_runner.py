import os
import subprocess
from typing import List, Optional

class EMXSimInterface:
    def __init__(
        self, 
        destination_folder: str,
        emx_bin_path: str = "emx",
        proc_file_path: str = "/disk/smiao23/GF22/PDK_240627_121855_281/22FDX-PLUS/V1.0_2.3/Emagnetic/EMX/10M_2Mx_5Cx_1Jx_2Qx_LBthick/22fdsoi_10M_2Mx_5Cx_1Jx_2Qx_LBthick_nominal_detailed.encrypted.proc"
    ):
        """
        Initialize the EMX Interface with your specific environment paths.
        """
        self.emx_bin = emx_bin_path
        self.proc_file = proc_file_path
        self.dest_folder = os.path.abspath(destination_folder)
        
        # Ensure destination folder exists
        os.makedirs(self.dest_folder, exist_ok=True)

    def run_batch(self, gds_files: List[str], cell_names: List[str]) -> List[str]:
        """
        Iterates through the list of GDS files and Cell names, running EMX for each.
        """
        results = []
        
        if len(gds_files) != len(cell_names):
            raise ValueError("The list of GDS files and Cell names must be the same length.")

        for gds, cell in zip(gds_files, cell_names):
            print(f"--- Starting Simulation for: {cell} ---")
            result_file = self._run_single_emx(gds, cell)
            if result_file:
                results.append(result_file)
                print(f"Success: {result_file}")
            else:
                print(f"Failed: {cell}")
        
        return results

    def _run_single_emx(self, gds_file: str, cell_name: str) -> Optional[str]:
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
        # Based on your command: -p P000=P1 -p P001=P2 ...
        # This maps the Layout Label (Value) to the S-Param Port Name (Key)
        port_flags = [
            "-p", "P000=P1",
            "-p", "P001=P1_G",
            "-p", "P002=P2",
            "-p", "P003=P2_G"
        ]

        # 3. Construct the Command List
        cmd = [
            self.emx_bin,
            abs_gds_path,                   # Input GDS
            cell_name,                      # Top Cell Name
            self.proc_file,                 # Process File
            
            # --- Meshing & Geometry ---
            "-e", "0.25", 
            "-t", "0.25", 
            "-v", "0.5", 
            "--3d=-*",                      # Exclude 3D export for specific layers if needed
            
            # --- Ports ---
            *port_flags,
            
            # --- Sweep Settings (0 to 60GHz, 500MHz step) ---
            "--sweep", "0", "6e+10", 
            "--sweep-stepsize", "5e+08",
            
            # --- Logging & Verbosity ---
            "--verbose=3",
            "--print-command-line",
            "--log-time-level=0",
            "--extended-warnings",
            f"--message-file={log_file}",
            
            # --- Simulation Flags ---
            "-l", "0",                      # Layer 0 (Substrate/GND usually)
            "--quasistatic",
            "--dump-connectivity",
            "--parallel=4",             
            "--simultaneous-frequencies=0",
            "--recommended-memory",
            "--key=EMXkey",
            
            # --- Output Format ---
            "--format=touchstone2",
            "-s", output_arg                # Result saved to destination folder
        ]

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

    # 4. Run Batch
    s_files = runner.run_batch(my_gds_list, my_cell_names)
    
    print("\n--- Batch Complete ---")
    print("Generated S-parameter files:")
    for f in s_files:
        print(f)