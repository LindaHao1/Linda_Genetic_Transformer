"""
Main Simulation Engine with parallel processing capability.
"""

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Callable
from datetime import datetime
import numpy as np
import gdsfactory as gf
from pathlib import Path

from simulation_engine.pixel_gds import draw_pixelated_device
from simulation_engine.emx_runner import EMXSimInterface
from simulation_engine.snp_processing import process_snp_file
from simulation_engine.layers import LayerMap
from .job import SimulationJob, SimulationResult


class SimulationEngine:
    """
    A simulation engine that initializes once and processes jobs in parallel.
    
    Uses a producer-consumer pattern where:
    - Producer: Submits pixel matrices as jobs
    - Consumer: Processes jobs in parallel using a thread pool
    """
    
    def __init__(
        self,
        base_output_dir: str = "./simulation_results",
        max_parallel: int = 4,
        rows: int = 21,
        cols: int = 30,
        pitch_x: float = 30.0,
        pitch_y: float = 30.0,
        pixel_w: float = 34.0,
        pixel_h: float = 34.0,
        tl_w: float = 34.0,
        tl_l: float = 60.0,
        port_stick_len: float = 5.0,
    ):
        """
        Initialize the simulation engine.
        
        Args:
            base_output_dir: Root directory for all simulation outputs
            max_parallel: Maximum number of parallel simulations
            rows, cols: Grid dimensions
            pitch_x, pitch_y: Pixel spacing
            pixel_w, pixel_h: Pixel dimensions
            tl_w, tl_l: Transmission line parameters
            port_stick_len: Port stick length
        """
        print("🚀 Initializing Simulation Engine...")
        
        # Store configuration
        self.base_output_dir = Path(base_output_dir)
        self.max_parallel = max_parallel
        self.rows = rows
        self.cols = cols
        self.pitch_x = pitch_x
        self.pitch_y = pitch_y
        self.pixel_w = pixel_w
        self.pixel_h = pixel_h
        self.tl_w = tl_w
        self.tl_l = tl_l
        self.port_stick_len = port_stick_len
        
        # Create directory structure
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        (self.base_output_dir / "gds_files").mkdir(exist_ok=True)
        (self.base_output_dir / "snp_files").mkdir(exist_ok=True)
        (self.base_output_dir / "matrices").mkdir(exist_ok=True)
        (self.base_output_dir / "plots").mkdir(exist_ok=True)
        (self.base_output_dir / "logs").mkdir(exist_ok=True)
        
        # ONE-TIME INITIALIZATION (the slow parts)
        print("  ⏳ Activating PDK...")
        gf.gpdk.PDK.activate()
        
        print("  ⏳ Loading LayerMap...")
        self.layermap = LayerMap
        
        print("  ⏳ Setting up EMX interface...")
        # Import config to calculate per-simulation resources
        from . import emx_config
        
        # Calculate resources per simulation based on max_parallel
        parallel_threads = emx_config.get_parallel_threads(max_parallel)
        memory_per_sim = emx_config.get_memory_setting(max_parallel)
        
        self.emx_runner = EMXSimInterface(
            destination_folder=str(self.base_output_dir / "snp_files"),
            parallel_threads=parallel_threads,
            max_memory=memory_per_sim
        )
        
        print(f"  📊 Per-simulation resources:")
        print(f"     Threads: {parallel_threads if parallel_threads > 0 else 'auto-detect'}")
        print(f"     Memory: {memory_per_sim}")
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=max_parallel)
        
        # Lock for thread-safe plotting
        # gdsfactory's plot() function loads/creates layer property files (.lyp) which
        # causes race conditions when multiple threads try to access them simultaneously.
        # This lock ensures only one thread generates plots at a time.
        self.plot_lock = threading.Lock()
        
        print(f"✅ Engine initialized (max_parallel={max_parallel})")
        print(f"   Output directory: {self.base_output_dir.absolute()}")
    
    def submit_job(self, job: SimulationJob) -> threading.Thread:
        """
        Submit a single job for asynchronous processing.
        
        Args:
            job: SimulationJob containing the design matrix
            
        Returns:
            Future object that can be used to check status and get results
        """
        future = self.executor.submit(self._process_job, job)
        return future
    
    def submit_batch(
        self,
        jobs: List[SimulationJob],
        callback: Optional[Callable[[SimulationResult], None]] = None
    ) -> List[SimulationResult]:
        """
        Submit multiple jobs and wait for all to complete.
        
        Args:
            jobs: List of SimulationJob objects
            callback: Optional callback function called when each job completes
            
        Returns:
            List of SimulationResult objects
        """
        print(f"\n📦 Submitting batch of {len(jobs)} jobs...")
        
        futures = {self.executor.submit(self._process_job, job): job 
                   for job in jobs}
        
        results = []
        completed = 0
        
        for future in as_completed(futures):
            completed += 1
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                status_icon = "✅" if result.success else "❌"
                print(f"{status_icon} [{completed}/{len(jobs)}] Job {result.job_id} completed")
                
                if callback:
                    callback(result)
                    
            except Exception as e:
                print(f"❌ [{completed}/{len(jobs)}] Job {job.job_id} failed with exception: {e}")
                results.append(SimulationResult(
                    job_id=job.job_id,
                    success=False,
                    error_message=str(e)
                ))
        
        print(f"\n🎉 Batch complete! {sum(r.success for r in results)}/{len(results)} succeeded")
        return results
    
    def _process_job(self, job: SimulationJob) -> SimulationResult:
        """
        Process a single simulation job.
        
        This is the worker function that runs in the thread pool.
        """
        try:
            print(f"  🔧 Processing job {job.job_id}...")
            
            # Generate unique name for this job
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            device_name = f"device_{job.job_id}_{timestamp}"
            
            # Paths for outputs
            gds_path = self.base_output_dir / "gds_files" / f"{device_name}.gds"
            matrix_path = self.base_output_dir / "matrices" / f"{device_name}.npy"
            matrix_plot = self.base_output_dir / "plots" / f"{device_name}_binary.png"
            device_plot = self.base_output_dir / "plots" / f"{device_name}.png"
            
            # Draw the device
            device, _ = draw_pixelated_device(
                design_matrix=job.design_matrix,
                layer_top=self.layermap.LB,
                layer_ground=self.layermap.JA,
                layer_top_port=self.layermap.LB_PIN,
                layer_ground_port=self.layermap.JA_PIN,
                tl_w=self.tl_w,
                pitch_x=self.pitch_x,
                pitch_y=self.pitch_y,
                pixel_w=self.pixel_w,
                pixel_h=self.pixel_h,
                tl_l=self.tl_l,
                port_stick_len=self.port_stick_len,
                device_name=device_name
            )
            
            # Save GDS
            device.write_gds(str(gds_path))
            
            # Save matrix
            np.save(str(matrix_path), job.design_matrix)
            
            # Generate plots
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for thread safety
            import matplotlib.pyplot as plt
            
            # Thread-safe plotting: Use lock to prevent race conditions in gdsfactory's plot()
            # which accesses shared layer property files
            with self.plot_lock:
                # Binary matrix plot
                fig1, ax1 = plt.subplots(figsize=(10, 7))
                ax1.imshow(job.design_matrix, cmap='gray_r', interpolation='nearest')
                ax1.set_title(f"Design Matrix - Job {job.job_id}")
                ax1.set_xlabel("Columns")
                ax1.set_ylabel("Rows")
                fig1.savefig(str(matrix_plot))
                plt.close(fig1)
                
                # Device layout plot
                fig2 = device.plot(return_fig=True)
                fig2.savefig(str(device_plot))
                plt.close(fig2)
            
            # Run EMX simulation
            snp_path = self.emx_runner.run_single_emx(
                gds_file=str(gds_path),
                cell_name=device_name
            )
            
            if not snp_path:
                raise RuntimeError("EMX simulation failed to produce output file")
            
            # Process S-parameters
            snp_analysis_dir = self.base_output_dir / "snp_files" / device_name
            snp_analysis_dir.mkdir(exist_ok=True)
            process_snp_file(snp_path, save_path=str(snp_analysis_dir))
            
            # Create result
            result = SimulationResult(
                job_id=job.job_id,
                success=True,
                gds_path=str(gds_path),
                snp_path=snp_path,
                matrix_path=str(matrix_path),
                plot_paths={
                    'matrix': str(matrix_plot),
                    'device': str(device_plot)
                },
                metadata={
                    'device_name': device_name,
                    'rows': self.rows,
                    'cols': self.cols,
                    **job.metadata
                }
            )
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            
            # Log error
            log_path = self.base_output_dir / "logs" / f"{job.job_id}_error.log"
            with open(log_path, 'w') as f:
                f.write(error_msg)
            
            return SimulationResult(
                job_id=job.job_id,
                success=False,
                error_message=error_msg
            )
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the engine and clean up resources.
        
        Args:
            wait: If True, wait for all pending jobs to complete
        """
        print("🛑 Shutting down Simulation Engine...")
        self.executor.shutdown(wait=wait)
        print("✅ Shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)
