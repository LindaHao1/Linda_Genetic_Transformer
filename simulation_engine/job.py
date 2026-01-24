"""
Job and Result definitions for the simulation engine.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class SimulationJob:
    """Represents a simulation job to be processed."""
    
    job_id: str
    design_matrix: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.design_matrix, np.ndarray):
            raise TypeError("design_matrix must be a numpy array")


@dataclass
class SimulationResult:
    """Represents the result of a completed simulation."""
    
    job_id: str
    success: bool
    gds_path: Optional[str] = None
    snp_path: Optional[str] = None
    matrix_path: Optional[str] = None
    plot_paths: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"SimulationResult(job_id={self.job_id}, status={status})"
