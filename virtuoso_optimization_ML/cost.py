"""
cost.py — ML-local re-export of SA cost utilities.

Provides a stable import boundary so the rest of the ML package can import
cost functions from here instead of directly from virtuoso_optimization_SA.
"""

from virtuoso_optimization_SA.cost import (
    FilterStateSpec,
    specs_from_config,
    compute_total_cost,
)

__all__ = ["FilterStateSpec", "specs_from_config", "compute_total_cost"]
