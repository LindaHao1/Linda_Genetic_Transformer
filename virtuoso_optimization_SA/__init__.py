"""
virtuoso_optimization_SA — Simulated Annealing optimiser for pixelated switchable RF filters.

Public API
----------
    from virtuoso_optimization_SA import SwitchableFilterSA, run_sa
    from virtuoso_optimization_SA import DesignState, generate_random_design
    from virtuoso_optimization_SA import is_feasible
    from virtuoso_optimization_SA import compute_total_cost, FilterStateSpec, specs_from_config

Quick start::

    from virtuoso_optimization_SA import run_sa
    best_design, best_cost = run_sa("virtuoso_optimization_SA/sa_config.yaml")
"""

from .design      import DesignState
from .feasibility import (
    is_feasible,
    generate_random_design,
    check_connectivity,
    check_switch_spacing,
    check_inductor_spacing,
)
from .actions     import random_action
from .cost        import (
    compute_total_cost,
    FilterStateSpec,
    specs_from_config,
    load_s2p,
    get_s21_db,
)
from .sa_optimizer import SwitchableFilterSA, run_sa

__all__ = [
    "DesignState",
    "is_feasible",
    "generate_random_design",
    "check_connectivity",
    "check_switch_spacing",
    "check_inductor_spacing",
    "random_action",
    "compute_total_cost",
    "FilterStateSpec",
    "specs_from_config",
    "load_s2p",
    "get_s21_db",
    "SwitchableFilterSA",
    "run_sa",
]
