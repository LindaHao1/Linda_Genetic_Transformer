"""
virtuoso_engine — Python driver for parametric RF pixel layout + simulation.

Quick start::

    from virtuoso_engine import VirtuosoPixelEngine

    engine = VirtuosoPixelEngine("virtuoso_engine/config.yaml")

    # Single run with default config
    result = engine.run()

    # Override inline
    result = engine.run({"nmos": {"width": "6u", "fingers": 8}})

    # Sweep a single parameter
    results = engine.sweep(["nmos", "width"], ["3u", "6u", "9u"])

    # Sweep multiple parameters at once (one override dict per run)
    results = engine.sweep_configs([
        {"nmos": {"width": "3u", "fingers": 4}},
        {"nmos": {"width": "6u", "fingers": 8}},
        {"nmos": {"width": "9u", "fingers": 8}, "mim_cap": {"length": "15u"}},
    ])
"""

from .engine import VirtuosoPixelEngine, RunResult
from .runner import VirtuosoResult, PersistentVirtuosoRunner
from .plotter import plot_s11_results

__all__ = [
    "VirtuosoPixelEngine",
    "RunResult",
    "VirtuosoResult",
    "PersistentVirtuosoRunner",
    "plot_s11_results",
]
