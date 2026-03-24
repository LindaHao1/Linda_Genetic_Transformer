"""
warmup.py — Step 0 of the online ML training loop.

Generates ~cfg.warmup_n_designs random feasible designs, simulates them all
via ParallelVirtuosoEngine, and populates the replay buffer with real data.

This data is used to warm up both the generative model and the world model
before the iterative online loop begins.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np

from virtuoso_optimization_SA.feasibility import generate_random_design

from ..dataset.replay_buffer import ReplayBuffer, make_infeasible_sentinel
from ..encoding.design_encoder import DesignEncoder
from ..utils.s_param_utils import result_to_s_param_tensor


def run_warmup(
    cfg,                        # MLConfig
    encoder: DesignEncoder,
    buffer:  ReplayBuffer,
    freq_points: np.ndarray,
    rng: random.Random,
    output_dir: Path,
) -> int:
    """
    Run the warm-up phase: generate random feasible designs, simulate them,
    and add results to the replay buffer as real data.

    Parameters
    ----------
    cfg         : MLConfig instance.
    encoder     : DesignEncoder for converting DesignState → image tensor.
    buffer      : ReplayBuffer to populate.
    freq_points : canonical frequency grid (n_freq,).
    rng         : random.Random for reproducible design generation.
    output_dir  : base directory for simulation outputs.

    Returns
    -------
    Number of successfully simulated designs added to buffer.
    """
    from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine  # deferred

    n_target  = cfg.warmup_n_designs
    n_workers = cfg.parallel_num_workers
    sim_dir   = output_dir / "warmup_sims"

    print(f"\n[Warmup] Generating {n_target} random feasible designs ...")

    # --- Generate random designs ---
    designs = []
    attempts = 0
    max_attempts = n_target * 50
    while len(designs) < n_target and attempts < max_attempts:
        attempts += 1
        d = generate_random_design(
            rows              = cfg.rows,
            cols              = cfg.cols,
            n_switches        = cfg.n_switches,
            max_inductors     = cfg.max_inductors,
            n_states          = cfg.n_states,
            cap_min           = cfg.cap_min,
            cap_max           = cfg.cap_max,
            rng               = rng,
            n_inductor_values = cfg.n_inductor_values,
        )
        if d is not None:
            designs.append(d)

    print(f"[Warmup] Generated {len(designs)} designs in {attempts} attempts.")

    if not designs:
        print("[Warmup] WARNING: No feasible designs generated — skipping simulation.")
        return 0

    # --- Build simulation overrides ---
    overrides: List[dict] = []
    for i, design in enumerate(designs):
        tag = f"warmup_{i:04d}"
        override = {
            "mode":      "pixelated_design_simple",
            "pixelated": design.to_pixelated_config(),
            "project":   {"parent_output_dir": str(sim_dir / tag)},
        }
        overrides.append(override)

    # --- Run simulations in parallel ---
    print(f"[Warmup] Simulating {len(designs)} designs with {n_workers} workers ...")
    with ParallelVirtuosoEngine(
        config_path = cfg.virtuoso_config,
        num_workers = n_workers,
        log_path    = output_dir / "warmup_engine.log",
    ) as engine:
        results = engine.run_batch(overrides)

    # --- Add to buffer ---
    n_success = 0
    for i, (design, result) in enumerate(zip(designs, results)):
        image = encoder.encode(design)

        s_params = result_to_s_param_tensor(result, freq_points, cfg.n_states)
        if s_params is None:
            # Simulation failed — add as infeasible sentinel so the buffer
            # at least has this design's image (skipped in world model training)
            print(f"[Warmup] Design {i}: simulation failed, skipping.")
            continue

        buffer.add(
            image       = image,
            s_params    = s_params,
            is_real     = True,
            is_feasible = True,
            iteration   = 0,
            confidence  = 0.0,
        )
        n_success += 1

    print(f"[Warmup] Buffer populated with {n_success}/{len(designs)} real designs.")
    return n_success
