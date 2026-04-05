"""
warmup.py — Step 0 of the online ML training loop.

Generates ~cfg.warmup_n_designs random feasible designs, simulates them all
via the shared ParallelVirtuosoEngine, and populates the replay buffer with
real data.

This data is used to warm up both the generative model and the world model
before the iterative online loop begins.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np
import yaml

from ..feasibility import generate_random_design
from ..cost import compute_total_cost
from ..design import DesignState

from ..dataset.replay_buffer import ReplayBuffer, make_infeasible_sentinel
from ..encoding.design_encoder import DesignEncoder
from ..utils.s_param_utils import result_to_s_param_tensor, load_and_interpolate_s21
from ..utils.cost_utils import compute_cost_from_tensor


def run_warmup(
    cfg,                        # MLConfig
    encoder: DesignEncoder,
    buffer:  ReplayBuffer,
    freq_points: np.ndarray,
    rng: random.Random,
    output_dir: Path,
    engine,                     # ParallelVirtuosoEngine — already started by trainer
    specs=None,                 # list[FilterStateSpec] for computing design score
) -> int:
    """
    Run the warm-up phase: generate random feasible designs, simulate them
    using the shared engine, and add results to the replay buffer as real data.

    Parameters
    ----------
    cfg         : MLConfig instance.
    encoder     : DesignEncoder for converting DesignState → image tensor.
    buffer      : ReplayBuffer to populate.
    freq_points : canonical frequency grid (n_freq,).
    rng         : random.Random for reproducible design generation.
    output_dir  : base directory for simulation outputs.
    engine      : running ParallelVirtuosoEngine shared with the trainer.
    specs       : filter specs for computing design cost (score). When None,
                  scores are stored as inf.

    Returns
    -------
    Number of successfully simulated designs added to buffer.
    """
    n_target = cfg.warmup_n_designs
    sim_dir  = output_dir / "warmup_sims"

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
    print(f"[Warmup] Simulating {len(designs)} designs ...")
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

        cost = float("inf")
        if specs is not None:
            try:
                cost = compute_total_cost(result.s_param_files, specs)
            except Exception:
                pass

        buffer.add(
            image       = image,
            s_params    = s_params,
            is_real     = True,
            is_feasible = True,
            iteration   = 0,
            confidence  = 0.0,
            score       = cost,
        )
        n_success += 1

    print(f"[Warmup] Buffer populated with {n_success}/{len(designs)} real designs.")
    return n_success


def reload_warmup_from_sim_folder(
    sim_dir: Path,
    encoder: DesignEncoder,
    buffer: ReplayBuffer,
    freq_points: np.ndarray,
    specs=None,      # List[FilterStateSpec] — used to compute cost
    n_states: int = 2,
) -> int:
    """
    Reload warmup buffer from an existing simulation folder.

    Scans ``sim_dir`` for ``warmup_{i:04d}/`` subdirectories.  For each one it:
      1. Reads the ``config.yaml`` saved by the simulator to reconstruct a DesignState.
      2. Re-encodes the image using the current encoder configuration.
      3. Loads the ``.s2p`` files (one per state) from ``emx_results/``.
      4. Computes the cost with the **current** filter specs.
      5. Adds the entry to the buffer as real data.

    This is used by the score-based guidance mode so that config changes (e.g. new
    filter specs) are reflected in the warmup dataset without re-running simulations.

    Parameters
    ----------
    sim_dir     : path to the ``warmup_sims/`` directory.
    encoder     : DesignEncoder for converting DesignState → image tensor.
    buffer      : ReplayBuffer to populate.
    freq_points : canonical frequency grid (n_freq,).
    specs       : filter specs for computing design cost. When None, scores stored as inf.
    n_states    : number of filter states.

    Returns
    -------
    Number of successfully loaded designs added to buffer.
    """
    sim_dir = Path(sim_dir)
    if not sim_dir.exists():
        print(f"[Warmup] Sim folder not found: {sim_dir} — cannot reload.")
        return 0

    # Collect all warmup subdirectories in sorted order
    warmup_dirs = sorted(sim_dir.glob("warmup_*/"))
    if not warmup_dirs:
        print(f"[Warmup] No warmup_* subdirectories found in {sim_dir}.")
        return 0

    print(f"[Warmup] Reloading warmup data from {sim_dir}  ({len(warmup_dirs)} dirs found) ...")

    n_success = 0
    for warmup_dir in warmup_dirs:
        # --- Find config.yaml across any worker_N subdirectory ---
        config_paths = sorted(warmup_dir.glob("worker_*/*/config.yaml"))
        if not config_paths:
            continue
        config_path = config_paths[0]

        try:
            cfg_data = yaml.safe_load(config_path.read_text())
            px = cfg_data.get("pixelated", {})
            design = DesignState(
                matrix          = px["pixel_state_matrix"],
                switch_states   = px["state_list"],
                cap_values      = px["cap_values"],
                inductor_values = px.get("inductor_values", []),
            )
        except Exception as e:
            print(f"[Warmup] {warmup_dir.name}: failed to parse config ({e}) — skipping.")
            continue

        # --- Re-encode image ---
        try:
            image = encoder.encode(design)
        except Exception as e:
            print(f"[Warmup] {warmup_dir.name}: encode failed ({e}) — skipping.")
            continue

        # --- Load .s2p files (one per state) across any worker ---
        sim_subdir = config_path.parent
        emx_dir = sim_subdir / "emx_results"
        s_param_rows = []
        ok = True
        for s in range(n_states):
            s2p_files = list(emx_dir.glob(f"*_state{s}.s2p"))
            if not s2p_files:
                s2p_files = sorted(warmup_dir.glob(f"worker_*/*/emx_results/*_state{s}.s2p"))
            if not s2p_files:
                ok = False
                break
            try:
                s_row = load_and_interpolate_s21(s2p_files[0], freq_points)
                s_param_rows.append(s_row)
            except Exception as e:
                print(f"[Warmup] {warmup_dir.name}: s2p load failed state{s} ({e}) — skipping.")
                ok = False
                break

        if not ok or len(s_param_rows) != n_states:
            continue

        import torch
        s_params = torch.stack(s_param_rows, dim=0)   # (n_states, n_freq)

        # --- Compute cost ---
        cost = float("inf")
        if specs is not None:
            try:
                cost = compute_cost_from_tensor(s_params, freq_points, specs)
            except Exception:
                pass

        buffer.add(
            image       = image,
            s_params    = s_params,
            is_real     = True,
            is_feasible = True,
            iteration   = 0,
            confidence  = 0.0,
            score       = cost,
        )
        n_success += 1

    print(f"[Warmup] Reloaded {n_success}/{len(warmup_dirs)} designs from sim folder.")
    return n_success
