#!/usr/bin/env python3
"""
data_generation.py — Collect simulation data for world model training.

Generates random switchable-filter designs and simulates each one across
multiple switch states using the parallel Virtuoso engine.  Results are
written to a JSONL dataset file alongside the raw s-parameter files produced
by the engine.

Each JSONL record contains the full DesignState fields (matrix, switch_states,
cap_values, …) plus a list of paths to the per-state .s2p files so that
S-parameter data can be loaded at training time via scikit-rf.

Usage
-----
    python data_generation.py --count 1000
    python data_generation.py --count 500 --num-workers 8 --seed 0
    python data_generation.py --count 200 --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Allow imports from the project root when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from virtuoso_engine import ParallelVirtuosoEngine          # noqa: E402
from virtuoso_engine.config_loader import load_config       # noqa: E402
from virtuoso_optimization_SA.feasibility import generate_random_design  # noqa: E402

DEFAULT_CONFIG = Path(__file__).parent / "wm_data_collection.yaml"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate simulation dataset for world model training."
    )
    p.add_argument(
        "--count", type=int, required=True,
        help="Number of random designs to generate and simulate.",
    )
    p.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG}).",
    )
    p.add_argument(
        "--num-workers", type=int, default=None,
        help="Override parallel.num_workers from the config file.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the output directory from the config file.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _extract_grid_params(raw: dict) -> dict:
    """Pull all design-generation parameters out of the loaded config dict."""
    g = raw.get("grid", {})
    return {
        "rows":                   int(g.get("rows",               21)),
        "cols":                   int(g.get("cols",               27)),
        "n_states":               int(g.get("n_states",           10)),
        "max_inductors":          int(g.get("max_inductors",        8)),
        "n_max_vias":             int(g.get("n_max_vias",           5)),
        "init_min_switches":      int(g.get("init_min_switches",    4)),
        "init_max_switches":      int(g.get("init_max_switches",    6)),
        "init_min_inductors":     int(g.get("init_min_inductors",   4)),
        "init_max_inductors":     int(g.get("init_max_inductors",   8)),
        "init_min_vias":          int(g.get("init_min_vias",        0)),
        "init_max_vias":          int(g.get("init_max_vias",        0)),
        "cap_min":              float(g.get("cap_min",         100e-15)),
        "cap_max":              float(g.get("cap_max",        1000e-15)),
        "n_inductor_values":      int(g.get("n_inductor_values",    1)),
        "single_inductor_value": bool(g.get("single_inductor_value", True)),
        "enable_cap_pairs":      bool(g.get("enable_cap_pairs",    True)),
        "max_cap_pairs":          int(g.get("max_cap_pairs",        8)),
        "cap_pair_init_min":      int(g.get("cap_pair_init_min",    0)),
        "cap_pair_init_max":      int(g.get("cap_pair_init_max",    4)),
        "cap_pair_min":         float(g.get("cap_pair_min",      50e-15)),
        "cap_pair_max":         float(g.get("cap_pair_max",    1000e-15)),
        "cap_pair_min_dist":      int(g.get("cap_pair_min_dist",    2)),
        "cap_pair_max_dist":      int(g.get("cap_pair_max_dist",    3)),
        "inductor_cap_min":     float(g.get("inductor_cap_min",  200e-15)),
        "inductor_cap_max":     float(g.get("inductor_cap_max", 1000e-15)),
    }


# ---------------------------------------------------------------------------
# Design generation
# ---------------------------------------------------------------------------

def _generate_designs(g: dict, count: int, rng: random.Random) -> list:
    """Generate *count* feasible random DesignState objects."""
    designs = []
    attempts = 0
    while len(designs) < count:
        attempts += 1
        design = generate_random_design(
            rows                  = g["rows"],
            cols                  = g["cols"],
            n_switches            = rng.randint(g["init_min_switches"], g["init_max_switches"]),
            max_inductors         = g["max_inductors"],
            n_initial_inductors   = rng.randint(g["init_min_inductors"], g["init_max_inductors"]),
            n_states              = g["n_states"],
            cap_min               = g["cap_min"],
            cap_max               = g["cap_max"],
            rng                   = rng,
            n_inductor_values     = g["n_inductor_values"],
            single_inductor_value = g["single_inductor_value"],
            n_init_vias           = rng.randint(g["init_min_vias"], g["init_max_vias"]),
            max_vias              = g["n_max_vias"],
            max_cap_pairs         = g["max_cap_pairs"] if g["enable_cap_pairs"] else 0,
            cap_pair_init_min     = g["cap_pair_init_min"] if g["enable_cap_pairs"] else 0,
            cap_pair_init_max     = g["cap_pair_init_max"] if g["enable_cap_pairs"] else 0,
            cap_pair_min          = g["cap_pair_min"],
            cap_pair_max          = g["cap_pair_max"],
            cap_pair_min_dist     = g["cap_pair_min_dist"],
            cap_pair_max_dist     = g["cap_pair_max_dist"],
            inductor_cap_min      = g["inductor_cap_min"],
            inductor_cap_max      = g["inductor_cap_max"],
        )
        if design is not None:
            designs.append(design)
            if len(designs) % 100 == 0:
                print(f"  Generated {len(designs)}/{count} designs ({attempts} total attempts).")

    print(f"Generated {count} designs in {attempts} total attempts.")
    return designs


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _build_override(design, tag: str, sims_dir: Path) -> dict:
    return {
        "mode":      "pixelated_design_simple",
        "pixelated": design.to_pixelated_config(),
        "project":   {"parent_output_dir": str(sims_dir / tag)},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    raw = load_config(args.config)
    g   = _extract_grid_params(raw)

    virtuoso_config = Path(raw["virtuoso_config"])
    num_workers = args.num_workers or raw.get("parallel", {}).get("num_workers", 1)

    output_dir = args.output_dir or Path(
        raw.get("output_dir", "/disk/mcao53/Sim_engine_results/world_model_data")
    )
    sims_dir     = output_dir / "sims"
    dataset_file = output_dir / "dataset.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    sims_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print("=" * 60)
    print("World Model Data Generation")
    print("=" * 60)
    print(f"Config      : {args.config}")
    print(f"Virt config : {virtuoso_config}")
    print(f"Grid        : {g['rows']}×{g['cols']}, n_states={g['n_states']}")
    print(f"Workers     : {num_workers}")
    print(f"Designs     : {args.count}")
    print(f"Output dir  : {output_dir}")
    print(f"Dataset file: {dataset_file}")
    print()

    t_start = time.time()

    # --- Phase 1: generate random designs ---
    print(f"[1/2] Generating {args.count} random designs...")
    t0 = time.time()
    designs = _generate_designs(g, args.count, rng)
    print(f"      Done in {time.time() - t0:.1f}s.\n")

    # --- Phase 2: simulate all designs ---
    overrides = [
        _build_override(d, f"design_{i:06d}", sims_dir)
        for i, d in enumerate(designs)
    ]

    log_file = output_dir / "data_generation.log"
    print(f"[2/2] Simulating {args.count} designs...")
    t1 = time.time()
    with ParallelVirtuosoEngine(virtuoso_config, num_workers=num_workers, quiet=True, log_path=log_file) as engine:
        results = engine.run_batch(overrides)
    print(f"      Done in {time.time() - t1:.1f}s.\n")

    # --- Save dataset ---
    n_success = 0
    n_fail    = 0
    with dataset_file.open("w") as f:
        for i, (design, result) in enumerate(zip(designs, results)):
            tag = f"design_{i:06d}"
            record = {
                "tag":                    tag,
                "success":                result.success,
                # DesignState fields
                "matrix":                 design.matrix,
                "switch_states":          design.switch_states,
                "cap_values":             design.cap_values,
                "inductor_values":        design.inductor_values,
                "inductor_switch_states": design.inductor_switch_states,
                "cap_pixel_pairs":        design.cap_pixel_pairs,
                "cap_pair_values":        design.cap_pair_values,
                "inductor_cap_values":    design.inductor_cap_values,
                # Simulation output paths (load with skrf at training time)
                "s_param_files": [str(p) for p in result.s_param_files] if result.success else [],
                "sim_dir":       str(result.run_dir) if result.success else None,
            }
            f.write(json.dumps(record) + "\n")
            if result.success:
                n_success += 1
            else:
                n_fail += 1

    total = time.time() - t_start
    print(f"Dataset saved: {dataset_file}")
    print(f"  Success : {n_success}/{args.count}")
    print(f"  Failed  : {n_fail}/{args.count}")
    print(f"  Total   : {total:.1f}s  ({total / args.count:.2f}s/design)")


if __name__ == "__main__":
    main()
