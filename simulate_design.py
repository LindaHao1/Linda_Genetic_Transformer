"""
simulate_design.py — Run a simulation from a saved config.yaml or best_design.json.

Single-file mode
----------------
    # From a full config.yaml (used directly as the engine config):
    python simulate_design.py path/to/config.yaml

    # From a best_design.json (pixelated params overlaid onto base config):
    python simulate_design.py path/to/best_design.json

    # Optionally set a custom output dir:
    python simulate_design.py path/to/best_design.json --output_dir /tmp/my_run

Batch mode (folder of configs)
-------------------------------
    # Simulate every .yaml / .json in a folder, each in its own subfolder:
    python simulate_design.py path/to/configs/ --batch

    # Custom output root and worker count:
    python simulate_design.py path/to/configs/ --batch \\
        --output_dir /tmp/batch_results --num_workers 4

    Each config file named ``foo.yaml`` or ``bar.json`` will write its
    simulation outputs to ``<output_dir>/foo/`` and ``<output_dir>/bar/``
    respectively.  All configs run concurrently via ParallelVirtuosoEngine.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

from virtuoso_engine.config_loader import load_config

from virtuoso_engine import VirtuosoPixelEngine
from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine

BASE_CONFIG = "virtuoso_engine/config.yaml"


def load_from_config_yaml(path: str) -> dict:
    """Load a full simulation config.yaml — return it as the override dict."""
    cfg = load_config(path)

    # Strip internal/worker keys that are not valid engine overrides
    for key in ("_worker_id",):
        cfg.pop(key, None)

    return cfg


def load_from_best_design_json(path: str) -> dict:
    """Convert a best_design.json into an engine override dict."""
    with open(path) as f:
        d = json.load(f)

    overrides: dict = {
        "pixelated": {
            "pixel_state_matrix":    d["matrix"],
            "state_list":            d["switch_states"],
            "cap_values":            d["cap_values"],
            "inductor_values":       d["inductor_values"],
            "inductor_switch_states": d["inductor_switch_states"],
        }
    }
    return overrides


def _strip_worker_suffix(name: str) -> str:
    """Remove a trailing _wN suffix that the parallel engine may have appended."""
    return re.sub(r"_w\d+$", "", name)


def _collect_configs(folder: Path) -> List[Path]:
    """Return all .yaml/.yml/.json files in *folder* (sorted by name)."""
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in (".yaml", ".yml", ".json")
    )
    return files


def _build_override(cfg_file: Path, output_dir: str) -> dict:
    """Return an engine override dict for *cfg_file* with output rooted at *output_dir*."""
    suffix = cfg_file.suffix.lower()
    if suffix in (".yaml", ".yml"):
        ov = load_from_config_yaml(str(cfg_file))
    else:
        ov = load_from_best_design_json(str(cfg_file))
    ov.setdefault("project", {})
    ov["project"]["parent_output_dir"] = output_dir
    # Strip any _wN suffix the parallel engine may have baked into the saved lib name;
    # _apply_worker_config will re-append the correct suffix for the current worker.
    if "lib_name" in ov["project"]:
        ov["project"]["lib_name"] = _strip_worker_suffix(ov["project"]["lib_name"])
    return ov


def run_batch(folder: Path, output_dir: str | None, num_workers: int | None) -> None:
    """Simulate every config in *folder* in parallel; each gets its own subfolder."""
    cfg_files = _collect_configs(folder)
    if not cfg_files:
        print(f"ERROR: no .yaml/.yml/.json files found in {folder}", file=sys.stderr)
        sys.exit(1)

    # Default: create per-config subfolders inside the configs folder itself
    base_output = Path(output_dir) if output_dir else folder

    print(f"Batch folder : {folder}")
    print(f"Output root  : {base_output}")
    print(f"Config files : {len(cfg_files)}")
    for f in cfg_files:
        print(f"  {f.name} -> {base_output / f.stem}/")

    overrides = [
        _build_override(f, str(base_output / f.stem))
        for f in cfg_files
    ]

    log_path = base_output / "batch_simulate.log"
    base_output.mkdir(parents=True, exist_ok=True)

    with ParallelVirtuosoEngine(
        BASE_CONFIG,
        num_workers=num_workers,
        log_path=log_path,
    ) as engine:
        results = engine.run_batch(overrides)

    print("\n--- Batch Simulation Summary ---")
    for cfg_file, result in zip(cfg_files, results):
        status = "OK  " if result.success else "FAIL"
        print(f"  [{status}] {cfg_file.name}: {result.s_param_files}")


def main():
    parser = argparse.ArgumentParser(description="Simulate a saved design or a batch of designs.")
    parser.add_argument(
        "design_path",
        help="Path to a config.yaml / best_design.json (single mode) "
             "or a folder of such files (batch mode).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat design_path as a folder and simulate all configs inside it in parallel.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory.  In batch mode this is the root under which "
             "per-config subfolders are created.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel Virtuoso workers (batch mode only; "
             "falls back to parallel.num_workers in the config).",
    )
    args = parser.parse_args()

    design_path = Path(args.design_path)
    if not design_path.exists():
        print(f"ERROR: path not found: {design_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ batch
    if args.batch or design_path.is_dir():
        if not design_path.is_dir():
            print(f"ERROR: --batch requires a directory, got: {design_path}", file=sys.stderr)
            sys.exit(1)
        run_batch(design_path, args.output_dir, args.num_workers)
        return

    # ----------------------------------------------------------------- single
    suffix = design_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        print(f"Loading full config from: {design_path}")
        # A full config.yaml already includes project/layout/simulation settings,
        # so we use it directly as the engine config (not as overrides on base config).
        engine_config = str(design_path)
        overrides = {}   # nothing extra to override — the file IS the config
    elif suffix == ".json":
        print(f"Loading best_design from: {design_path}")
        overrides = load_from_best_design_json(str(design_path))
        engine_config = BASE_CONFIG
    else:
        print(f"ERROR: unsupported file type '{suffix}'. Expected .yaml or .json", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        overrides.setdefault("project", {})["parent_output_dir"] = args.output_dir

    print(f"Engine config : {engine_config}")
    print(f"Overrides keys: {list(overrides.keys()) or '(none)'}")

    with VirtuosoPixelEngine(engine_config) as engine:
        result = engine.run(overrides)

    print("\n--- Simulation Result ---")
    print(f"Success      : {result.success}")
    print(f"S-param files: {result.s_param_files}")


if __name__ == "__main__":
    main()
