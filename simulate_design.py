"""
simulate_design.py — Run a simulation from a saved config.yaml or best_design.json.

Usage:
    # From a full config.yaml (used directly as the engine config):
    python simulate_design.py path/to/config.yaml

    # From a best_design.json (pixelated params overlaid onto base config):
    python simulate_design.py path/to/best_design.json

    # Optionally set a custom output dir:
    python simulate_design.py path/to/best_design.json --output_dir /tmp/my_run
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from virtuoso_engine import VirtuosoPixelEngine

BASE_CONFIG = "virtuoso_engine/config.yaml"


def load_from_config_yaml(path: str) -> dict:
    """Load a full simulation config.yaml — return it as the override dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

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


def main():
    parser = argparse.ArgumentParser(description="Simulate a saved design.")
    parser.add_argument("design_file", help="Path to config.yaml or best_design.json")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override parent_output_dir (optional)",
    )
    args = parser.parse_args()

    design_path = Path(args.design_file)
    if not design_path.exists():
        print(f"ERROR: file not found: {design_path}", file=sys.stderr)
        sys.exit(1)

    suffix = design_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        print(f"Loading full config from: {design_path}")
        overrides = load_from_config_yaml(str(design_path))
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
