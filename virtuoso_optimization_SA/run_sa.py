#!/usr/bin/env python3
"""
run_sa.py — Entry-point script for the switchable-filter SA optimiser.

Usage
-----
    python run_sa.py                               # uses sa_config.yaml in this dir
    python run_sa.py path/to/my_sa_config.yaml     # explicit config path
    python run_sa.py --config path/to/config.yaml  # flag form

The script runs the full SA loop and prints the best design + cost at the
end.  All intermediate results are saved to the ``output_dir`` defined in
the config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the parent package importable when run directly as a script
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from virtuoso_optimization_SA.sa_optimizer import run_sa  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulated Annealing optimiser for a pixelated switchable RF filter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_HERE / "sa_config.yaml"),
        help="Path to the SA config YAML file (default: sa_config.yaml next to this script).",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help="Alternative flag form for the config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config_flag or args.config)

    if not config_path.exists():
        print(f"[run_sa] ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[run_sa] Using config: {config_path}")
    best_design, best_cost = run_sa(config_path)

    print("\n" + "=" * 70)
    print("SA OPTIMISATION COMPLETE")
    print("=" * 70)
    print(f"Best cost : {best_cost:.4f}")
    print(f"Best design:\n{best_design}")


if __name__ == "__main__":
    main()
