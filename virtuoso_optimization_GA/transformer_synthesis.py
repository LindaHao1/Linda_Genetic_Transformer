#!/usr/bin/env python3
"""Create and optionally run a transformer synthesis job.

The command accepts desired lumped parameters and an initial transformer
geometry, writes a complete YAML job, and dispatches the existing
Virtuoso/EMX genetic optimizer when ``--run`` is supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from virtuoso_engine.config_loader import load_config  # noqa: E402
from virtuoso_optimization_GA.ga_optimizer import run_ga  # noqa: E402


def build_job(args: argparse.Namespace) -> dict:
    """Return a resolved synthesis configuration from command-line inputs."""
    cfg = load_config(args.base_config)
    cfg["mode"] = "oct_transformer"
    cfg["target"] = {
        "Lp": args.lp,
        "Ls": args.ls,
        "k": args.k,
        "Qp": args.qp,
        "Qs": args.qs,
    }
    cfg.setdefault("optimizer", {})["seed_design"] = {
        "pri_inner_r": args.pri_inner_r,
        "pri_width": args.pri_width,
        "sec_inner_r": args.sec_inner_r,
        "sec_width": args.sec_width,
    }
    cfg["optimizer"]["population_size"] = args.population
    cfg["optimizer"]["generations"] = args.generations
    cfg["optimizer"]["n_parallel"] = args.workers
    cfg.setdefault("parallel", {})["num_workers"] = args.workers
    cfg.setdefault("outputs", {})["run_name"] = args.run_name
    return cfg


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a transformer geometry matching target Lp/Ls/k/Q using Virtuoso and EMX."
    )
    p.add_argument("--base-config", type=Path, required=True,
                   help="Site-specific YAML containing PDK, Virtuoso, EMX, and search-range settings.")
    p.add_argument("--output-config", type=Path, default=Path("transformer_job.yaml"))
    p.add_argument("--run-name", default="transformer_synthesis")
    p.add_argument("--run", action="store_true", help="Run the GA after writing the job YAML.")

    target = p.add_argument_group("desired lumped parameters")
    target.add_argument("--lp", type=float, required=True, help="Primary inductance in H.")
    target.add_argument("--ls", type=float, required=True, help="Secondary inductance in H.")
    target.add_argument("--k", type=float, required=True, help="Coupling coefficient.")
    target.add_argument("--qp", type=float, required=True, help="Desired primary Q.")
    target.add_argument("--qs", type=float, required=True, help="Desired secondary Q.")

    geometry = p.add_argument_group("initial geometry in micrometres")
    geometry.add_argument("--pri-inner-r", type=float, required=True)
    geometry.add_argument("--pri-width", type=float, required=True)
    geometry.add_argument("--sec-inner-r", type=float, required=True)
    geometry.add_argument("--sec-width", type=float, required=True)

    search = p.add_argument_group("search controls")
    search.add_argument("--population", type=int, default=12)
    search.add_argument("--generations", type=int, default=20)
    search.add_argument("--workers", type=int, default=1)
    return p


def validate(args: argparse.Namespace) -> None:
    if not 0.0 < args.k <= 1.0:
        raise SystemExit("--k must be in (0, 1].")
    for name in ("lp", "ls", "qp", "qs", "pri_inner_r", "pri_width", "sec_inner_r", "sec_width"):
        if getattr(args, name) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    if args.population < 2 or args.generations < 1 or args.workers < 1:
        raise SystemExit("population must be >=2; generations and workers must be >=1.")


def main() -> None:
    args = parser().parse_args()
    validate(args)
    cfg = build_job(args)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Wrote transformer synthesis job: {args.output_config}")
    if args.run:
        design, cost = run_ga(args.output_config)
        print(f"Best cost: {cost:.6g}")
        print(f"Best geometry:\n{design}")


if __name__ == "__main__":
    main()
