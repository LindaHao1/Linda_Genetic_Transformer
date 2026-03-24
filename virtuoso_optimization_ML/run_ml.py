"""
run_ml.py — Entry point for the ML-based switchable filter optimizer.

Usage
-----
    python -m virtuoso_optimization_ML.run_ml ml_config.yaml
    python run_ml.py ml_config.yaml [--resume /path/to/checkpoint]

Options
-------
    config_path   : path to the YAML configuration file.
    --resume DIR  : resume training from a checkpoint directory produced by
                    a previous run (loads models, optimizers, and buffer).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_ml(config_path: str, resume_from: str | None = None) -> None:
    from virtuoso_optimization_ML.training.online_trainer import OnlineMLTrainer

    trainer = OnlineMLTrainer(config_path)

    if resume_from is not None:
        trainer.load_checkpoint(resume_from)

    trainer.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML-based online optimizer for pixelated RFIC filter design."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(Path(__file__).parent / "ml_config.yaml"),
        help="Path to ml_config.yaml (default: ml_config.yaml next to this script).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume training from a checkpoint directory.",
    )
    args = parser.parse_args()
    run_ml(args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()
