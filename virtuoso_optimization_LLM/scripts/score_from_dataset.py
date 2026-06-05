#!/usr/bin/env python3
"""
score_from_dataset.py — extract a real PV-score conditioning vector from one
design in the world-model dataset, for reproducing LM inference.

The LM (`scripts/test_generation.py`) is conditioned on a per-state PV-score
vector: K=3 states x 3 scalars [pv_pb_level, pv_contrast, pv_pb_std], passed as
9 raw-dB numbers via `--scores=`.  Those scores are a pure function of each
state's S21 curve, computed with the *same* GA scoring (`src/scoring.py`,
which reuses `virtuoso_optimization_SA/cost.py`) the model was trained on.

This script:
  1. Picks ONE design folder from the dataset (sims_dirs in the config).
  2. Selects K state indices (default 0,1,2; or `--auto-best` to pick the
     distinct state-triple that maximizes the aggregated GA design score).
  3. Loads those states' S21 (`world_model_test/wm_dataset._load_s21`).
  4. Computes the (K, 3) PV-score matrix and prints:
       - the design identity,
       - the per-state score matrix + aggregate design score,
       - the ready-to-paste `--scores=` CSV and full test_generation.py command.

Run (from the LLM project dir, rfic2 python):
    PY=/disk/mcao53/conda_env/rfic2/bin/python
    $PY scripts/score_from_dataset.py --config configs/default.yaml \
        --design design_000000 --auto-best
"""
from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import yaml

# Project root on sys.path so `src.*` imports resolve when run from this dir.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_WM_DIR = "/disk/mcao53/RFIC_RL/world_model_test"
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)

from src.scoring import (  # noqa: E402
    SCORE_SCALARS, design_score, design_score_matrix, load_specs,
)
from wm_dataset import _load_s21          # noqa: E402
from process_dataset import _state_index  # noqa: E402


def _find_sim_dir(design_arg: str, sims_dirs: list[str]) -> Path:
    """Resolve a design argument to the sim directory that holds emx_results.

    Accepts an absolute path (a design_* folder OR the inner sim dir) or a bare
    name like 'design_000123' to be looked up under the config's sims_dirs.
    """
    cand = Path(design_arg)
    search_roots: list[Path] = []
    if cand.is_absolute() and cand.exists():
        search_roots = [cand]
    else:
        for sd in sims_dirs:
            search_roots.append(Path(sd) / design_arg)

    for root in search_roots:
        if (root / "emx_results").is_dir():
            return root
        hits = sorted(root.glob("worker_*/*/emx_results"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        f"Could not locate emx_results for {design_arg!r}. Looked under: "
        + ", ".join(str(r) for r in search_roots)
    )


def _state_s2p_files(sim_dir: Path) -> list[Path]:
    files = sorted(sim_dir.glob("emx_results/*_state*.s2p"), key=_state_index)
    if not files:
        raise FileNotFoundError(f"No *_state*.s2p files under {sim_dir}/emx_results")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--design", default="design_000000",
                    help="design_* name (looked up under sims_dirs) or an absolute path")
    ap.add_argument("--states", default="0,1,2",
                    help="comma-separated state indices to assign to spec slots 0..K-1")
    ap.add_argument("--auto-best", action="store_true",
                    help="ignore --states; pick the distinct state-triple maximizing the GA design score")
    ap.add_argument("--checkpoint", default="runs/sft.pt",
                    help="LM checkpoint to put in the printed inference command")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    K = int(cfg["grid"]["n_states"])
    freqs = np.linspace(float(cfg["freq"]["min"]), float(cfg["freq"]["max"]),
                        int(cfg["freq"]["n"]))
    specs = load_specs(cfg["scoring"]["filter_specs_config"])
    agg = cfg["scoring"].get("aggregation_mode", "soft_max")
    temp = float(cfg["scoring"].get("aggregation_temperature", 2.0))
    sims_dirs = [str(p) for p in cfg["data"]["sims_dirs"]]

    sim_dir = _find_sim_dir(args.design, sims_dirs)
    s2p_files = _state_s2p_files(sim_dir)
    K_avail = len(s2p_files)
    print(f"Design        : {sim_dir}")
    print(f"States avail. : {K_avail}   (need K={K})")
    if K_avail < K:
        raise SystemExit(f"Design has only {K_avail} states (< K={K}).")

    # Pre-load S21 for every available state once.
    s21_all = np.stack([_load_s21(str(f), freqs) for f in s2p_files], axis=0)  # (K_avail, n_freq)

    if args.auto_best:
        # Cost of assigning state s to spec slot j, for all (s, j); brute-force
        # the best distinct assignment by maximizing the aggregated design score.
        best = None
        for triple in permutations(range(K_avail), K):
            s21_K = s21_all[list(triple)]
            sc = design_score(freqs, s21_K, specs, agg, temp)
            if best is None or sc > best[0]:
                best = (sc, triple)
        chosen = list(best[1])
        print(f"Selection     : --auto-best  -> states {chosen} "
              f"(design_score={best[0]:+.4f})")
    else:
        chosen = [int(x) for x in args.states.split(",")]
        if len(chosen) != K:
            raise SystemExit(f"--states must list exactly K={K} indices, got {chosen}")
        if any(s < 0 or s >= K_avail for s in chosen):
            raise SystemExit(f"--states out of range [0,{K_avail-1}]: {chosen}")
        print(f"Selection     : states {chosen}")

    s21_K = s21_all[chosen]                                   # (K, n_freq)
    score_mat = design_score_matrix(freqs, s21_K, specs)      # (K, n_scalars)
    agg_score = design_score(freqs, s21_K, specs, agg, temp)

    print("\nPV score matrix (rows = spec slot / chosen state, cols = "
          f"{SCORE_SCALARS}):")
    for j, st in enumerate(chosen):
        vals = "  ".join(f"{score_mat[j, c]:+8.3f}" for c in range(score_mat.shape[1]))
        print(f"  slot {j} (state {st}):  {vals}")
    print(f"\nAggregated GA design score ({agg}, T={temp}): {agg_score:+.4f}")

    # Flatten row-major (state, scalar) -> the exact order test_generation expects.
    flat = score_mat.reshape(-1)
    scores_csv = ",".join(f"{v:.3f}" for v in flat)
    print("\n--scores CSV (raw dB, (state,scalar) row-major):")
    print(f"  {scores_csv}")

    py = "/disk/mcao53/conda_env/rfic2/bin/python"
    out = f"runs/test_generation/{sim_dir.parent.parent.name if (sim_dir/'..').exists() else 'run'}_score"
    print("\nInference command:")
    print(
        f"  {py} scripts/test_generation.py \\\n"
        f"      --config {args.config} \\\n"
        f"      --checkpoint {args.checkpoint} \\\n"
        f"      --scores={scores_csv} \\\n"
        f"      --no-simulate \\\n"
        f"      --output runs/test_generation/{args.design}_score"
    )
    print("\n(drop --no-simulate to also run the real EM sim and overlay S21.)")


if __name__ == "__main__":
    main()
