"""
Evaluation: best-of-N true score on a fixed set of target specs, plus pool
validity and diversity.  Importable (`run_eval`) and runnable as a CLI.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factory import (build_model, build_score_norm, build_specs,  # noqa: E402
                         build_surrogate, build_tokenizer, freq_points,
                         load_config)
from src.scoring import PerStateBest, design_score  # noqa: E402
from src.training import decode_to_design, sample_score_targets  # noqa: E402


def _pairwise_grid_diversity(designs) -> float:
    grids = [np.asarray(d.pixel_grid).ravel() for d in designs if d]
    if len(grids) < 2:
        return 0.0
    tot = cnt = 0.0
    for i in range(len(grids)):
        for j in range(i + 1, len(grids)):
            tot += float(np.mean(grids[i] != grids[j]))
            cnt += 1
    return tot / cnt


def run_eval(model, surrogate, tok, targets: np.ndarray, specs,
             freqs: np.ndarray, n_per_target: int = 8, cfg_scale: float = 4.0,
             temperature: float = 1.0, agg: str = "soft_max",
             temp: float = 2.0, constraint_config=None) -> Dict[str, float]:
    """targets: (n_targets, K, score_dim) normalized conditioning vectors.

    Quality is the GA score (− soft_max cost) of each generated design,
    independent of the conditioning target. Returns best-of-N aggregates.
    """
    device = model.device
    best_scores: List[float] = []
    all_designs = []
    n_valid = n_total = 0

    for t in range(targets.shape[0]):
        spec = torch.tensor(
            np.repeat(targets[t][None], n_per_target, axis=0)).float().to(device)
        seqs = model.generate_batch(
            spec, max_len=model.mcfg.max_seq_len,
            temperature=temperature, cfg_scale=cfg_scale,
            apply_constraints=True, constraint_config=constraint_config)
        designs = [decode_to_design(tok, s) for s in seqs]
        n_total += len(designs)
        valid = [d for d in designs if d is not None]
        n_valid += len(valid)
        all_designs += valid
        scored = []
        for d in valid:
            try:
                mean, _ = surrogate.predict_multistate(d)
            except Exception:
                continue
            scored.append(design_score(freqs, mean, specs, agg, temp))
        if scored:
            best_scores.append(max(scored))

    return {
        "best_of_n_mean_score": float(np.mean(best_scores)) if best_scores else float("nan"),
        "valid_frac": n_valid / max(1, n_total),
        "pool_diversity": _pairwise_grid_diversity(all_designs),
        "n_targets_with_valid": float(len(best_scores)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-targets", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = build_tokenizer(cfg)
    model = build_model(cfg, tok)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    surrogate = build_surrogate(cfg)
    K = cfg["grid"]["n_states"]
    specs = build_specs(cfg)
    fpts = freq_points(cfg)
    score_norm = build_score_norm(cfg)
    sc = cfg["spec"]

    rng = np.random.default_rng(123)
    targets = sample_score_targets(
        PerStateBest(K), K, args.n_targets, rng,
        sc["target_margins"], sc["bounds"], sc["cold_start_target"],
        sc["stretch_frac"], score_norm)
    metrics = run_eval(model, surrogate, tok, targets, specs, fpts,
                       n_per_target=cfg["iterated"].get("n_per_target", 8),
                       cfg_scale=cfg["iterated"]["cfg_scale"],
                       agg=cfg["scoring"]["aggregation_mode"],
                       temp=cfg["scoring"]["aggregation_temperature"])
    print(metrics)


if __name__ == "__main__":
    main()
