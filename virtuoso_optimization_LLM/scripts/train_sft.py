"""
Phase 1: initial SFT.

Standard next-token CE with:
  - hindsight relabeling (spec = each example's OWN achieved S21)
  - token-type loss weighting (pixel 1, value 3, state 3)
  - 15% spec-dropout for CFG (handled inside the model in training mode)
  - per-token-type loss logging + checkpointing

Usage:
  python scripts/train_sft.py --config configs/default.yaml \
      --steps 200 --out ./runs/sft.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (LMDataset, build_from_sims,  # noqa: E402
                         build_from_sims_cached, make_random_design)
from src.factory import (build_model, build_score_norm, build_specs,  # noqa: E402
                         build_surrogate, build_tokenizer, freq_points,
                         load_config)
from src.training import collate, sft_step, token_type_weights  # noqa: E402
from src.types import SParams, TrainingExample  # noqa: E402


def build_dataset(cfg, tok, surrogate, fpts) -> LMDataset:
    d = cfg["data"]
    if d.get("sims_dirs"):
        cache_path = d.get("cache_path")
        if cache_path:
            return build_from_sims_cached(
                d["sims_dirs"], fpts, cache_path=cache_path,
                limit=d.get("limit"),
                n_states=cfg["grid"]["n_states"],
                seed=d.get("state_select_seed", 0),
                refresh=bool(d.get("cache_refresh", False)))
        return build_from_sims(
            d["sims_dirs"], fpts, limit=d.get("limit"),
            n_states=cfg["grid"]["n_states"],          # subsample 10 -> K states
            seed=d.get("state_select_seed", 0))
    if not d.get("use_synthetic", True):
        raise SystemExit("No data: set data.sims_dirs or data.use_synthetic")
    rng = np.random.default_rng(0)
    exs = []
    for _ in range(d.get("synthetic_n", 256)):
        des = make_random_design(tok.config, rng)
        mean, _ = surrogate.predict_multistate(des)         # hindsight label
        exs.append(TrainingExample(
            des, SParams(fpts, mean), source="surrogate_labeled"))
    return LMDataset(exs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="./runs/sft.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["surrogate"].get("device", "cpu"))
    tok = build_tokenizer(cfg)
    model = build_model(cfg, tok).to(device)
    surrogate = build_surrogate(cfg)
    fpts = freq_points(cfg)
    K = cfg["grid"]["n_states"]
    specs = build_specs(cfg)
    score_norm = build_score_norm(cfg)

    ds = build_dataset(cfg, tok, surrogate, fpts)
    print(f"SFT dataset: {len(ds)} examples")

    type_w = token_type_weights(tok, cfg["sft"].get("type_weights"))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["sft"]["lr"]),
                            weight_decay=1e-4)
    steps = args.steps or cfg["sft"]["steps"]
    bs = cfg["sft"]["batch_size"]
    rng = np.random.default_rng(1)

    for step in range(steps):
        idx = rng.integers(0, len(ds), size=min(bs, len(ds)))
        batch = collate([ds[int(i)] for i in idx], tok, specs, fpts,
                        score_norm, K, device)
        stats = sft_step(model, batch, type_w, opt)
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            extra = " ".join(f"{k}={v:.3f}" for k, v in stats.items()
                             if k.startswith("loss_"))
            print(f"step {step:5d} | loss {stats['loss']:.4f} | {extra}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(),
                "tokenizer_config": tok.config.__dict__}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
