"""
End-to-end LM generation test.

Given user-supplied target PV scores (K*n_scalars dB floats), this script:

  1. Builds tokenizer/model/surrogate from `--config`.
  2. Loads the LM checkpoint and generates one design conditioned on the
     normalized target spec.
  3. Writes the design to (a) `design.json` (record dict, world-model format)
     and (b) `design.yaml` (simulation-engine override; `_base_:` inherits
     from a base virtuoso config).
  4. Runs the world model on the design (K-state predicted S21).
  5. Optionally runs the real EM simulator and reads back the per-state S21.
  6. Saves `test_generation.png` (grid + predicted vs simulated S21 + per-
     state requested/predicted/simulated PV scores) and `scores.json`.

Usage:
  python scripts/test_generation.py \
      --config configs/default.yaml \
      --checkpoint runs/sft.pt \
      --scores "-5,20,2,-7,18,3,-6,22,2" \
      --output runs/test_generation/run_001 \
      [--no-simulate]

Note: `runs/sft.pt` was trained on the old vocab; load may fail with a vocab
mismatch.  Re-run `scripts/train_sft.py` to mint a checkpoint compatible with
the new tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_yaml import (build_pixelated_override,  # noqa: E402
                             resolve_engine_config_from_data_collection,
                             write_simulation_yaml)
from src.factory import (build_model, build_score_norm,  # noqa: E402
                         build_specs, build_surrogate,
                         build_tokenizer, freq_points,
                         load_config)
from src.scoring import SCORE_SCALARS, design_score_matrix  # noqa: E402
from src.training import decode_to_design  # noqa: E402
from src.visualize import plot_test_generation  # noqa: E402

_WM_DIR = "/disk/mcao53/RFIC_RL/world_model_test"
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)

# Default: the same data-collection config used to produce the world-model
# training data.  We follow its `_base_:` chain + `virtuoso_config:` field to
# the matching engine config -- this guarantees the LM-generated design YAML
# inherits exactly the same engine parameters as the training sims (pixel
# size, inductor table, EMX/Spectre settings, etc.).
DEFAULT_DATA_COLLECTION_CONFIG = (
    "/disk/mcao53/RFIC_RL/world_model_test/wm_data_collection.yaml"
)


def _parse_scores(scores_str: str, K: int, n_sc: int) -> np.ndarray:
    parts = [p.strip() for p in scores_str.split(",") if p.strip()]
    if len(parts) != K * n_sc:
        raise SystemExit(
            f"--scores expected {K * n_sc} comma-separated values "
            f"(K={K} states * {n_sc} scalars), got {len(parts)}")
    arr = np.array([float(p) for p in parts], dtype=np.float32)
    return arr.reshape(K, n_sc)


def _load_s21_from_files(s2p_files, freq_pts: np.ndarray) -> np.ndarray:
    """Each path -> (n_freq,) dB.  Stack -> (K, n_freq)."""
    from wm_dataset import _load_s21        # noqa: E402

    return np.stack([_load_s21(str(p), freq_pts) for p in s2p_files], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="runs/sft.pt")
    ap.add_argument("--scores", required=True,
                    help="comma-separated K*n_scalars raw dB values "
                         "(e.g. K=3 -> '-5,20,2,-7,18,3,-6,22,2')")
    ap.add_argument("--output", default="runs/test_generation/run")
    ap.add_argument("--data-collection-config",
                    default=DEFAULT_DATA_COLLECTION_CONFIG,
                    help="GA-style data-collection YAML whose `virtuoso_config:` "
                         "is followed to find the engine config the LM "
                         "design YAML should inherit from.")
    ap.add_argument("--base-sim-config", default=None,
                    help="Optional: explicit engine config path; overrides "
                         "the value resolved from --data-collection-config.")
    ap.add_argument("--no-simulate", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the engine config the LM design YAML should inherit from.
    if args.base_sim_config:
        engine_cfg_path = args.base_sim_config
        print(f"[test_gen] engine config (explicit): {engine_cfg_path}")
    else:
        engine_cfg_path = resolve_engine_config_from_data_collection(
            args.data_collection_config)
        print(f"[test_gen] engine config (from {args.data_collection_config}):"
              f"\n           -> {engine_cfg_path}")

    K = int(cfg["grid"]["n_states"])
    scalars = cfg["spec"].get("score_scalars", list(SCORE_SCALARS))
    n_sc = len(scalars)

    scores_raw = _parse_scores(args.scores, K, n_sc)
    print(f"[test_gen] requested scores (K={K}, scalars={scalars}):")
    for k in range(K):
        print(f"  state {k}: " + ", ".join(
            f"{n}={scores_raw[k, j]:+.2f}" for j, n in enumerate(scalars)))

    # ---- build model / tokenizer / surrogate ----------------------------
    device = torch.device(cfg["surrogate"].get("device", "cpu"))
    tok = build_tokenizer(cfg)
    model = build_model(cfg, tok).to(device)
    surrogate = build_surrogate(cfg)
    score_norm = build_score_norm(cfg)
    specs = build_specs(cfg)
    fpts = freq_points(cfg)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_file():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(blob["model"])
            print(f"[test_gen] loaded checkpoint {ckpt_path}")
        except RuntimeError as exc:
            print(f"[test_gen] WARNING: checkpoint load failed "
                  f"(probably trained on the old vocab) — proceeding with "
                  f"random init. Re-run scripts/train_sft.py to mint a "
                  f"compatible checkpoint.\n  reason: {exc}")
    else:
        print(f"[test_gen] WARNING: no checkpoint at {ckpt_path} — using "
              f"random init (smoke test only).")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- generate -------------------------------------------------------
    spec_norm = score_norm.normalize(scores_raw).astype(np.float32)
    spec_tensor = torch.from_numpy(spec_norm[None]).to(device)        # (1,K,n_sc)
    temperature = (args.temperature
                   if args.temperature is not None
                   else float(cfg["iterated"]["temperature"]))
    cfg_scale = (args.cfg_scale
                 if args.cfg_scale is not None
                 else float(cfg["iterated"]["cfg_scale"]))
    seqs = model.generate_batch(
        spec_tensor, max_len=int(cfg["model"]["max_seq_len"]),
        temperature=temperature, cfg_scale=cfg_scale,
        apply_constraints=True)
    design = decode_to_design(tok, seqs[0])
    if design is None:
        raise SystemExit("[test_gen] decoded design is None (ungrammatical) — "
                         "abort.")
    print(f"[test_gen] generated design: rows={design.rows}, cols={design.cols}"
          f", n_sw={design.n_switches}, n_ind={design.n_inductors}, "
          f"n_pairs={len(design.cap_pixel_pairs)}, K={design.n_states}")

    # ---- write design artifacts ----------------------------------------
    design_json = out_dir / "design.json"
    with design_json.open("w") as f:
        json.dump(design.to_record(), f, indent=2)

    yaml_path = out_dir / "design.yaml"
    sim_output_dir = out_dir / "sim_workspace"
    write_simulation_yaml(
        design, engine_cfg_path, str(yaml_path),
        output_parent_dir=str(sim_output_dir),
    )
    print(f"[test_gen] wrote {design_json}, {yaml_path}")

    # ---- world-model prediction ----------------------------------------
    s21_pred, s21_var = surrogate.predict_multistate(design)            # (K, n_freq)
    scores_pred = design_score_matrix(fpts, s21_pred, specs, scalars)

    # ---- real simulator -----------------------------------------------
    s21_sim = None
    scores_sim = None
    if not args.no_simulate:
        try:
            from virtuoso_engine.parallel_engine import \
                ParallelVirtuosoEngine        # noqa: E402
            override = build_pixelated_override(
                design, output_parent_dir=str(sim_output_dir))
            sim_output_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "engine.log"
            with ParallelVirtuosoEngine(
                config_path=engine_cfg_path,
                num_workers=1,
                log_path=str(log_path),
            ) as eng:
                result = eng.run_batch([override])[0]
            if result.success and result.s_param_files:
                s21_sim = _load_s21_from_files(result.s_param_files, fpts)
                scores_sim = design_score_matrix(fpts, s21_sim, specs, scalars)
                print(f"[test_gen] simulator: success, K={s21_sim.shape[0]} states")
            else:
                print(f"[test_gen] simulator: failed or returned no S2P files "
                      f"(success={result.success})")
        except Exception as exc:
            print(f"[test_gen] simulator path errored: {exc}")

    # ---- scores JSON ---------------------------------------------------
    scores_blob = {
        "scalars": list(scalars),
        "requested": scores_raw.tolist(),
        "predicted": scores_pred.tolist(),
    }
    if scores_sim is not None:
        scores_blob["simulated"] = scores_sim.tolist()
    with (out_dir / "scores.json").open("w") as f:
        json.dump(scores_blob, f, indent=2)

    # ---- plot ----------------------------------------------------------
    plot_path = out_dir / "test_generation.png"
    plot_test_generation(
        design=design,
        s21_pred=s21_pred,
        s21_sim=s21_sim,
        specs=specs,
        freqs=fpts,
        scores_requested=scores_raw,
        scores_pred=scores_pred,
        scores_sim=scores_sim,
        save_path=str(plot_path),
        scalars=scalars,
    )
    print(f"[test_gen] wrote {plot_path}")

    # readable summary
    print("\n[test_gen] PV-score summary (per state):")
    header = "  " + " | ".join(f"{n}".rjust(13) for n in ["state"] + list(scalars))
    print(header)
    for k in range(K):
        line = f"  state {k}  req: " + "  ".join(
            f"{scores_raw[k, j]:+.2f}" for j in range(n_sc))
        print(line)
        line = f"  state {k}  pred:" + "  ".join(
            f"{scores_pred[k, j]:+.2f}" for j in range(n_sc))
        print(line)
        if scores_sim is not None:
            line = f"  state {k}  sim: " + "  ".join(
                f"{scores_sim[k, j]:+.2f}" for j in range(n_sc))
            print(line)


if __name__ == "__main__":
    main()
