#!/usr/bin/env python3
"""
process_dataset.py — Convert one or more raw world model datasets into a single
processed dataset ready for training.

A raw dataset is a directory produced by data_generation.py, containing:
  dataset.jsonl   (one design record per line, with paths to per-state .s2p files)
  sims/           (the simulation output tree with .s2p files)

This script:
  1. Reads successful JSONL records from all input directories.
  2. Encodes every (design, state) pair into a 3-channel image tensor.
  3. Loads S21 from the corresponding .s2p file and interpolates onto a uniform
     frequency grid.
  4. Saves the full dataset as two bulk tensors + a metadata JSON file:
       images.pt  — float32  (N, 3, H, W)
       s21.pt     — float32  (N, n_freq)
       meta.json  — source dirs, encoding params, frequency grid, sample count

The output directory can be passed directly to ProcessedWMDataset and to train.py.

Usage
-----
    python world_model_test/process_dataset.py \\
        --input-dirs /path/to/raw1 /path/to/raw2 \\
        --output     /path/to/processed \\
        --config     world_model_test/wm_data_collection.yaml \\
        --n-freq 71 --freq-min 0 --freq-max 70e9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model_test.encode import build_encode_config, encode_sample  # noqa: E402
from world_model_test.wm_dataset import _load_s21                       # noqa: E402


# ---------------------------------------------------------------------------
# Config loader (mirrors the one in train.py)
# ---------------------------------------------------------------------------

def _load_yaml_with_base(path: str | Path) -> dict:
    def _deep_merge(base: dict, override: dict) -> dict:
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = _deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    base_rel = data.pop("_base_", None)
    if base_rel:
        base_data = _load_yaml_with_base(path.parent / base_rel)
        data = _deep_merge(base_data, data)
    return data


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw WM datasets (JSONL + s2p) to a processed bulk tensor dataset."
    )
    p.add_argument(
        "--input-dirs", nargs="+", required=True, metavar="DIR",
        help="Raw dataset directories (each must contain dataset.jsonl).",
    )
    p.add_argument(
        "--output", required=True, metavar="DIR",
        help="Output directory for the processed dataset.",
    )
    p.add_argument(
        "--config",
        default=str(Path(__file__).parent / "wm_data_collection.yaml"),
        help="YAML config file used to build the encoding config (supports _base_ inheritance).",
    )
    p.add_argument("--n-freq",   type=int,   default=71,   help="Number of frequency points.")
    p.add_argument("--freq-min", type=float, default=0.0,  help="Frequency range start (Hz).")
    p.add_argument("--freq-max", type=float, default=70e9, help="Frequency range end (Hz).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def _load_records(input_dirs: list[str]) -> list[dict]:
    """Read successful JSONL records from every input directory."""
    records: list[dict] = []
    for raw_dir in input_dirs:
        jsonl = Path(raw_dir) / "dataset.jsonl"
        if not jsonl.exists():
            print(f"  Warning: {jsonl} not found — skipping.")
            continue
        n_before = len(records)
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("success", False):
                    records.append(rec)
        print(f"  {jsonl}: {len(records) - n_before} successful records")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    config = _load_yaml_with_base(args.config)
    encode_cfg = build_encode_config(config)
    freq_points = np.linspace(args.freq_min, args.freq_max, args.n_freq)

    print("=" * 60)
    print("World Model Dataset Processor")
    print("=" * 60)
    print(f"Input dirs  : {args.input_dirs}")
    print(f"Output dir  : {args.output}")
    print(f"Config      : {args.config}")
    print(f"Grid        : {encode_cfg.rows}×{encode_cfg.cols}")
    print(f"Frequency   : {args.n_freq} pts, "
          f"{args.freq_min/1e9:.0f}–{args.freq_max/1e9:.0f} GHz")
    print()

    # --- Load records from all input directories ---
    print("Loading records...")
    records = _load_records(args.input_dirs)
    if not records:
        print("Error: no successful records found in any input directory.")
        sys.exit(1)
    print(f"Total records: {len(records)}\n")

    # --- Build flat (record_idx, state_idx) index ---
    index: list[tuple[int, int]] = []
    for rec_idx, rec in enumerate(records):
        n_states = len(rec["switch_states"])
        n_files  = len(rec.get("s_param_files", []))
        for s in range(min(n_states, n_files)):
            index.append((rec_idx, s))
    N = len(index)
    H, W = encode_cfg.rows, encode_cfg.cols
    print(f"Total (design, state) samples: {N}")

    # --- Encode ---
    print("Encoding samples (image + S21)...")
    t0 = time.time()

    images_list: list[torch.Tensor] = []
    s21_list:    list[torch.Tensor] = []
    n_errors = 0

    for i, (rec_idx, state_idx) in enumerate(index):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (N - i) / rate
            print(f"  {i}/{N}  ({rate:.0f} samples/s, ETA {eta:.0f}s)")

        rec = records[rec_idx]
        try:
            img_np  = encode_sample(rec, state_idx, encode_cfg)
            s2p_path = rec["s_param_files"][state_idx]
            s21_np  = _load_s21(s2p_path, freq_points)
            images_list.append(torch.from_numpy(img_np))
            s21_list.append(torch.from_numpy(s21_np))
        except Exception as exc:
            print(f"  Warning: sample {i} (rec={rec_idx}, state={state_idx}) failed: {exc}")
            n_errors += 1

    n_saved = len(images_list)
    elapsed = time.time() - t0
    print(f"Encoding done in {elapsed:.1f}s — {n_saved} ok, {n_errors} errors\n")

    if n_saved == 0:
        print("Error: all samples failed encoding.")
        sys.exit(1)

    images = torch.stack(images_list)  # (N, 3, H, W)
    s21    = torch.stack(s21_list)     # (N, n_freq)

    # --- Save ---
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Saving tensors to {out} ...")
    torch.save(images, out / "images.pt")
    torch.save(s21,    out / "s21.pt")

    meta = {
        "n_samples":         n_saved,
        "n_errors":          n_errors,
        "image_shape":       list(images.shape),
        "s21_shape":         list(s21.shape),
        "rows":              encode_cfg.rows,
        "cols":              encode_cfg.cols,
        "n_freq":            args.n_freq,
        "freq_min":          args.freq_min,
        "freq_max":          args.freq_max,
        "freq_points_hz":    freq_points.tolist(),
        "cap_min":           encode_cfg.cap_min,
        "cap_max":           encode_cfg.cap_max,
        "n_inductor_values": encode_cfg.n_inductor_values,
        "inductor_cap_min":  encode_cfg.inductor_cap_min,
        "inductor_cap_max":  encode_cfg.inductor_cap_max,
        "cap_pair_min":      encode_cfg.cap_pair_min,
        "cap_pair_max":      encode_cfg.cap_pair_max,
        "source_dirs":       [str(d) for d in args.input_dirs],
        "n_source_records":  len(records),
        "created_at":        datetime.now().isoformat(),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Processed dataset saved to {out}")
    print(f"  images.pt : {tuple(images.shape)}  ({images.nbytes / 1e6:.1f} MB)")
    print(f"  s21.pt    : {tuple(s21.shape)}   ({s21.nbytes / 1e6:.1f} MB)")
    print(f"  meta.json : written")


if __name__ == "__main__":
    main()
