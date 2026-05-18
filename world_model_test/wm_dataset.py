"""
PyTorch Datasets for surrogate world model training.

Two dataset classes are provided:

  WMDataset          — loads directly from raw JSONL records + .s2p files.
                       Used by process_dataset.py to build processed datasets.

  ProcessedWMDataset — loads from a pre-processed directory (images.pt, s21.pt,
                       meta.json) produced by process_dataset.py. Use this for
                       training — it requires no s-parameter I/O or encoding at
                       runtime.

Each sample:
  image: float32 (3, rows, cols)
  s21:   float32 (n_freq,)  — S21 in dB at uniform frequency points
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import skrf
import torch
from torch.utils.data import Dataset

from encode import EncodeConfig, encode_sample


def _load_s21(s2p_path: str, freq_points: np.ndarray) -> np.ndarray:
    """Load S21 from a .s2p file and interpolate onto freq_points. Returns float32 (n_freq,)."""
    ntwk = skrf.Network(s2p_path)
    f_sim = ntwk.f
    s21_db = ntwk.s_db[:, 1, 0]
    return np.interp(freq_points, f_sim, s21_db).astype(np.float32)


class WMDataset(Dataset):
    """
    Raw world model dataset. Expands JSONL records into (design, state) samples,
    encoding images and loading S-parameters on the fly.

    Primarily used by process_dataset.py to build ProcessedWMDataset files.
    For training, prefer ProcessedWMDataset.

    Args:
        jsonl_paths: One or more paths to dataset.jsonl files produced by data_generation.py.
        encode_cfg:  EncodeConfig for image encoding.
        freq_points: Frequency points in Hz for S21 interpolation.
    """

    def __init__(
        self,
        jsonl_paths: str | Path | List[str | Path],
        encode_cfg: EncodeConfig,
        freq_points: np.ndarray,
    ) -> None:
        self.encode_cfg = encode_cfg
        self.freq_points = freq_points

        if isinstance(jsonl_paths, (str, Path)):
            jsonl_paths = [jsonl_paths]

        self.records: List[dict] = []
        for jsonl_path in jsonl_paths:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("success", False):
                        self.records.append(rec)

        # Build flat index: (record_idx, state_idx)
        self.index: List[Tuple[int, int]] = []
        for rec_idx, rec in enumerate(self.records):
            n_states = len(rec["switch_states"])
            n_files = len(rec.get("s_param_files", []))
            n = min(n_states, n_files)
            for s in range(n):
                self.index.append((rec_idx, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec_idx, state_idx = self.index[idx]
        rec = self.records[rec_idx]

        image_np = encode_sample(rec, state_idx, self.encode_cfg)
        s2p_path = rec["s_param_files"][state_idx]
        s21_np = _load_s21(s2p_path, self.freq_points)

        image = torch.from_numpy(image_np)   # (3, rows, cols)
        s21 = torch.from_numpy(s21_np)       # (n_freq,)
        return image, s21


class ProcessedWMDataset(Dataset):
    """
    Processed world model dataset. Loads pre-encoded bulk tensors from a directory
    produced by process_dataset.py.

    The directory must contain:
      images.pt  — float32 tensor (N, 3, H, W)
      s21.pt     — float32 tensor (N, n_freq)
      meta.json  — dataset metadata (source dirs, freq range, encoding params, ...)

    Args:
        processed_dir: Path to the processed dataset directory.
    """

    def __init__(self, processed_dir: str | Path) -> None:
        processed_dir = Path(processed_dir)
        with open(processed_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.images: torch.Tensor = torch.load(
            processed_dir / "images.pt", weights_only=True
        )
        self.s21: torch.Tensor = torch.load(
            processed_dir / "s21.pt", weights_only=True
        )

    @property
    def n_freq(self) -> int:
        return self.s21.shape[1]

    @property
    def freq_points(self) -> np.ndarray:
        return np.array(self.meta["freq_points_hz"], dtype=np.float64)

    def __len__(self) -> int:
        return self.s21.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.s21[idx]
