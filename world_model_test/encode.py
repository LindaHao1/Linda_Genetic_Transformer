"""
Design-to-tensor encoding for the world model.

Each design is encoded as a 3-channel image (rows, cols):
  Channel 0: pixel type + continuous value (static across states)
  Channel 1: ON/OFF spatial map for a specific switch state
  Channel 2: secondary continuous values — inductor_cap_value at inductor pixels,
             cap_pair_value at cap-pair endpoint pixels; 0.0 elsewhere (static)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Pixel type → value mapping boundaries (channel 0)
_GROUND_VAL = 0.05
_METAL_VAL = 0.20
_INDUCTOR_BASE = 0.30
_INDUCTOR_RANGE = 0.20   # [0.30, 0.50]
_SWITCH_BASE = 0.60
_SWITCH_RANGE = 0.30     # [0.60, 0.90]
_VIA_VAL = 0.95          # above switch range top

PIXEL_TYPE_GROUND = 0
PIXEL_TYPE_METAL = 1
PIXEL_TYPE_INDUCTOR = 2
PIXEL_TYPE_SWITCH = 3
PIXEL_TYPE_VIA = 4


@dataclass
class EncodeConfig:
    rows: int
    cols: int
    cap_min: float              # switch cap range (Farads)
    cap_max: float
    n_inductor_values: int      # length of inductor_translation_table
    inductor_cap_min: float     # inductor MIM cap range (Farads)
    inductor_cap_max: float
    cap_pair_min: float         # cap-pair shunt cap range (Farads)
    cap_pair_max: float


def build_encode_config(config: dict[str, Any]) -> EncodeConfig:
    """Build EncodeConfig from a merged YAML config dict."""
    grid = config.get("grid", {})
    rows = grid["rows"]
    cols = grid["cols"]
    cap_min = grid.get("cap_min", 100e-15)
    cap_max = grid.get("cap_max", 1000e-15)

    # n_inductor_values: from grid override or count of inductor_translation_table
    n_inductor_values = grid.get("n_inductor_values", None)
    if n_inductor_values is None:
        table = config.get("inductor_translation_table", [])
        n_inductor_values = max(len(table), 1)

    return EncodeConfig(
        rows=rows,
        cols=cols,
        cap_min=cap_min,
        cap_max=cap_max,
        n_inductor_values=n_inductor_values,
        inductor_cap_min=grid.get("inductor_cap_min", 200e-15),
        inductor_cap_max=grid.get("inductor_cap_max", 1000e-15),
        cap_pair_min=grid.get("cap_pair_min", 50e-15),
        cap_pair_max=grid.get("cap_pair_max", 1000e-15),
    )


def make_pixel_channel(record: dict[str, Any], cfg: EncodeConfig) -> np.ndarray:
    """
    Build channel 0: pixel type + continuous value.
    Returns float32 array of shape (rows, cols).
    Static across all states of the same design.
    """
    matrix = record["matrix"]          # List[List[int]], shape rows×cols
    cap_values = record["cap_values"]  # List[float], one per switch pixel (row-major order)
    inductor_values = record.get("inductor_values", [])  # List[int], one per inductor pixel

    ch = np.zeros((cfg.rows, cfg.cols), dtype=np.float32)

    sw_idx = 0
    ind_idx = 0
    cap_range = cfg.cap_max - cfg.cap_min
    n_ind = max(cfg.n_inductor_values - 1, 1)

    for r in range(cfg.rows):
        for c in range(cfg.cols):
            ptype = matrix[r][c]
            if ptype == PIXEL_TYPE_GROUND:
                ch[r, c] = _GROUND_VAL
            elif ptype == PIXEL_TYPE_METAL:
                ch[r, c] = _METAL_VAL
            elif ptype == PIXEL_TYPE_INDUCTOR:
                ind_val = inductor_values[ind_idx] if ind_idx < len(inductor_values) else 0
                ch[r, c] = _INDUCTOR_BASE + (ind_val / n_ind) * _INDUCTOR_RANGE
                ind_idx += 1
            elif ptype == PIXEL_TYPE_SWITCH:
                cap = cap_values[sw_idx] if sw_idx < len(cap_values) else cfg.cap_min
                t = np.clip((cap - cfg.cap_min) / cap_range, 0.0, 1.0)
                ch[r, c] = _SWITCH_BASE + t * _SWITCH_RANGE
                sw_idx += 1
            elif ptype == PIXEL_TYPE_VIA:
                ch[r, c] = _VIA_VAL

    return ch


def make_state_channel(record: dict[str, Any], state_idx: int, cfg: EncodeConfig) -> np.ndarray:
    """
    Build channel 1: ON/OFF spatial map for state_idx.
    Returns float32 array of shape (rows, cols).
    A pixel is 1.0 if it is a switch/inductor that is ON in this state, else 0.0.
    """
    matrix = record["matrix"]
    switch_states = record["switch_states"]          # List[List[int]], (n_states, n_switches)
    inductor_switch_states = record.get("inductor_switch_states", [])  # (n_states, n_inductors)

    state_sw = switch_states[state_idx]              # List[int], length n_switches
    state_ind = inductor_switch_states[state_idx] if inductor_switch_states else []

    ch = np.zeros((cfg.rows, cfg.cols), dtype=np.float32)

    sw_idx = 0
    ind_idx = 0
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            ptype = matrix[r][c]
            if ptype == PIXEL_TYPE_SWITCH:
                if sw_idx < len(state_sw) and state_sw[sw_idx] == 1:
                    ch[r, c] = 1.0
                sw_idx += 1
            elif ptype == PIXEL_TYPE_INDUCTOR:
                if ind_idx < len(state_ind) and state_ind[ind_idx] == 1:
                    ch[r, c] = 1.0
                ind_idx += 1

    return ch


def make_value_channel(record: dict[str, Any], cfg: EncodeConfig) -> np.ndarray:
    """
    Build channel 2: secondary continuous values.
    - Inductor pixels: inductor_cap_value normalized to [0, 1]
    - Cap-pair endpoint pixels: cap_pair_value normalized to [0, 1]
    - All other pixels: 0.0
    Returns float32 array of shape (rows, cols). Static across all states.
    """
    ch = np.zeros((cfg.rows, cfg.cols), dtype=np.float32)
    matrix = record["matrix"]

    inductor_cap_values = record.get("inductor_cap_values", [])
    ind_cap_range = cfg.inductor_cap_max - cfg.inductor_cap_min
    ind_idx = 0
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            if matrix[r][c] == PIXEL_TYPE_INDUCTOR:
                cap = inductor_cap_values[ind_idx] if ind_idx < len(inductor_cap_values) else cfg.inductor_cap_min
                ch[r, c] = float(np.clip((cap - cfg.inductor_cap_min) / ind_cap_range, 0.0, 1.0))
                ind_idx += 1

    cap_pixel_pairs = record.get("cap_pixel_pairs", [])
    cap_pair_values = record.get("cap_pair_values", [])
    cap_pair_range = cfg.cap_pair_max - cfg.cap_pair_min
    for pair_idx, pair in enumerate(cap_pixel_pairs):
        val = cap_pair_values[pair_idx] if pair_idx < len(cap_pair_values) else cfg.cap_pair_min
        t = float(np.clip((val - cfg.cap_pair_min) / cap_pair_range, 0.0, 1.0))
        for (r, c) in pair:
            ch[r, c] = t

    return ch


def encode_sample(record: dict[str, Any], state_idx: int, cfg: EncodeConfig) -> np.ndarray:
    """
    Encode one (design, state) pair into a 3-channel image.
    Returns float32 array of shape (3, rows, cols).
    """
    ch0 = make_pixel_channel(record, cfg)
    ch1 = make_state_channel(record, state_idx, cfg)
    ch2 = make_value_channel(record, cfg)
    return np.stack([ch0, ch1, ch2], axis=0)
