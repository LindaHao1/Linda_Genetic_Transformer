#!/usr/bin/env python3
"""
pixel_gui_designer/designer.py
Interactive GUI for editing pixelated RF filter designs (config.yaml).

Pixel types
-----------
  0  GND  — ground layer only (dark grey)
  1  MTL  — top + ground metal (steel blue)
  2  IND  — inductor port (amber)
  3  SW   — switch port (red)
  4  VIA  — via / gnd-short (purple)
  5  SCAP — shunt-capacitor port (permanent MIM cap to ground) (teal-green)

Usage
-----
  python pixel_gui_designer/designer.py
  python pixel_gui_designer/designer.py path/to/config.yaml
"""

from __future__ import annotations

import copy
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from PyQt5.QtCore import Qt, QPoint, QProcess, QRect, QSize, pyqtSignal
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPixmap,
)
from PyQt5.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QAction, QApplication,
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QToolBar, QVBoxLayout, QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PIXEL_INFO: Dict[int, Tuple[str, QColor]] = {
    0: ("GND", QColor("#4a4a4a")),
    1: ("MTL", QColor("#5ba8d4")),
    2: ("IND", QColor("#c8861a")),
    3: ("SW",  QColor("#c0392b")),
    4: ("VIA", QColor("#7d3c98")),
    5: ("SCAP", QColor("#17a589")),
}

# Metal-layer code → (label, colour) for differential multi-layer designs.
# Used as the BASE cell colour for non-empty pixels so M10 / M9 / double are
# distinguishable; advanced types (IND/SW/VIA/SCAP) keep their text labels on top.
# Light, high-contrast hues matching virtuoso_engine/plotter.py.
LAYER_INFO: Dict[int, Tuple[str, QColor]] = {
    0: ("M10",    QColor("#74c0e8")),   # light blue   — primary top metal (AP)
    1: ("M9",     QColor("#6fcf97")),   # light green  — second metal
    2: ("M9+M10", QColor("#c79be0")),   # light orchid — double metal
}

INDUCTOR_LABELS = ["300 pH (EM)"]
DEFAULT_CAP_F   = 1.0e-12   # 1 pF
DEFAULT_PAIR_F  = 1.0e-12
DEFAULT_SCAP_F  = 122.0e-15  # 122 fF — default shunt-cap value

MODE_SELECT = "select"
MODE_BRUSH  = "brush"

PAIR_COLORS = [
    QColor("#ffffff"), QColor("#ffe066"), QColor("#ff9ff3"),
    QColor("#54a0ff"), QColor("#00d2d3"), QColor("#ff6b6b"),
]

DEFAULT_INDUCTOR_TABLE = [
    "/disk/mcao53/RFIC_RL/Inductor_spar/Ind_from_layout/Active_inductor_300pH_EM_on.s1p",
]

DEFAULT_INDUCTOR_OFF_TABLE = [
    "/disk/mcao53/RFIC_RL/Inductor_spar/Ind_from_layout/Active_inductor_300pH_EM_off.s1p",
]

# Persistent GUI settings file (conda env, script paths)
SETTINGS_PATH = Path(__file__).parent / "gui_settings.json"


# ─────────────────────────────────────────────────────────────────────────────
# Nested-dict helpers  (dot-separated key paths, e.g. "project.virtuoso_dir")
# ─────────────────────────────────────────────────────────────────────────────

def _get_nested(d: dict, key_path: str, default: Any = "") -> Any:
    for k in key_path.split("."):
        if not isinstance(d, dict):
            return default
        d = d.get(k, None)
        if d is None:
            return default
    return d if d is not None else default


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# ─────────────────────────────────────────────────────────────────────────────
# Persistent GUI Settings
# ─────────────────────────────────────────────────────────────────────────────

class GuiSettings:
    """
    Lightweight JSON-backed settings for execution environment.
    Stored in pixel_gui_designer/gui_settings.json and auto-loaded on startup.
    """

    DEFAULTS: Dict[str, Any] = {
        "use_conda":          True,
        "conda_env":          "",
        "conda_executable":   "conda",
        "simulate_script":    "",   # blank → auto-detect
        "python_executable":  "",   # blank → sys.executable (when not using conda)
    }

    def __init__(self) -> None:
        self.data: Dict[str, Any] = dict(self.DEFAULTS)
        self.load()

    def load(self) -> None:
        if SETTINGS_PATH.exists():
            try:
                with open(SETTINGS_PATH) as f:
                    self.data.update(json.load(f))
            except Exception:
                pass

    def save(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w") as f:
            json.dump(self.data, f, indent=2)

    def __getitem__(self, key: str) -> Any:
        return self.data.get(key, self.DEFAULTS.get(key, ""))

    def __setitem__(self, key: str, val: Any) -> None:
        self.data[key] = val

    def get_simulate_script(self) -> str:
        if self.data.get("simulate_script"):
            return self.data["simulate_script"]
        return str(Path(__file__).parent.parent / "simulate_design.py")

    def build_command(self, yaml_path: str, output_dir: str) -> List[str]:
        """Build the full shell command list to run the simulation."""
        script   = self.get_simulate_script()
        py_args  = [script, yaml_path, "--output_dir", str(output_dir)]
        env_name = self.data.get("conda_env", "").strip()
        use_conda = self.data.get("use_conda", True)
        conda_exe = self.data.get("conda_executable", "conda") or "conda"
        pyexe     = self.data.get("python_executable", "").strip()

        if use_conda and env_name:
            return [conda_exe, "run", "-n", env_name,
                    "--no-capture-output", "python"] + py_args
        elif pyexe:
            return [pyexe] + py_args
        else:
            return [sys.executable] + py_args


# ─────────────────────────────────────────────────────────────────────────────
# Design Model
# ─────────────────────────────────────────────────────────────────────────────

class DesignModel:
    """
    Holds the mutable design state and keeps it consistent.

    Invariants maintained:
      len(cap_values)      == number of type-3 (SW) pixels (row-first order)
      cap_values[i]        == List[float] of capacitances in the i-th switch bank
      len(inductor_values) == number of type-2 (IND) pixels (row-first order)
      each state in state_list has length == number of SW pixels
      state_list[s][i]     == List[int] of on/off bits matching cap_values[i]
      if inductor_switch_states is not None, each inner list has length == number of IND pixels
      len(cap_pair_values) == len(cap_pixel_pairs)
    """

    def __init__(self) -> None:
        self.full_config:           Dict[str, Any]              = {}
        self.matrix:                List[List[int]]             = [[0]]
        self.state_list:            List[List[List[int]]]       = []
        self.cap_values:            List[List[float]]           = []
        self.inductor_values:       List[int]                   = []
        self.inductor_switch_states: Optional[List[List[int]]] = None
        self.cap_pixel_pairs:       List[List[List[int]]]       = []
        self.cap_pair_values:       List[float]                 = []
        self.shunt_cap_values:      List[float]                 = []   # one per type-5 (SCAP) pixel, row-first
        self.pixel_layers:          List[List[int]]             = [[0]] # parallel to matrix; 0=M10/AP 1=M9 2=double
        self.file_path:             Optional[str]               = None

    def is_differential(self) -> bool:
        """True when the loaded config is a differential-filter (multi-layer) design."""
        return self.full_config.get("mode") == "differential_filter"

    def layer_at(self, r: int, c: int) -> int:
        """Metal layer of pixel (r, c): 0=M10/AP, 1=M9, 2=double."""
        if r < len(self.pixel_layers) and c < len(self.pixel_layers[r]):
            return int(self.pixel_layers[r][c])
        return 0

    def set_layer(self, r: int, c: int, layer: int) -> None:
        """Set the metal layer of pixel (r, c).  No-op for empty (type-0) cells."""
        if int(self.matrix[r][c]) == 0:
            return
        self.pixel_layers[r][c] = int(layer)

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load(self, path: str) -> None:
        with open(path) as f:
            self.full_config = yaml.safe_load(f) or {}
        self.file_path = path
        self._pull_from_config()

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.file_path
        if not target:
            raise ValueError("No file path — use Save As.")
        self._push_to_config()
        with open(target, "w") as f:
            yaml.dump(self.full_config, f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
        if path:
            self.file_path = path

    def _pull_from_config(self) -> None:
        px = self.full_config.get("pixelated", {})
        self.matrix          = [list(r) for r in px.get("pixel_state_matrix", [[0]])]
        self.inductor_values = list(px.get("inductor_values", []))

        # Normalize cap_values: flat List[float] → List[List[float]]
        raw_caps = px.get("cap_values", [])
        if raw_caps and isinstance(raw_caps[0], (int, float)):
            self.cap_values = [[float(v)] for v in raw_caps]
        else:
            self.cap_values = [[float(c) for c in bank] for bank in raw_caps]

        # Normalize state_list: List[List[int]] → List[List[List[int]]]
        raw_states = px.get("state_list", [])
        if raw_states and raw_states[0] and isinstance(raw_states[0][0], (int, float)):
            self.state_list = [[[int(bit)] for bit in sv] for sv in raw_states]
        else:
            self.state_list = [[list(bank) for bank in sv] for sv in raw_states]
        iss = px.get("inductor_switch_states", None)
        self.inductor_switch_states = [list(s) for s in iss] if iss else None
        cpp = px.get("cap_pixel_pairs", []) or []
        self.cap_pixel_pairs    = [[list(p) for p in pair] for pair in cpp]
        self.cap_pair_values    = list(px.get("cap_pair_values", []) or [])
        self.shunt_cap_values   = [float(v) for v in (px.get("shunt_cap_values", []) or [])]
        raw_layers = px.get("pixel_layers", None)
        if raw_layers:
            self.pixel_layers = [[int(v) for v in row] for row in raw_layers]
        else:
            self.pixel_layers = [[0] * len(row) for row in self.matrix]
        self._repair_lengths()

    def _push_to_config(self) -> None:
        if "pixelated" not in self.full_config:
            self.full_config["pixelated"] = {}
        px = self.full_config["pixelated"]
        px["pixel_state_matrix"] = self.matrix
        px["state_list"]         = self.state_list
        px["cap_values"]         = self.cap_values
        px["inductor_values"]    = self.inductor_values
        if self.inductor_switch_states:
            px["inductor_switch_states"] = self.inductor_switch_states
        else:
            px.pop("inductor_switch_states", None)
        px["cap_pixel_pairs"] = self.cap_pixel_pairs
        px["cap_pair_values"] = self.cap_pair_values
        px["shunt_cap_values"] = self.shunt_cap_values
        # Emit pixel_layers for differential (multi-layer) designs or whenever any
        # non-M10 layer is present; otherwise keep legacy single-layer configs clean.
        if self.is_differential() or any(v != 0 for row in self.pixel_layers for v in row):
            px["pixel_layers"] = self.pixel_layers
        else:
            px.pop("pixel_layers", None)

    def _repair_layers(self) -> None:
        """Keep pixel_layers the same shape as matrix (pad/truncate with 0)."""
        nr, nc = self.num_rows(), self.num_cols()
        # Row count
        while len(self.pixel_layers) < nr:
            self.pixel_layers.append([0] * nc)
        del self.pixel_layers[nr:]
        # Column count per row; also force empty cells to layer 0.
        for r in range(nr):
            row = self.pixel_layers[r]
            while len(row) < nc:
                row.append(0)
            del row[nc:]
            for c in range(nc):
                if int(self.matrix[r][c]) == 0:
                    row[c] = 0

    def _repair_lengths(self) -> None:
        """Clamp derived lists to match actual pixel counts."""
        self._repair_layers()
        n_sw  = len(self.sw_pixels())
        n_ind = len(self.ind_pixels())
        n_scap = len(self.scap_pixels())
        # shunt_cap_values — one float per type-5 (SCAP) pixel, row-first
        while len(self.shunt_cap_values) < n_scap:
            self.shunt_cap_values.append(DEFAULT_SCAP_F)
        self.shunt_cap_values = self.shunt_cap_values[:n_scap]
        # cap_values — each entry is a List[float] bank
        while len(self.cap_values) < n_sw:
            self.cap_values.append([DEFAULT_CAP_F])
        self.cap_values = self.cap_values[:n_sw]
        for bank in self.cap_values:
            if not bank:
                bank.append(DEFAULT_CAP_F)
        # inductor_values
        while len(self.inductor_values) < n_ind:
            self.inductor_values.append(0)
        self.inductor_values = self.inductor_values[:n_ind]
        # state_list: each state[sw_i] is a List[int] matching cap_values[sw_i]
        for state in self.state_list:
            while len(state) < n_sw:
                bank_len = len(self.cap_values[len(state)]) if len(state) < n_sw else 1
                state.append([0] * bank_len)
            del state[n_sw:]
            for sw_i, bank in enumerate(self.cap_values):
                bank_len = len(bank)
                if sw_i < len(state):
                    while len(state[sw_i]) < bank_len:
                        state[sw_i].append(0)
                    del state[sw_i][bank_len:]
        # inductor_switch_states columns
        if self.inductor_switch_states is not None:
            n_states = len(self.state_list)
            while len(self.inductor_switch_states) < n_states:
                self.inductor_switch_states.append([1] * n_ind)
            self.inductor_switch_states = self.inductor_switch_states[:n_states]
            for s in self.inductor_switch_states:
                while len(s) < n_ind:
                    s.append(1)
                del s[n_ind:]
        # cap_pair_values
        while len(self.cap_pair_values) < len(self.cap_pixel_pairs):
            self.cap_pair_values.append(DEFAULT_PAIR_F)
        self.cap_pair_values = self.cap_pair_values[:len(self.cap_pixel_pairs)]

    # ── Pixel enumeration ────────────────────────────────────────────────────

    def num_rows(self) -> int:
        return len(self.matrix)

    def num_cols(self) -> int:
        return len(self.matrix[0]) if self.matrix else 0

    def get_pixel(self, r: int, c: int) -> int:
        return int(self.matrix[r][c])

    def sw_pixels(self) -> List[Tuple[int, int, int]]:
        """(row, col, sw_index) for all SW pixels in row-first order."""
        out, idx = [], 0
        for r in range(self.num_rows()):
            for c in range(self.num_cols()):
                if self.matrix[r][c] == 3:
                    out.append((r, c, idx)); idx += 1
        return out

    def ind_pixels(self) -> List[Tuple[int, int, int]]:
        """(row, col, ind_index) for all IND pixels in row-first order."""
        out, idx = [], 0
        for r in range(self.num_rows()):
            for c in range(self.num_cols()):
                if self.matrix[r][c] == 2:
                    out.append((r, c, idx)); idx += 1
        return out

    def sw_index_at(self, row: int, col: int) -> Optional[int]:
        for r, c, i in self.sw_pixels():
            if r == row and c == col:
                return i
        return None

    def ind_index_at(self, row: int, col: int) -> Optional[int]:
        for r, c, i in self.ind_pixels():
            if r == row and c == col:
                return i
        return None

    def scap_pixels(self) -> List[Tuple[int, int, int]]:
        """(row, col, scap_index) for all SCAP (type-5) pixels in row-first order."""
        out, idx = [], 0
        for r in range(self.num_rows()):
            for c in range(self.num_cols()):
                if self.matrix[r][c] == 5:
                    out.append((r, c, idx)); idx += 1
        return out

    def scap_index_at(self, row: int, col: int) -> Optional[int]:
        for r, c, i in self.scap_pixels():
            if r == row and c == col:
                return i
        return None

    def cap_pair_index_at(self, row: int, col: int) -> Optional[int]:
        for i, pair in enumerate(self.cap_pixel_pairs):
            if [row, col] in pair:
                return i
        return None

    # ── Pixel mutation ───────────────────────────────────────────────────────

    def set_pixel(self, row: int, col: int, new_type: int) -> None:
        old_type = int(self.matrix[row][col])
        if old_type == new_type:
            return

        # Remove from any cap pair first (pair becomes invalid on type change)
        self._drop_cap_pairs_for(row, col)

        # --- Remove bookkeeping for old type BEFORE matrix change ---
        if old_type == 3:
            si = self.sw_index_at(row, col)
            if si is not None:
                _pop_idx(self.cap_values, si)
                for state in self.state_list:
                    _pop_idx(state, si)

        elif old_type == 2:
            ii = self.ind_index_at(row, col)
            if ii is not None:
                _pop_idx(self.inductor_values, ii)
                if self.inductor_switch_states:
                    for s in self.inductor_switch_states:
                        _pop_idx(s, ii)

        elif old_type == 5:
            ci = self.scap_index_at(row, col)
            if ci is not None:
                _pop_idx(self.shunt_cap_values, ci)

        # --- Commit the change ---
        self.matrix[row][col] = new_type
        # Clearing a cell resets its metal layer to M10 (0); other transitions
        # keep the existing layer so e.g. a metal pixel turned into a shunt cap
        # stays on its chosen layer.
        if new_type == 0 and row < len(self.pixel_layers) and col < len(self.pixel_layers[row]):
            self.pixel_layers[row][col] = 0

        # --- Insert bookkeeping for new type AFTER matrix change ---
        if new_type == 3:
            si = self.sw_index_at(row, col)
            if si is not None:
                self.cap_values.insert(si, [DEFAULT_CAP_F])
                for state in self.state_list:
                    state.insert(si, [0])

        elif new_type == 2:
            ii = self.ind_index_at(row, col)
            if ii is not None:
                self.inductor_values.insert(ii, 0)
                if self.inductor_switch_states:
                    for s in self.inductor_switch_states:
                        s.insert(ii, 1)

        elif new_type == 5:
            ci = self.scap_index_at(row, col)
            if ci is not None:
                self.shunt_cap_values.insert(ci, DEFAULT_SCAP_F)

    # ── Cap bank management ──────────────────────────────────────────────────

    def add_cap_to_bank(self, sw_idx: int, cap_value: float = DEFAULT_CAP_F) -> None:
        """Append a new capacitor to switch sw_idx's bank."""
        if sw_idx < len(self.cap_values):
            self.cap_values[sw_idx].append(cap_value)
            for state in self.state_list:
                if sw_idx < len(state):
                    state[sw_idx].append(0)

    def remove_cap_from_bank(self, sw_idx: int, cap_idx: int) -> None:
        """Remove capacitor cap_idx from switch sw_idx's bank (min 1 cap kept)."""
        if sw_idx < len(self.cap_values) and len(self.cap_values[sw_idx]) > 1:
            self.cap_values[sw_idx].pop(cap_idx)
            for state in self.state_list:
                if sw_idx < len(state) and cap_idx < len(state[sw_idx]):
                    state[sw_idx].pop(cap_idx)

    # ── Cap pairs ────────────────────────────────────────────────────────────

    def add_cap_pair(self, r1: int, c1: int, r2: int, c2: int,
                     cap_value: float) -> bool:
        if [r1, c1] == [r2, c2]:
            return False
        if self.matrix[r1][c1] == 0 or self.matrix[r2][c2] == 0:
            return False
        self._drop_cap_pairs_for(r1, c1)
        self._drop_cap_pairs_for(r2, c2)
        self.cap_pixel_pairs.append([[r1, c1], [r2, c2]])
        self.cap_pair_values.append(cap_value)
        return True

    def remove_cap_pair(self, pair_idx: int) -> None:
        if 0 <= pair_idx < len(self.cap_pixel_pairs):
            self.cap_pixel_pairs.pop(pair_idx)
            _pop_idx(self.cap_pair_values, pair_idx)

    def _drop_cap_pairs_for(self, row: int, col: int) -> None:
        keep_p, keep_v = [], []
        for pair, val in zip(self.cap_pixel_pairs, self.cap_pair_values):
            if [row, col] not in pair:
                keep_p.append(pair); keep_v.append(val)
        self.cap_pixel_pairs = keep_p
        self.cap_pair_values = keep_v

    # ── State list ───────────────────────────────────────────────────────────

    def add_state(self) -> None:
        n_ind = len(self.ind_pixels())
        self.state_list.append([[0] * len(bank) for bank in self.cap_values])
        if self.inductor_switch_states is not None:
            self.inductor_switch_states.append([1] * n_ind)

    def remove_state(self, idx: int) -> None:
        if 0 <= idx < len(self.state_list):
            self.state_list.pop(idx)
            if self.inductor_switch_states and idx < len(self.inductor_switch_states):
                self.inductor_switch_states.pop(idx)

    def enable_inductor_switch_states(self) -> None:
        if self.inductor_switch_states is None:
            n_ind    = len(self.ind_pixels())
            n_states = len(self.state_list)
            self.inductor_switch_states = [[1] * n_ind for _ in range(n_states)]

    def disable_inductor_switch_states(self) -> None:
        self.inductor_switch_states = None

    # ── Grid resizing ────────────────────────────────────────────────────────

    def add_row(self, idx: int) -> None:
        """Insert a blank row before row *idx* (0 = top, num_rows() = bottom)."""
        new_row = [0] * self.num_cols()
        self.matrix.insert(idx, new_row)
        self.pixel_layers.insert(idx, [0] * self.num_cols())
        for pair in self.cap_pixel_pairs:
            for p in pair:
                if p[0] >= idx:
                    p[0] += 1

    def remove_row(self, idx: int) -> None:
        if self.num_rows() <= 1:
            return
        # Collect sw/ind indices in this row (descending so pops don't shift)
        sw_in  = sorted([si for r, c, si in self.sw_pixels()  if r == idx], reverse=True)
        ind_in = sorted([ii for r, c, ii in self.ind_pixels() if r == idx], reverse=True)
        scap_in = sorted([ci for r, c, ci in self.scap_pixels() if r == idx], reverse=True)
        for si in sw_in:
            _pop_idx(self.cap_values, si)
            for state in self.state_list:
                _pop_idx(state, si)
        for ii in ind_in:
            _pop_idx(self.inductor_values, ii)
            if self.inductor_switch_states:
                for s in self.inductor_switch_states:
                    _pop_idx(s, ii)
        for ci in scap_in:
            _pop_idx(self.shunt_cap_values, ci)
        # Drop cap pairs referencing this row, keep others + shift rows
        keep_p, keep_v = [], []
        for pair, val in zip(self.cap_pixel_pairs, self.cap_pair_values):
            if pair[0][0] == idx or pair[1][0] == idx:
                continue
            adjusted = [[p[0] - (1 if p[0] > idx else 0), p[1]] for p in pair]
            keep_p.append(adjusted); keep_v.append(val)
        self.cap_pixel_pairs = keep_p
        self.cap_pair_values = keep_v
        self.matrix.pop(idx)
        if idx < len(self.pixel_layers):
            self.pixel_layers.pop(idx)

    def add_col(self, idx: int) -> None:
        """Insert a blank column before col *idx* (0 = left, num_cols() = right)."""
        for row in self.matrix:
            row.insert(idx, 0)
        for row in self.pixel_layers:
            row.insert(idx, 0)
        for pair in self.cap_pixel_pairs:
            for p in pair:
                if p[1] >= idx:
                    p[1] += 1

    def remove_col(self, idx: int) -> None:
        if self.num_cols() <= 1:
            return
        sw_in  = sorted([si for r, c, si in self.sw_pixels()  if c == idx], reverse=True)
        ind_in = sorted([ii for r, c, ii in self.ind_pixels() if c == idx], reverse=True)
        scap_in = sorted([ci for r, c, ci in self.scap_pixels() if c == idx], reverse=True)
        for si in sw_in:
            _pop_idx(self.cap_values, si)
            for state in self.state_list:
                _pop_idx(state, si)
        for ii in ind_in:
            _pop_idx(self.inductor_values, ii)
            if self.inductor_switch_states:
                for s in self.inductor_switch_states:
                    _pop_idx(s, ii)
        for ci in scap_in:
            _pop_idx(self.shunt_cap_values, ci)
        keep_p, keep_v = [], []
        for pair, val in zip(self.cap_pixel_pairs, self.cap_pair_values):
            if pair[0][1] == idx or pair[1][1] == idx:
                continue
            adjusted = [[p[0], p[1] - (1 if p[1] > idx else 0)] for p in pair]
            keep_p.append(adjusted); keep_v.append(val)
        self.cap_pixel_pairs = keep_p
        self.cap_pair_values = keep_v
        for row in self.matrix:
            row.pop(idx)
        for row in self.pixel_layers:
            if idx < len(row):
                row.pop(idx)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "DesignModel":
        m = cls()
        m.full_config = {
            "mode": "pixelated_design_simple",
            "pixelated": {
                "pixel_state_matrix": [
                    [1, 1, 2, 1, 0],
                    [0, 3, 1, 3, 1],
                    [0, 0, 2, 1, 0],
                ],
                "state_list":      [[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]],
                "cap_values":      [[1.0e-12], [1.0e-12]],
                "inductor_values": [0],
                "cap_pixel_pairs": [],
                "cap_pair_values": [],
                "inductor_translation_table": DEFAULT_INDUCTOR_TABLE,
                "inductor_off_translation_table": DEFAULT_INDUCTOR_OFF_TABLE,
                "feedline": {
                    "feedline_row":     1,
                    "feedline_width":   10.2,
                    "feedline_padding": 10.0,
                },
                "inductor_s1p":     DEFAULT_INDUCTOR_TABLE[0],
                "inductor_off_s1p": DEFAULT_INDUCTOR_OFF_TABLE[0],
            },
        }
        m._pull_from_config()
        return m


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pop_idx(lst: list, idx: int) -> None:
    if 0 <= idx < len(lst):
        lst.pop(idx)


def _fmt_cap(f_val: float) -> str:
    """Format a capacitance value in human-readable pF."""
    pf = f_val * 1e12
    if pf >= 1:
        return f"{pf:.3g} pF"
    ff = f_val * 1e15
    return f"{ff:.3g} fF"


# ─────────────────────────────────────────────────────────────────────────────
# Cap Pair Dialog
# ─────────────────────────────────────────────────────────────────────────────

class CapPairDialog(QDialog):
    def __init__(self, model: DesignModel, r1: int, c1: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Cap Pair")
        self.model = model
        layout = QFormLayout(self)

        info = QLabel(f"Pairing pixel ({r1},{c1}) with:")
        info.setStyleSheet("font-style: italic;")
        layout.addRow(info)

        self.row_spin = QSpinBox()
        self.row_spin.setRange(0, model.num_rows() - 1)
        self.col_spin = QSpinBox()
        self.col_spin.setRange(0, model.num_cols() - 1)
        self.cap_spin = QDoubleSpinBox()
        self.cap_spin.setRange(0.001, 1000.0)
        self.cap_spin.setDecimals(3)
        self.cap_spin.setValue(1.0)
        self.cap_spin.setSuffix(" pF")
        self.cap_spin.setSingleStep(0.1)

        layout.addRow("Target row:", self.row_spin)
        layout.addRow("Target col:", self.col_spin)
        layout.addRow("Cap value:",  self.cap_spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def result(self) -> Tuple[int, int, float]:
        return (self.row_spin.value(),
                self.col_spin.value(),
                self.cap_spin.value() * 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Pixel Grid Widget
# ─────────────────────────────────────────────────────────────────────────────

class PixelGridWidget(QWidget):
    """
    Renders the pixel matrix and handles selection / brush interaction.

    Zoom is controlled by ``cell_size`` (px per grid cell).
    Use zoom_fit(viewport_w, viewport_h) to auto-fit after loading a design.
    Ctrl+scroll wheel zooms in/out interactively.

    Signals
    -------
    pixel_clicked(row, col)  — emitted in selection mode on mouse press
    pixel_painted(row, col)  — emitted in brush mode when a cell is changed
    """

    pixel_clicked = pyqtSignal(int, int)
    pixel_painted = pyqtSignal(int, int)

    CELL_MIN  = 24
    CELL_MAX  = 200
    PAD       = 32   # outer padding reserved for axis labels

    def __init__(self, model: DesignModel, parent=None):
        super().__init__(parent)
        self.model      = model
        self._fit_mode  = True   # True = fill widget size; False = fixed _manual_cs
        self._manual_cs = 76     # cell size used when _fit_mode is False
        self.mode       = MODE_SELECT
        self.brush_type = 0
        self.selected:  Optional[Tuple[int, int]] = None
        self._hover:    Optional[Tuple[int, int]] = None
        self._dragging  = False
        self.setMouseTracking(True)
        # Expand to fill whatever space the layout gives us
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ── cell_size property (computed live in fit mode) ────────────────────────

    @property
    def cell_size(self) -> int:
        if self._fit_mode:
            return self._compute_fit_cs()
        return self._manual_cs

    def _compute_fit_cs(self) -> int:
        """Compute cell size so the full grid fits in self.width() × self.height()."""
        nr = self.model.num_rows()
        nc = self.model.num_cols()
        if nr == 0 or nc == 0:
            return self._manual_cs
        w = max(self.width(),  10)
        h = max(self.height(), 10)
        cs_w = (w - 2 * self.PAD - 2) // nc
        cs_h = (h - 2 * self.PAD - 26) // nr   # 26 px reserved for legend
        return max(self.CELL_MIN, min(cs_w, cs_h, self.CELL_MAX))

    # ── public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def set_brush_type(self, t: int) -> None:
        self.brush_type = t

    def set_selected(self, row: int, col: int) -> None:
        self.selected = (row, col)
        self.update()

    def clear_selection(self) -> None:
        self.selected = None
        self.update()

    def refresh(self) -> None:
        if not self._fit_mode:
            self.updateGeometry()   # tell scroll area about new fixed size
        self.update()

    # ── zoom ─────────────────────────────────────────────────────────────────

    def enter_fit_mode(self) -> None:
        """Switch to fit mode: cell_size adapts to widget dimensions automatically."""
        self._fit_mode = True
        self.update()

    def zoom_in(self) -> None:
        """Snapshot current computed cs, scale up, lock to manual mode."""
        self._manual_cs = min(int(self.cell_size * 1.25), self.CELL_MAX)
        self._fit_mode  = False
        self.updateGeometry()
        self.update()

    def zoom_out(self) -> None:
        self._manual_cs = max(int(self.cell_size / 1.25), self.CELL_MIN)
        self._fit_mode  = False
        self.updateGeometry()
        self.update()

    def zoom_reset(self) -> None:
        self._manual_cs = 76
        self._fit_mode  = False
        self.updateGeometry()
        self.update()

    # ── sizing ────────────────────────────────────────────────────────────────

    def sizeHint(self) -> QSize:
        nr = self.model.num_rows()
        nc = self.model.num_cols()
        if self._fit_mode:
            # In fit mode the layout stretches us; return a small minimum hint
            cs = self.CELL_MIN
        else:
            cs = self._manual_cs
        return QSize(nc * cs + 2 * self.PAD + 4,
                     nr * cs + 2 * self.PAD + 26)

    def minimumSizeHint(self) -> QSize:
        # Always report a minimum that shows all cells at CELL_MIN size
        nr = self.model.num_rows()
        nc = self.model.num_cols()
        cs = self.CELL_MIN
        return QSize(nc * cs + 2 * self.PAD + 4,
                     nr * cs + 2 * self.PAD + 26)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _rc(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        cs  = self.cell_size
        pad = self.PAD
        c   = (x - pad) // cs
        r   = (y - pad) // cs
        if 0 <= r < self.model.num_rows() and 0 <= c < self.model.num_cols():
            return (r, c)
        return None

    def _cell_rect(self, r: int, c: int) -> QRect:
        cs  = self.cell_size
        pad = self.PAD
        return QRect(pad + c * cs, pad + r * cs, cs - 1, cs - 1)

    # ── painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        self._draw(p)

    def _draw(self, p: QPainter) -> None:
        model     = self.model
        nr, nc    = model.num_rows(), model.num_cols()
        cs        = self.cell_size
        pad       = self.PAD

        # Derived font sizes — scale with cell, keep readable minimums
        fs_main  = max(7,  int(cs * 0.125))   # primary label (type/index)
        fs_sub   = max(6,  int(cs * 0.092))   # secondary label (nH, pF)
        fs_micro = max(5,  int(cs * 0.075))   # tiny r,c corner tag

        # In differential (multi-layer) designs, colour non-empty cells by their
        # metal layer so M10 / M9 / double are visible; advanced types keep their
        # text label.  Other modes use the legacy per-type colour.
        multilayer = model.is_differential()

        # --- build cap-pair lookup (row,col) -> pair_index ---
        pair_map: Dict[Tuple[int,int], int] = {}
        for i, pair in enumerate(model.cap_pixel_pairs):
            pair_map[tuple(pair[0])] = i
            pair_map[tuple(pair[1])] = i

        # ── Draw cells ──────────────────────────────────────────────────────
        for r in range(nr):
            for c in range(nc):
                ptype = model.get_pixel(r, c)
                label, base_color = PIXEL_INFO.get(ptype, ("?", QColor("#222")))
                rect  = self._cell_rect(r, c)

                # Base colour by metal layer for differential designs (non-empty
                # cells); empty (GND) cells and non-differential designs unchanged.
                if multilayer and ptype != 0:
                    base_color = LAYER_INFO.get(model.layer_at(r, c), LAYER_INFO[0])[1]

                # Fill
                fill = QColor(base_color)
                if self._hover == (r, c) and self.mode == MODE_BRUSH:
                    fill = fill.lighter(160)
                p.setBrush(QBrush(fill))

                # Border — thicker + white when selected
                if self.selected == (r, c):
                    p.setPen(QPen(QColor("#ffffff"), 3))
                elif self._hover == (r, c):
                    p.setPen(QPen(QColor("#cccccc"), 2))
                else:
                    p.setPen(QPen(QColor("#111111"), 1))
                p.drawRect(rect)

                # --- build text ---
                sw_idx  = model.sw_index_at(r, c)
                ind_idx = model.ind_index_at(r, c)

                if ptype == 2 and ind_idx is not None:
                    top_text = f"IND_{ind_idx}"
                    iv       = (model.inductor_values[ind_idx]
                                if ind_idx < len(model.inductor_values) else 0)
                    bot_text = INDUCTOR_LABELS[iv] if iv < len(INDUCTOR_LABELS) else "?"
                elif ptype == 3 and sw_idx is not None:
                    top_text = f"SW_{sw_idx}"
                    bank     = (model.cap_values[sw_idx]
                                if sw_idx < len(model.cap_values) else [DEFAULT_CAP_F])
                    if len(bank) == 1:
                        bot_text = _fmt_cap(bank[0])
                    else:
                        bot_text = f"{len(bank)}×" + _fmt_cap(bank[0])
                elif ptype == 4:
                    top_text = "VIA"; bot_text = ""
                elif ptype == 1 and multilayer:
                    # Metal pixel: show its layer (colour already encodes it too).
                    top_text = LAYER_INFO.get(model.layer_at(r, c), LAYER_INFO[0])[0]
                    bot_text = ""
                else:
                    top_text = label; bot_text = ""

                pi = pair_map.get((r, c))
                if pi is not None:
                    bot_text = f"CP{pi}" + (f" {bot_text}" if bot_text else "")

                # Primary label (upper half)
                p.setPen(QPen(QColor("#ffffff")))
                p.setFont(QFont("Monospace", fs_main, QFont.Bold))
                half = rect.height() // 2
                top_rect = QRect(rect.x()+2, rect.y()+2, rect.width()-4,
                                 half if bot_text else rect.height()-4)
                p.drawText(top_rect, Qt.AlignCenter, top_text)

                # Secondary label (lower half)
                if bot_text:
                    p.setFont(QFont("Monospace", fs_sub))
                    p.setPen(QPen(QColor("#f0f0c0")))
                    bot_rect = QRect(rect.x()+2, rect.y()+half,
                                     rect.width()-4, rect.height()-half-2)
                    p.drawText(bot_rect, Qt.AlignCenter, bot_text)

                # ── r,c micro-label in top-left corner ──────────────────────
                # Draw a small semi-transparent dark pill behind the text
                if cs >= 36:    # skip when cells are tiny
                    micro_text = f"{r},{c}"
                    p.setFont(QFont("Monospace", fs_micro))
                    fm   = p.fontMetrics()
                    tw   = fm.horizontalAdvance(micro_text) + 4
                    th   = fm.height()
                    bx   = rect.x() + 2
                    by   = rect.y() + 2
                    p.setBrush(QBrush(QColor(0, 0, 0, 140)))
                    p.setPen(Qt.NoPen)
                    p.drawRoundedRect(bx, by, tw, th, 2, 2)
                    p.setPen(QPen(QColor("#e8e8e8")))
                    p.drawText(bx + 2, by + th - 2, micro_text)

        # ── Cap pair connecting lines ────────────────────────────────────────
        for i, pair in enumerate(model.cap_pixel_pairs):
            (r1, c1), (r2, c2) = pair
            x1 = pad + c1*cs + cs//2;  y1 = pad + r1*cs + cs//2
            x2 = pad + c2*cs + cs//2;  y2 = pad + r2*cs + cs//2
            color = PAIR_COLORS[i % len(PAIR_COLORS)]
            p.setPen(QPen(color, 2, Qt.DashLine))
            p.drawLine(x1, y1, x2, y2)

        # ── Axis labels ──────────────────────────────────────────────────────
        fs_ax   = max(7, min(10, int(cs * 0.14)))
        ax_font = QFont("Monospace", fs_ax, QFont.Bold)
        p.setFont(ax_font)
        fm_ax   = p.fontMetrics()

        # Column headers (above each column)
        for c in range(nc):
            text = f"c{c}"
            tw   = fm_ax.horizontalAdvance(text)
            th   = fm_ax.height()
            cx   = pad + c*cs + cs//2 - tw//2
            cy   = pad - 4 - th            # just above the grid
            # Dark background pill
            p.setBrush(QBrush(QColor(0, 0, 0, 180)))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(cx - 2, cy, tw + 4, th + 2, 3, 3)
            p.setPen(QPen(QColor("#f5d020")))   # bright yellow
            p.drawText(cx, cy + th, text)

        # Row labels (left of each row)
        for r in range(nr):
            text = f"r{r}"
            tw   = fm_ax.horizontalAdvance(text)
            th   = fm_ax.height()
            ry   = pad + r*cs + cs//2 + th//2 - 2
            rx   = pad - tw - 5
            p.setBrush(QBrush(QColor(0, 0, 0, 180)))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(rx - 2, ry - th, tw + 4, th + 2, 3, 3)
            p.setPen(QPen(QColor("#6dd5fa")))   # bright cyan
            p.drawText(rx, ry, text)

        # ── Legend ───────────────────────────────────────────────────────────
        legend_y = pad + nr*cs + 6
        sw       = 13
        p.setFont(QFont("Monospace", max(7, int(cs * 0.10))))
        lx = pad
        for ptype, (lbl, color) in PIXEL_INFO.items():
            p.setBrush(QBrush(color))
            p.setPen(QPen(QColor("#111111"), 1))
            p.drawRect(lx, legend_y, sw, sw)
            p.setPen(QPen(QColor("#dddddd")))
            p.drawText(lx + sw + 3, legend_y + sw - 1, f"{ptype}:{lbl}")
            lx += sw + p.fontMetrics().horizontalAdvance(f"{ptype}:{lbl}") + 14

    # ── Mouse / wheel events ─────────────────────────────────────────────────

    def mousePressEvent(self, ev) -> None:
        rc = self._rc(ev.x(), ev.y())
        if rc is None:
            return
        r, c = rc
        if self.mode == MODE_SELECT:
            self.selected = (r, c)
            self.update()
            self.pixel_clicked.emit(r, c)
        else:  # BRUSH
            self._dragging = True
            self._do_brush(r, c)

    def mouseMoveEvent(self, ev) -> None:
        rc = self._rc(ev.x(), ev.y())
        old_hover = self._hover
        self._hover = rc
        if old_hover != rc:
            self.update()
        if self._dragging and rc and self.mode == MODE_BRUSH:
            self._do_brush(*rc)

    def mouseReleaseEvent(self, _ev) -> None:
        self._dragging = False

    def leaveEvent(self, _ev) -> None:
        self._hover = None
        self.update()

    def wheelEvent(self, ev) -> None:
        """Ctrl+scroll → zoom in/out."""
        if ev.modifiers() & Qt.ControlModifier:
            delta = ev.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            elif delta < 0:
                self.zoom_out()
            ev.accept()
        else:
            ev.ignore()

    def _do_brush(self, r: int, c: int) -> None:
        if self.brush_type not in (0, 1):
            return
        if self.model.get_pixel(r, c) != self.brush_type:
            self.model.set_pixel(r, c, self.brush_type)
            self.update()
            self.pixel_painted.emit(r, c)


# ─────────────────────────────────────────────────────────────────────────────
# Property Panel
# ─────────────────────────────────────────────────────────────────────────────

class PropertyPanel(QGroupBox):
    """
    Bottom panel that shows and edits properties of the currently selected pixel.
    Visible when a pixel is selected in selection mode.
    """

    changed = pyqtSignal()  # data has changed → caller should refresh grid+table

    def __init__(self, model: DesignModel, parent=None):
        super().__init__("Pixel Properties", parent)
        self.model      = model
        self.selected:  Optional[Tuple[int, int]] = None
        self._loading   = False
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(12)

        # Hint when nothing is selected
        self._hint = QLabel("Click a pixel to inspect / edit it.")
        self._hint.setStyleSheet("color: #888; font-style: italic;")
        root.addWidget(self._hint)

        # ── Type selector ─────────────────────────────────────────────────
        gb_type = QGroupBox("Type")
        gb_type.setMinimumWidth(120)
        vl = QVBoxLayout(gb_type)
        self._type_combo = QComboBox()
        for ptype, (name, _) in PIXEL_INFO.items():
            self._type_combo.addItem(f"{ptype} — {name}", ptype)
        self._type_combo.currentIndexChanged.connect(self._on_type)
        vl.addWidget(self._type_combo)
        root.addWidget(gb_type)
        self._type_gb = gb_type

        # ── Inductor ──────────────────────────────────────────────────────
        gb_ind = QGroupBox("Inductor value")
        vl2 = QVBoxLayout(gb_ind)
        self._ind_combo = QComboBox()
        for lbl in INDUCTOR_LABELS:
            self._ind_combo.addItem(lbl)
        self._ind_combo.currentIndexChanged.connect(self._on_ind)
        vl2.addWidget(self._ind_combo)
        root.addWidget(gb_ind)
        self._ind_gb = gb_ind

        # ── Shunt cap (type 5) ────────────────────────────────────────────
        gb_scap = QGroupBox("Shunt cap value")
        vl_sc = QVBoxLayout(gb_scap)
        self._scap_spin = QDoubleSpinBox()
        self._scap_spin.setRange(0.001, 1000.0)
        self._scap_spin.setDecimals(3)
        self._scap_spin.setSuffix(" pF")
        self._scap_spin.setSingleStep(0.01)
        self._scap_spin.valueChanged.connect(self._on_scap)
        vl_sc.addWidget(self._scap_spin)
        root.addWidget(gb_scap)
        self._scap_gb = gb_scap

        # ── Metal layer (differential multi-layer designs only) ───────────────
        gb_layer = QGroupBox("Metal layer")
        vl_ly = QVBoxLayout(gb_layer)
        self._layer_combo = QComboBox()
        for code, (name, _) in LAYER_INFO.items():
            self._layer_combo.addItem(f"{code} — {name}", code)
        self._layer_combo.currentIndexChanged.connect(self._on_layer)
        vl_ly.addWidget(self._layer_combo)
        root.addWidget(gb_layer)
        self._layer_gb = gb_layer

        # ── Switch cap bank ───────────────────────────────────────────────
        gb_sw = QGroupBox("Switch cap bank")
        gb_sw.setMinimumWidth(220)
        vl3 = QVBoxLayout(gb_sw)
        vl3.setSpacing(3)
        self._sw_bank_widget = QWidget()
        self._sw_bank_layout = QVBoxLayout(self._sw_bank_widget)
        self._sw_bank_layout.setSpacing(2)
        self._sw_bank_layout.setContentsMargins(0, 0, 0, 0)
        vl3.addWidget(self._sw_bank_widget)
        self._sw_add_btn = QPushButton("+ Add cap")
        self._sw_add_btn.clicked.connect(self._on_add_cap_to_bank)
        vl3.addWidget(self._sw_add_btn)
        root.addWidget(gb_sw)
        self._sw_gb = gb_sw
        self._sw_spins: List[QDoubleSpinBox] = []

        # ── Cap pair ──────────────────────────────────────────────────────
        gb_cp = QGroupBox("Cap pair")
        gb_cp.setMinimumWidth(200)
        vl4 = QVBoxLayout(gb_cp)
        self._cp_info     = QLabel("No pair")
        self._cp_val_spin = QDoubleSpinBox()
        self._cp_val_spin.setRange(0.001, 1000.0)
        self._cp_val_spin.setDecimals(3)
        self._cp_val_spin.setSuffix(" pF")
        self._cp_val_spin.setSingleStep(0.1)
        self._cp_val_spin.valueChanged.connect(self._on_pair_val)
        self._cp_add_btn  = QPushButton("Create pair…")
        self._cp_rem_btn  = QPushButton("Remove pair")
        self._cp_add_btn.clicked.connect(self._on_add_pair)
        self._cp_rem_btn.clicked.connect(self._on_rem_pair)
        row_btns = QHBoxLayout()
        row_btns.addWidget(self._cp_add_btn)
        row_btns.addWidget(self._cp_rem_btn)
        vl4.addWidget(self._cp_info)
        vl4.addLayout(row_btns)
        vl4.addWidget(self._cp_val_spin)
        root.addWidget(gb_cp)
        self._cp_gb = gb_cp

        root.addStretch()

    # ── public API ────────────────────────────────────────────────────────────

    def select_pixel(self, row: int, col: int) -> None:
        self.selected = (row, col)
        self._refresh()

    def clear_selection(self) -> None:
        self.selected = None
        self._refresh()

    def refresh(self) -> None:
        self._refresh()

    # ── internal refresh ─────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self._loading = True
        try:
            self._do_refresh()
        finally:
            self._loading = False

    def _do_refresh(self) -> None:
        if self.selected is None:
            self._hint.show()
            self._type_gb.hide(); self._ind_gb.hide()
            self._sw_gb.hide();   self._cp_gb.hide()
            self._scap_gb.hide(); self._layer_gb.hide()
            return

        r, c   = self.selected
        model  = self.model
        ptype  = model.get_pixel(r, c)

        self._hint.hide()
        self._type_gb.show()

        # Type combo
        idx = self._type_combo.findData(ptype)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)

        # Inductor
        if ptype == 2:
            ii = model.ind_index_at(r, c)
            self._ind_gb.show()
            if ii is not None and ii < len(model.inductor_values):
                self._ind_combo.setCurrentIndex(model.inductor_values[ii])
        else:
            self._ind_gb.hide()

        # Switch cap bank
        if ptype == 3:
            si = model.sw_index_at(r, c)
            self._sw_gb.show()
            self._rebuild_sw_bank_ui(si)
        else:
            self._sw_gb.hide()

        # Shunt cap value (type 5)
        if ptype == 5:
            ci = model.scap_index_at(r, c)
            self._scap_gb.show()
            if ci is not None and ci < len(model.shunt_cap_values):
                self._scap_spin.setValue(model.shunt_cap_values[ci] * 1e12)
        else:
            self._scap_gb.hide()

        # Metal layer (differential multi-layer designs; any non-empty pixel)
        if model.is_differential() and ptype > 0:
            self._layer_gb.show()
            li = self._layer_combo.findData(model.layer_at(r, c))
            if li >= 0:
                self._layer_combo.setCurrentIndex(li)
        else:
            self._layer_gb.hide()

        # Cap pair (available for any non-ground pixel)
        if ptype > 0:
            self._cp_gb.show()
            pi = model.cap_pair_index_at(r, c)
            if pi is not None:
                pair  = model.cap_pixel_pairs[pi]
                other = pair[1] if pair[0] == [r, c] else pair[0]
                val   = (model.cap_pair_values[pi]
                         if pi < len(model.cap_pair_values) else DEFAULT_PAIR_F)
                self._cp_info.setText(f"Pair {pi} ↔ ({other[0]},{other[1]})")
                self._cp_val_spin.setValue(val * 1e12)
                self._cp_val_spin.show()
                self._cp_rem_btn.show()
                self._cp_add_btn.hide()
            else:
                self._cp_info.setText("No pair")
                self._cp_val_spin.hide()
                self._cp_rem_btn.hide()
                self._cp_add_btn.show()
        else:
            self._cp_gb.hide()

    # ── signal handlers ───────────────────────────────────────────────────────

    def _on_type(self, _idx: int) -> None:
        if self._loading or self.selected is None:
            return
        r, c      = self.selected
        new_type  = self._type_combo.currentData()
        self.model.set_pixel(r, c, new_type)
        self._refresh()
        self.changed.emit()

    def _on_ind(self, idx: int) -> None:
        if self._loading or self.selected is None:
            return
        r, c = self.selected
        ii   = self.model.ind_index_at(r, c)
        if ii is not None and ii < len(self.model.inductor_values):
            self.model.inductor_values[ii] = idx
            self.changed.emit()

    def _on_scap(self, pf: float) -> None:
        if self._loading or self.selected is None:
            return
        r, c = self.selected
        ci   = self.model.scap_index_at(r, c)
        if ci is not None and ci < len(self.model.shunt_cap_values):
            self.model.shunt_cap_values[ci] = pf * 1e-12
            self.changed.emit()

    def _on_layer(self, _idx: int) -> None:
        if self._loading or self.selected is None:
            return
        r, c = self.selected
        self.model.set_layer(r, c, self._layer_combo.currentData())
        self._refresh()
        self.changed.emit()

    def _rebuild_sw_bank_ui(self, si: Optional[int]) -> None:
        """Clear and rebuild the per-cap spinbox rows for switch si."""
        # Remove old rows
        while self._sw_bank_layout.count():
            item = self._sw_bank_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sw_spins = []

        if si is None or si >= len(self.model.cap_values):
            return

        bank = self.model.cap_values[si]
        can_remove = len(bank) > 1
        for cap_idx, cap_f in enumerate(bank):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)
            row_l.addWidget(QLabel(f"Cap {cap_idx}:"))
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 1000.0)
            spin.setDecimals(3)
            spin.setSuffix(" pF")
            spin.setSingleStep(0.1)
            spin.setValue(cap_f * 1e12)
            spin.valueChanged.connect(
                lambda pf, j=cap_idx: self._on_sw_cap(j, pf))
            row_l.addWidget(spin, stretch=1)
            self._sw_spins.append(spin)
            rem_btn = QPushButton("×")
            rem_btn.setFixedWidth(24)
            rem_btn.setEnabled(can_remove)
            rem_btn.clicked.connect(
                lambda checked, j=cap_idx: self._on_rem_cap_from_bank(j))
            row_l.addWidget(rem_btn)
            self._sw_bank_layout.addWidget(row_w)

    def _on_sw_cap(self, cap_idx: int, pf: float) -> None:
        if self._loading or self.selected is None:
            return
        r, c = self.selected
        si   = self.model.sw_index_at(r, c)
        if si is not None and si < len(self.model.cap_values):
            if cap_idx < len(self.model.cap_values[si]):
                self.model.cap_values[si][cap_idx] = pf * 1e-12
                self.changed.emit()

    def _on_add_cap_to_bank(self) -> None:
        if self.selected is None:
            return
        r, c = self.selected
        si   = self.model.sw_index_at(r, c)
        if si is not None:
            self.model.add_cap_to_bank(si)
            self._refresh()
            self.changed.emit()

    def _on_rem_cap_from_bank(self, cap_idx: int) -> None:
        if self.selected is None:
            return
        r, c = self.selected
        si   = self.model.sw_index_at(r, c)
        if si is not None:
            self.model.remove_cap_from_bank(si, cap_idx)
            self._refresh()
            self.changed.emit()

    def _on_add_pair(self) -> None:
        if self.selected is None:
            return
        r1, c1 = self.selected
        dlg = CapPairDialog(self.model, r1, c1, self)
        if dlg.exec_() == QDialog.Accepted:
            r2, c2, val = dlg.result()
            if not self.model.add_cap_pair(r1, c1, r2, c2, val):
                QMessageBox.warning(self, "Cap Pair",
                    "Cannot create pair.\n"
                    "Both pixels must be non-ground (type ≥ 1) and must not be the same cell.")
                return
            self._refresh()
            self.changed.emit()

    def _on_rem_pair(self) -> None:
        if self.selected is None:
            return
        r, c = self.selected
        pi   = self.model.cap_pair_index_at(r, c)
        if pi is not None:
            self.model.remove_cap_pair(pi)
            self._refresh()
            self.changed.emit()

    def _on_pair_val(self, pf: float) -> None:
        if self._loading or self.selected is None:
            return
        r, c = self.selected
        pi   = self.model.cap_pair_index_at(r, c)
        if pi is not None and pi < len(self.model.cap_pair_values):
            self.model.cap_pair_values[pi] = pf * 1e-12
            self.changed.emit()


# ─────────────────────────────────────────────────────────────────────────────
# States Table Widget
# ─────────────────────────────────────────────────────────────────────────────

class StatesTableWidget(QWidget):
    """
    Right-panel table showing switch ON/OFF and inductor ON/OFF per state.
    Click a cell to toggle its value.  Add / remove states with the buttons.
    """

    changed = pyqtSignal()

    def __init__(self, model: DesignModel, parent=None):
        super().__init__(parent)
        self.model    = model
        self._loading = False
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        title = QLabel("Simulation States")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        lay.addWidget(title)

        # Control row
        ctrl = QHBoxLayout()
        self._add_btn = QPushButton("+ Add state")
        self._rem_btn = QPushButton("– Remove state")
        self._add_btn.clicked.connect(self._on_add)
        self._rem_btn.clicked.connect(self._on_remove)
        ctrl.addWidget(self._add_btn)
        ctrl.addWidget(self._rem_btn)
        lay.addLayout(ctrl)

        # Inductor switch states checkbox
        self._ind_cb = QCheckBox("Enable per-state inductor switching")
        self._ind_cb.stateChanged.connect(self._on_ind_toggle)
        lay.addWidget(self._ind_cb)

        # State count info
        self._info = QLabel()
        self._info.setStyleSheet("color: #888; font-size: 11px;")
        lay.addWidget(self._info)

        # Table
        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.cellClicked.connect(self._on_cell_click)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        lay.addWidget(self._table)

        # Legend row
        leg = QHBoxLayout()
        for color, label in [(QColor("#c0392b"), "SW ON"),
                              (QColor("#4a4a4a"), "SW OFF"),
                              (QColor("#c8861a"), "IND ON"),
                              (QColor("#4a7a4a"), "IND OFF")]:
            dot = QLabel("■")
            dot.setStyleSheet(f"color: {color.name()}; font-size: 16px;")
            leg.addWidget(dot)
            leg.addWidget(QLabel(label))
            leg.addSpacing(8)
        leg.addStretch()
        lay.addLayout(leg)

    # ── public ────────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        self._loading = True
        self._table.blockSignals(True)
        try:
            self._do_refresh()
        finally:
            self._table.blockSignals(False)
            self._loading = False

    # ── internal ──────────────────────────────────────────────────────────────

    def _do_refresh(self) -> None:
        model    = self.model
        sw_px    = model.sw_pixels()
        ind_px   = model.ind_pixels()
        n_sw     = len(sw_px)
        n_states = len(model.state_list)
        use_ind  = model.inductor_switch_states is not None
        n_ind    = len(ind_px) if use_ind else 0

        # Build flat column list: (sw_i, cap_j) for each switch bank, then IND cols
        sw_cols: List[Tuple[int, int]] = []
        for sw_i in range(n_sw):
            bank_len = len(model.cap_values[sw_i]) if sw_i < len(model.cap_values) else 1
            for cap_j in range(bank_len):
                sw_cols.append((sw_i, cap_j))

        n_sw_cols = len(sw_cols)
        n_cols    = n_sw_cols + n_ind

        self._ind_cb.setChecked(use_ind)
        self._info.setText(
            f"{n_states} state(s)  |  {n_sw} switch(es)  |  {len(ind_px)} inductor(s)")

        self._table.setRowCount(n_states)
        self._table.setColumnCount(n_cols)

        h_labels = [f"SW_{sw_i}_{cap_j}"
                    if sw_i < len(model.cap_values) and len(model.cap_values[sw_i]) > 1
                    else f"SW_{sw_i}"
                    for sw_i, cap_j in sw_cols]
        if use_ind:
            h_labels += [f"IND_{i}" for i in range(n_ind)]
        self._table.setHorizontalHeaderLabels(h_labels)
        self._table.setVerticalHeaderLabels([f"S{i}" for i in range(n_states)])

        for state_idx, state in enumerate(model.state_list):
            for col_idx, (sw_i, cap_j) in enumerate(sw_cols):
                bank = state[sw_i] if sw_i < len(state) else []
                val  = bank[cap_j] if cap_j < len(bank) else 0
                self._set_cell(state_idx, col_idx, val, kind="sw",
                               sw_i=sw_i, cap_j=cap_j)

        if use_ind and model.inductor_switch_states:
            for state_idx, ind_state in enumerate(model.inductor_switch_states):
                for j in range(n_ind):
                    val = ind_state[j] if j < len(ind_state) else 1
                    self._set_cell(state_idx, n_sw_cols + j, val, kind="ind",
                                   sw_i=0, cap_j=j)

    def _set_cell(self, row: int, col: int, val: int, kind: str,
                  sw_i: int = 0, cap_j: int = 0) -> None:
        if kind == "sw":
            text  = "ON"  if val else "OFF"
            color = QColor("#c0392b") if val else QColor("#4a4a4a")
        else:
            text  = "ON"  if val else "OFF"
            color = QColor("#c8861a") if val else QColor("#4a7a4a")

        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        item.setBackground(QBrush(color))
        item.setForeground(QBrush(QColor("#ffffff")))
        # UserRole encodes (kind, state_row, sw_i, cap_j)
        item.setData(Qt.UserRole, (kind, row, sw_i, cap_j))
        self._table.setItem(row, col, item)

    def _on_cell_click(self, row: int, col: int) -> None:
        item = self._table.item(row, col)
        if item is None:
            return
        data = item.data(Qt.UserRole)
        if not data:
            return
        kind, state_idx, sw_i, cap_j = data
        if kind == "sw":
            state = self.model.state_list
            if (state_idx < len(state) and sw_i < len(state[state_idx])
                    and cap_j < len(state[state_idx][sw_i])):
                state[state_idx][sw_i][cap_j] ^= 1
        elif kind == "ind" and self.model.inductor_switch_states is not None:
            ind_state = self.model.inductor_switch_states
            # cap_j reused as ind sub-index for IND cells
            if state_idx < len(ind_state) and cap_j < len(ind_state[state_idx]):
                ind_state[state_idx][cap_j] ^= 1
        self.refresh()
        self.changed.emit()

    def _on_add(self) -> None:
        self.model.add_state()
        self.refresh()
        self.changed.emit()

    def _on_remove(self) -> None:
        selected = self._table.selectedItems()
        idx = self._table.row(selected[0]) if selected else len(self.model.state_list) - 1
        if idx >= 0:
            self.model.remove_state(idx)
            self.refresh()
            self.changed.emit()

    def _on_ind_toggle(self, state: int) -> None:
        if self._loading:
            return
        if state == Qt.Checked:
            self.model.enable_inductor_switch_states()
        else:
            self.model.disable_inductor_switch_states()
        self.refresh()
        self.changed.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Settings Dialog
# ─────────────────────────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """
    Five-tab settings dialog.

    Tab 1 – Execution      : conda env, python path, simulate_design.py path
    Tab 2 – Engine Paths   : virtuoso_dir, EMX binary, proc file, tsmc models
    Tab 3 – Inductor Files : 4 × s1p paths for the inductor translation table
    Tab 4 – Layout         : pixel_size, grid_size, layers, feedline geometry
    Tab 5 – Switch Model   : cap_value, r_on, r_off, c_off

    "Execution" settings are written to gui_settings.json.
    All other settings update model.full_config in-place (saved with the YAML).
    """

    def __init__(self, gui_settings: GuiSettings, model: "DesignModel",
                 parent=None) -> None:
        super().__init__(parent)
        self.gui_settings = gui_settings
        self.model        = model
        self.setWindowTitle("Settings")
        self.resize(680, 540)
        self._build_ui()
        self._load_values()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs)

        tabs.addTab(self._tab_execution(),   "Execution")
        tabs.addTab(self._tab_engine_paths(), "Engine Paths")
        tabs.addTab(self._tab_inductors(),   "Inductor Files")
        tabs.addTab(self._tab_layout(),      "Layout")
        tabs.addTab(self._tab_switch(),      "Switch Model")

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ── Tab builders ─────────────────────────────────────────────────────────

    def _tab_execution(self) -> QWidget:
        w   = QWidget()
        fl  = QFormLayout(w)
        fl.setRowWrapPolicy(QFormLayout.WrapAllRows)

        self._use_conda_cb    = QCheckBox("Use conda to run simulation")
        self._conda_env_edit  = QLineEdit()
        self._conda_exe_edit  = QLineEdit()
        self._sim_script_edit = QLineEdit()
        self._py_exe_edit     = QLineEdit()

        fl.addRow(self._use_conda_cb)
        fl.addRow("Conda env name:",        self._conda_env_edit)
        fl.addRow("Conda executable:",      self._conda_exe_edit)
        fl.addRow("simulate_design.py:",    self._make_browse_row(
                                                self._sim_script_edit, "file"))
        fl.addRow(QLabel(
            "  Leave simulate_design.py blank to auto-detect next to this script."))
        fl.addRow("Python exe (no-conda):", self._make_browse_row(
                                                self._py_exe_edit, "file"))
        fl.addRow(QLabel(
            "  Python exe is used only when conda is disabled."))

        self._use_conda_cb.stateChanged.connect(self._on_conda_toggle)
        return w

    def _tab_engine_paths(self) -> QWidget:
        w  = QWidget()
        fl = QFormLayout(w)
        fl.setRowWrapPolicy(QFormLayout.WrapAllRows)

        # (label, config_key, browse_type)
        fields = [
            ("Virtuoso dir:",        "project.virtuoso_dir",       "dir"),
            ("Parent output dir:",   "project.parent_output_dir",  "dir"),
            ("Virtuoso display:",    "project.display",            None),
            ("EMX binary:",          "paths.emx_binary",           "file"),
            ("Proc file (.proc):",   "paths.proc_file",            "file"),
            ("TSMC models (.scs):",  "paths.tsmc_models",          "file"),
        ]
        self._engine_fields: Dict[str, QLineEdit] = {}
        for label, key, btype in fields:
            edit = QLineEdit()
            self._engine_fields[key] = edit
            row = self._make_browse_row(edit, btype) if btype else edit
            fl.addRow(label, row)
        return w

    def _tab_inductors(self) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)

        vl.addWidget(QLabel(
            "ON-state inductor s1p files — one entry per inductor value "
            "(index 0 = 300 pH layout-extracted EM):"))
        fl_on = QFormLayout()
        vl.addLayout(fl_on)
        self._ind_edits: List[QLineEdit] = []
        for lbl in ["0 — 300 pH (ON):"]:
            edit = QLineEdit()
            self._ind_edits.append(edit)
            fl_on.addRow(lbl, self._make_browse_row(edit, "file"))

        vl.addSpacing(12)
        vl.addWidget(QLabel(
            "OFF-state inductor s1p files (optional) — leave blank to use "
            "large-resistor open-circuit model for that inductor value:"))
        fl_off = QFormLayout()
        vl.addLayout(fl_off)
        self._ind_off_edits: List[QLineEdit] = []
        for lbl in ["0 — 300 pH (OFF):"]:
            edit = QLineEdit()
            self._ind_off_edits.append(edit)
            fl_off.addRow(lbl, self._make_browse_row(edit, "file"))

        vl.addStretch()
        return w

    def _tab_layout(self) -> QWidget:
        w  = QWidget()
        fl = QFormLayout(w)

        self._pixel_size_spin  = QDoubleSpinBox()
        self._grid_size_spin   = QDoubleSpinBox()
        self._gnd_layer_edit   = QLineEdit()
        self._pix_layer_edit   = QLineEdit()
        self._fl_width_spin    = QDoubleSpinBox()
        self._fl_padding_spin  = QDoubleSpinBox()

        for spin in (self._pixel_size_spin, self._grid_size_spin,
                     self._fl_width_spin, self._fl_padding_spin):
            spin.setRange(0.1, 1000.0)
            spin.setDecimals(2)
            spin.setSuffix(" µm")

        fl.addRow("Pixel size:",       self._pixel_size_spin)
        fl.addRow("Grid size (pitch):", self._grid_size_spin)
        fl.addRow("GND layer name:",   self._gnd_layer_edit)
        fl.addRow("Pixel layer name:", self._pix_layer_edit)
        fl.addRow("Feedline width:",   self._fl_width_spin)
        fl.addRow("Feedline padding:", self._fl_padding_spin)
        return w

    def _tab_switch(self) -> QWidget:
        w  = QWidget()
        fl = QFormLayout(w)

        self._sw_cap_spin     = QDoubleSpinBox()
        self._sw_on_s1p_edit  = QLineEdit()
        self._sw_off_s1p_edit = QLineEdit()
        self._sw_mimflag_combo = QComboBox()

        self._sw_cap_spin.setRange(1e-6, 1000.0)
        self._sw_cap_spin.setDecimals(4)
        self._sw_cap_spin.setSuffix(" pF")

        self._sw_on_s1p_edit.setPlaceholderText("Path to switch ON  s1p file…")
        self._sw_off_s1p_edit.setPlaceholderText("Path to switch OFF s1p file…")

        self._sw_mimflag_combo.addItems([
            "1 — 1.0 fF/μm²",
            "2 — 1.5 fF/μm²",
            "3 — 2.0 fF/μm²",
        ])

        fl.addRow("Series cap (cap_value):", self._sw_cap_spin)
        fl.addRow("Switch ON s1p:",          self._make_browse_row(self._sw_on_s1p_edit,  "file"))
        fl.addRow("Switch OFF s1p:",         self._make_browse_row(self._sw_off_s1p_edit, "file"))
        fl.addRow("MIM cap density (mimflag):", self._sw_mimflag_combo)

        self._cleanup_cb = QCheckBox("Delete intermediate files after simulation")
        fl.addRow("Cleanup:", self._cleanup_cb)

        fl.addRow(QLabel(
            "cap_value is the global default. Per-switch cap is set in the "
            "Pixel Properties panel.\n"
            "Switch NMOS behavior is modeled via ON/OFF s1p files (TSMC mimcap_um_sin_rf model)."))
        return w

    # ── Browse-row helper ─────────────────────────────────────────────────────

    def _make_browse_row(self, edit: QLineEdit,
                         browse_type: Optional[str]) -> QWidget:
        """Wrap an edit in a row widget with an optional Browse button."""
        if browse_type is None:
            return edit
        w   = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        btn = QPushButton("Browse…")
        btn.setMaximumWidth(72)
        if browse_type == "file":
            btn.clicked.connect(lambda _, e=edit: self._browse_file(e))
        else:
            btn.clicked.connect(lambda _, e=edit: self._browse_dir(e))
        row.addWidget(btn)
        return w

    def _browse_file(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file",
                                              edit.text() or "")
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: QLineEdit) -> None:
        from PyQt5.QtWidgets import QFileDialog as QFD
        path = QFD.getExistingDirectory(self, "Select directory",
                                        edit.text() or "")
        if path:
            edit.setText(path)

    # ── Load / Save ───────────────────────────────────────────────────────────

    def _load_values(self) -> None:
        gs = self.gui_settings
        self._use_conda_cb.setChecked(bool(gs["use_conda"]))
        self._conda_env_edit.setText(str(gs["conda_env"]))
        self._conda_exe_edit.setText(str(gs["conda_executable"]))
        self._sim_script_edit.setText(str(gs["simulate_script"]))
        self._py_exe_edit.setText(str(gs["python_executable"]))
        self._on_conda_toggle()

        cfg = self.model.full_config
        for key, edit in self._engine_fields.items():
            edit.setText(str(_get_nested(cfg, key, "")))

        tbl = _get_nested(cfg, "pixelated.inductor_translation_table", [])
        for i, edit in enumerate(self._ind_edits):
            edit.setText(str(tbl[i]) if i < len(tbl) else "")

        off_tbl = _get_nested(cfg, "pixelated.inductor_off_translation_table", [])
        for i, edit in enumerate(self._ind_off_edits):
            edit.setText(str(off_tbl[i]) if i < len(off_tbl) else "")

        self._pixel_size_spin.setValue(
            float(_get_nested(cfg, "layout.pixel_size", 25.0)))
        self._grid_size_spin.setValue(
            float(_get_nested(cfg, "layout.grid_size",  20.0)))
        self._gnd_layer_edit.setText(
            str(_get_nested(cfg, "layout.gnd_layer",    "M4")))
        self._pix_layer_edit.setText(
            str(_get_nested(cfg, "layout.pixel_layer",  "AP")))
        self._fl_width_spin.setValue(
            float(_get_nested(cfg, "pixelated.feedline.feedline_width",   10.2)))
        self._fl_padding_spin.setValue(
            float(_get_nested(cfg, "pixelated.feedline.feedline_padding", 10.0)))

        self._sw_cap_spin.setValue(
            float(_get_nested(cfg, "switch_model.cap_value", 1e-12)) * 1e12)
        self._sw_on_s1p_edit.setText(
            str(_get_nested(cfg, "switch_model.switch_on_s1p", "")))
        self._sw_off_s1p_edit.setText(
            str(_get_nested(cfg, "switch_model.switch_off_s1p", "")))
        mimflag = int(_get_nested(cfg, "mim_cap.mimflag", 1))
        self._sw_mimflag_combo.setCurrentIndex(max(0, min(mimflag - 1, 2)))
        self._cleanup_cb.setChecked(
            bool(_get_nested(cfg, "cleanup.delete_intermediate_files", True)))

    def _save_and_accept(self) -> None:
        gs = self.gui_settings
        gs["use_conda"]         = self._use_conda_cb.isChecked()
        gs["conda_env"]         = self._conda_env_edit.text().strip()
        gs["conda_executable"]  = self._conda_exe_edit.text().strip() or "conda"
        gs["simulate_script"]   = self._sim_script_edit.text().strip()
        gs["python_executable"] = self._py_exe_edit.text().strip()
        gs.save()

        cfg = self.model.full_config
        for key, edit in self._engine_fields.items():
            v = edit.text().strip()
            if v:
                _set_nested(cfg, key, v)

        tbl = [e.text().strip() for e in self._ind_edits if e.text().strip()]
        if tbl:
            _set_nested(cfg, "pixelated.inductor_translation_table", tbl)

        off_tbl = [e.text().strip() for e in self._ind_off_edits]
        if any(off_tbl):
            _set_nested(cfg, "pixelated.inductor_off_translation_table", off_tbl)
        else:
            cfg.get("pixelated", {}).pop("inductor_off_translation_table", None)

        _set_nested(cfg, "layout.pixel_size",   self._pixel_size_spin.value())
        _set_nested(cfg, "layout.grid_size",    self._grid_size_spin.value())
        v = self._gnd_layer_edit.text().strip()
        if v: _set_nested(cfg, "layout.gnd_layer",   v)
        v = self._pix_layer_edit.text().strip()
        if v: _set_nested(cfg, "layout.pixel_layer", v)
        _set_nested(cfg, "pixelated.feedline.feedline_width",
                    self._fl_width_spin.value())
        _set_nested(cfg, "pixelated.feedline.feedline_padding",
                    self._fl_padding_spin.value())
        _set_nested(cfg, "switch_model.cap_value",
                    self._sw_cap_spin.value() * 1e-12)
        v = self._sw_on_s1p_edit.text().strip()
        if v:
            _set_nested(cfg, "switch_model.switch_on_s1p", v)
        v = self._sw_off_s1p_edit.text().strip()
        if v:
            _set_nested(cfg, "switch_model.switch_off_s1p", v)
        _set_nested(cfg, "mim_cap.mimflag", self._sw_mimflag_combo.currentIndex() + 1)
        _set_nested(cfg, "cleanup.delete_intermediate_files", self._cleanup_cb.isChecked())

        self.accept()

    def _on_conda_toggle(self) -> None:
        use = self._use_conda_cb.isChecked()
        self._conda_env_edit.setEnabled(use)
        self._conda_exe_edit.setEnabled(use)
        self._py_exe_edit.setEnabled(not use)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Window
# ─────────────────────────────────────────────────────────────────────────────

class SimulationWindow(QDialog):
    """
    Modeless dialog that saves a timestamped config YAML and launches
    simulate_design.py as a subprocess, streaming its output live.

    Layout
    ------
    • Top info bar  : config saved to, output dir
    • Command label : full command that will run
    • Log pane      : scrolling plain-text live output
    • Bottom bar    : status label  |  Run  |  Cancel  |  Close
    """

    def __init__(self, model: "DesignModel", gui_settings: GuiSettings,
                 parent=None) -> None:
        super().__init__(parent)
        self.model        = model
        self.gui_settings = gui_settings
        self._process:    Optional[QProcess] = None
        self._sim_yaml:   str = ""
        self._output_dir: str = ""

        self.setWindowTitle("Run Simulation")
        self.setMinimumSize(800, 560)
        self.resize(900, 620)
        self._build_ui()
        self._prepare()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Info section ──────────────────────────────────────────────────
        info_gb = QGroupBox("Simulation target")
        info_fl = QFormLayout(info_gb)
        self._cfg_label  = QLabel()
        self._odir_label = QLabel()
        self._cmd_label  = QLabel()
        self._cmd_label.setWordWrap(True)
        self._cmd_label.setStyleSheet("font-family: monospace; color: #aee3f5;")
        info_fl.addRow("Config YAML:",  self._cfg_label)
        info_fl.addRow("Output dir:",   self._odir_label)
        info_fl.addRow("Command:",      self._cmd_label)
        root.addWidget(info_gb)

        # ── Design summary ────────────────────────────────────────────────
        summ_gb  = QGroupBox("Design summary")
        summ_lay = QHBoxLayout(summ_gb)
        self._summ_label = QLabel()
        self._summ_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self._summ_label.setWordWrap(True)
        summ_lay.addWidget(self._summ_label)
        root.addWidget(summ_gb)

        # ── Log ───────────────────────────────────────────────────────────
        log_gb  = QGroupBox("Output log")
        log_lay = QVBoxLayout(log_gb)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._log.setStyleSheet(
            "QPlainTextEdit { background:#111; color:#d4d4d4; "
            "font-family: monospace; font-size: 11px; }")
        log_lay.addWidget(self._log)
        root.addWidget(log_gb, stretch=1)

        # ── Bottom bar ────────────────────────────────────────────────────
        bot = QHBoxLayout()
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("font-weight: bold;")
        self._run_btn    = QPushButton("▶  Run Simulation")
        self._run_btn.setStyleSheet(
            "QPushButton { background:#1a6496; color:white; padding:5px 18px; "
            "font-weight:bold; border-radius:4px; }")
        self._cancel_btn = QPushButton("■  Cancel")
        self._cancel_btn.setEnabled(False)
        self._close_btn  = QPushButton("Close")

        self._run_btn.clicked.connect(self._run)
        self._cancel_btn.clicked.connect(self._cancel)
        self._close_btn.clicked.connect(self.close)

        bot.addWidget(self._status_lbl, stretch=1)
        bot.addWidget(self._run_btn)
        bot.addWidget(self._cancel_btn)
        bot.addWidget(self._close_btn)
        root.addLayout(bot)

    # ── Preparation ───────────────────────────────────────────────────────────

    def _prepare(self) -> None:
        """Compute output paths and populate the info section."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.model.file_path:
            base = Path(self.model.file_path).parent
        else:
            base = Path.cwd()

        self._output_dir = str(base / f"sim_{ts}")
        self._sim_yaml   = str(base / f"sim_{ts}_config.yaml")

        cmd = self.gui_settings.build_command(self._sim_yaml, self._output_dir)

        self._cfg_label.setText(self._sim_yaml)
        self._odir_label.setText(self._output_dir)
        self._cmd_label.setText(" ".join(cmd))

        # Design summary
        m   = self.model
        sw  = m.sw_pixels()
        ind = m.ind_pixels()
        lines = [
            f"Grid : {m.num_rows()} × {m.num_cols()} pixels",
            f"Switches  : {len(sw)}   → {len(m.state_list)} simulation states",
            f"Inductors : {len(ind)}",
        ]
        if ind:
            ind_vals = [INDUCTOR_LABELS[m.inductor_values[i]]
                        if i < len(m.inductor_values) else "?"
                        for _, _, i in ind]
            lines.append("Ind values: " + ", ".join(ind_vals))
        if m.cap_pixel_pairs:
            lines.append(f"Cap pairs : {len(m.cap_pixel_pairs)}")
        self._summ_label.setText("\n".join(lines))

        if not self.model.file_path:
            self._log_line(
                "⚠  No file loaded — save the design first so the output "
                "folder is placed next to your YAML.", error=True)

    # ── Simulation ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if not self.model.file_path:
            QMessageBox.warning(self, "Not saved",
                                "Please save the design (Ctrl+S) before running a simulation.")
            return

        # --- Write the simulation config YAML ---
        self.model._push_to_config()          # make sure pixelated section is current
        sim_cfg = copy.deepcopy(self.model.full_config)
        # Preserve the design's own mode (e.g. differential_filter); only fall
        # back to the simple pixel mode when the config doesn't specify one.
        sim_cfg.setdefault("mode", "pixelated_design_simple")
        sim_cfg.setdefault("project", {})["parent_output_dir"] = self._output_dir

        try:
            Path(self._sim_yaml).parent.mkdir(parents=True, exist_ok=True)
            with open(self._sim_yaml, "w") as f:
                yaml.dump(sim_cfg, f, default_flow_style=False,
                          sort_keys=False, allow_unicode=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Cannot write config:\n{exc}")
            return

        cmd = self.gui_settings.build_command(self._sim_yaml, self._output_dir)
        self._log.clear()
        self._log_line(f"Config  : {self._sim_yaml}")
        self._log_line(f"Output  : {self._output_dir}")
        self._log_line(f"Command : {' '.join(cmd)}")
        self._log_line("─" * 72)

        # --- Launch QProcess ---
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_proc_error)

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status_lbl.setText("● Running…")
        self._status_lbl.setStyleSheet("color:#f39c12; font-weight:bold;")

        self._process.start(cmd[0], cmd[1:])

        if not self._process.waitForStarted(3000):
            self._log_line(
                f"ERROR: Failed to start process: {cmd[0]}", error=True)
            self._reset_buttons()

    def _on_stdout(self) -> None:
        data = (self._process.readAllStandardOutput()
                .data().decode("utf-8", errors="replace"))
        for line in data.splitlines():
            self._log_line(line)

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        # Drain any remaining output
        remaining = (self._process.readAllStandardOutput()
                     .data().decode("utf-8", errors="replace"))
        for line in remaining.splitlines():
            self._log_line(line)

        self._log_line("─" * 72)
        if exit_code == 0:
            self._log_line("✓  Simulation completed successfully.")
            self._status_lbl.setText("✓  Completed")
            self._status_lbl.setStyleSheet("color:#2ecc71; font-weight:bold;")
            self._show_results()
        else:
            self._log_line(f"✗  Process exited with code {exit_code}.", error=True)
            self._status_lbl.setText(f"✗  Failed (code {exit_code})")
            self._status_lbl.setStyleSheet("color:#e74c3c; font-weight:bold;")
        self._reset_buttons()

    def _show_results(self) -> None:
        """Open the results dialog; if no plots exist yet wait 2 s and retry."""
        plots = _find_plots(self._output_dir)
        if not plots:
            # Give the engine a moment to flush files, then open anyway
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(2000, self._show_results_now)
            self._log_line("  (waiting for plot files to appear…)")
        else:
            self._show_results_now()

    def _show_results_now(self) -> None:
        dlg = ResultsDialog(self._output_dir, self)
        dlg.show()

    def _on_proc_error(self, error) -> None:
        msgs = {
            QProcess.FailedToStart: "Failed to start — check command / conda env.",
            QProcess.Crashed:       "Process crashed.",
            QProcess.Timedout:      "Process timed out.",
            QProcess.WriteError:    "Write error.",
            QProcess.ReadError:     "Read error.",
        }
        self._log_line(f"Process error: {msgs.get(error, str(error))}", error=True)
        self._status_lbl.setText("✗  Error")
        self._status_lbl.setStyleSheet("color:#e74c3c; font-weight:bold;")
        self._reset_buttons()

    def _cancel(self) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._log_line("⚠  Cancelled by user.", error=True)
            self._status_lbl.setText("Cancelled")
            self._status_lbl.setStyleSheet("color:#e67e22; font-weight:bold;")
        self._reset_buttons()

    def _reset_buttons(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def _log_line(self, text: str, error: bool = False) -> None:
        if error:
            self._log.appendHtml(
                f'<span style="color:#ff6b6b;">{text}</span>')
        else:
            self._log.appendPlainText(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, ev) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            reply = QMessageBox.question(
                self, "Simulation running",
                "A simulation is still running. Kill it and close?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                ev.ignore()
                return
            self._process.kill()
        super().closeEvent(ev)


# ─────────────────────────────────────────────────────────────────────────────
# Results Dialog  (shown after a successful simulation)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical plot filenames produced by virtuoso_engine/plotter.py
_PLOT_FILES = [
    ("s21_magnitude.png",  "S21 — Transmission"),
    ("s11_magnitude.png",  "S11 — Input Match"),
    ("inductor_lq.png",    "Inductor L & Q"),
    ("design_matrix.png",  "Design Matrix"),
]


def _find_plots(output_dir: str) -> List[Tuple[str, str]]:
    """
    Walk *output_dir* recursively and return (tab_label, abs_path) for every
    recognised plot PNG, in the canonical order defined by _PLOT_FILES.
    """
    import os
    name_to_label = {fn: lbl for fn, lbl in _PLOT_FILES}
    found: Dict[str, str] = {}          # label → path
    for dirpath, _, filenames in os.walk(output_dir):
        for fn in filenames:
            if fn in name_to_label and name_to_label[fn] not in found:
                found[name_to_label[fn]] = os.path.join(dirpath, fn)

    # Return in canonical order
    return [(lbl, found[lbl]) for _, lbl in _PLOT_FILES if lbl in found]


class _ImageViewer(QWidget):
    """
    Single-image viewer with fit-to-window / manual zoom.
    Ctrl+scroll and +/- buttons adjust zoom; 'Fit' restores auto-scale.
    """

    def __init__(self, pixmap: "QPixmap", parent=None) -> None:
        super().__init__(parent)
        self._orig    = pixmap
        self._fit     = True
        self._scale   = 1.0

        vl  = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)

        # Zoom toolbar
        ztb = QHBoxLayout()
        for label, fn in [("＋", self._zi), ("－", self._zo),
                           ("Fit", self._fit_win), ("1:1", self._one2one)]:
            b = QPushButton(label)
            b.setMaximumWidth(48)
            b.clicked.connect(fn)
            ztb.addWidget(b)
        self._zoom_lbl = QLabel("fit")
        self._zoom_lbl.setStyleSheet("color:#888; font-size:10px;")
        ztb.addWidget(self._zoom_lbl)
        ztb.addStretch()
        vl.addLayout(ztb)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignCenter)
        self._scroll.setWidget(self._lbl)
        vl.addWidget(self._scroll, stretch=1)

        self._render()

    def resizeEvent(self, ev) -> None:
        if self._fit:
            self._render()
        super().resizeEvent(ev)

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.ControlModifier:
            if ev.angleDelta().y() > 0:
                self._zi()
            else:
                self._zo()
            ev.accept()
        else:
            ev.ignore()

    def _zi(self)       -> None: self._set_scale(self._scale * 1.25)
    def _zo(self)       -> None: self._set_scale(self._scale / 1.25)
    def _one2one(self)  -> None: self._set_scale(1.0)

    def _fit_win(self) -> None:
        self._fit = True
        self._render()

    def _set_scale(self, s: float) -> None:
        self._fit   = False
        self._scale = max(0.05, min(s, 10.0))
        self._render()

    def _render(self) -> None:
        if self._fit:
            vp = self._scroll.viewport()
            w, h = max(vp.width() - 4, 10), max(vp.height() - 4, 10)
            pm   = self._orig.scaled(w, h, Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation)
            self._scale = pm.width() / max(self._orig.width(), 1)
            self._zoom_lbl.setText("fit")
        else:
            w = int(self._orig.width()  * self._scale)
            h = int(self._orig.height() * self._scale)
            pm = self._orig.scaled(w, h, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
            self._zoom_lbl.setText(f"{self._scale*100:.0f}%")

        self._lbl.setPixmap(pm)
        self._lbl.resize(pm.size())


class ResultsDialog(QDialog):
    """
    Tabbed dialog that shows S-parameter and design-matrix plots generated
    by the simulation engine.  Opens automatically after a successful run.
    """

    def __init__(self, output_dir: str, parent=None) -> None:
        super().__init__(parent)
        self._output_dir = output_dir
        self.setWindowTitle("Simulation Results")
        self.resize(1000, 680)
        self._build_ui()
        self._load_plots()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(4)

        # Header
        hdr = QHBoxLayout()
        self._dir_lbl = QLabel()
        self._dir_lbl.setStyleSheet("color:#888; font-size:10px; font-family:monospace;")
        self._dir_lbl.setWordWrap(True)
        hdr.addWidget(self._dir_lbl, stretch=1)

        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(self._open_folder)
        hdr.addWidget(open_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Re-scan output folder for new plots")
        refresh_btn.clicked.connect(self._load_plots)
        hdr.addWidget(refresh_btn)
        root.addLayout(hdr)

        # Tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        # Status
        self._status = QLabel()
        self._status.setStyleSheet("color:#888; font-size:10px;")
        root.addWidget(self._status)

        # Close button
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

    def _load_plots(self) -> None:
        self._tabs.clear()
        self._dir_lbl.setText(f"Output: {self._output_dir}")

        plots = _find_plots(self._output_dir)
        if not plots:
            placeholder = QLabel(
                "No plot images found yet.\n"
                "The engine may still be writing files — click Refresh.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color:#888; font-size:13px;")
            self._tabs.addTab(placeholder, "No plots")
            self._status.setText(f"Searched: {self._output_dir}")
            return

        for label, path in plots:
            pm = QPixmap(path)
            if pm.isNull():
                tab = QLabel(f"Could not load:\n{path}")
                tab.setAlignment(Qt.AlignCenter)
            else:
                tab = _ImageViewer(pm)
            self._tabs.addTab(tab, label)

        self._status.setText(
            f"Found {len(plots)} plot(s) in {self._output_dir}")

    def _open_folder(self) -> None:
        import subprocess as _sp, platform
        path = self._output_dir
        if platform.system() == "Darwin":
            _sp.Popen(["open", path])
        elif platform.system() == "Windows":
            _sp.Popen(["explorer", path])
        else:
            _sp.Popen(["xdg-open", path])


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.model        = DesignModel.default()
        self.gui_settings = GuiSettings()          # load from gui_settings.json
        self.setWindowTitle("Pixel Filter Designer")
        self.resize(1300, 820)
        self._build_ui()
        self._apply_theme()

        if initial_path and Path(initial_path).exists():
            self._load_file(initial_path)
        else:
            self._full_refresh()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Toolbar ──────────────────────────────────────────────────────────
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setStyleSheet("QToolBar { spacing: 2px; }")
        self.addToolBar(tb)

        def act(label, shortcut, fn):
            a = QAction(label, self)
            a.setShortcut(shortcut)
            a.triggered.connect(fn)
            tb.addAction(a)
            return a

        act("Open",     "Ctrl+O",       self._on_open)
        act("Save",     "Ctrl+S",       self._on_save)
        act("Save As…", "Ctrl+Shift+S", self._on_save_as)

        tb.addSeparator()
        run_btn = QPushButton("▶  Run Sim")
        run_btn.setToolTip("Save a timestamped config and run the simulation engine")
        run_btn.setStyleSheet(
            "QPushButton { background:#1a6496; color:white; font-weight:bold; "
            "padding:3px 10px; border-radius:3px; }"
            "QPushButton:hover { background:#1e78b4; }")
        run_btn.clicked.connect(self._on_run_sim)
        tb.addWidget(run_btn)

        settings_btn = QPushButton("⚙  Settings")
        settings_btn.setToolTip("Configure conda env, paths, and simulation parameters")
        settings_btn.clicked.connect(self._on_settings)
        tb.addWidget(settings_btn)

        tb.addSeparator()

        tb.addWidget(QLabel(" Mode: "))
        self._sel_btn = QPushButton("Selection [S]")
        self._sel_btn.setCheckable(True); self._sel_btn.setChecked(True)
        self._sel_btn.setShortcut("S")
        self._sel_btn.clicked.connect(lambda: self._set_mode(MODE_SELECT))
        tb.addWidget(self._sel_btn)

        self._brush_btn = QPushButton("Brush [B]")
        self._brush_btn.setCheckable(True)
        self._brush_btn.setShortcut("B")
        self._brush_btn.clicked.connect(lambda: self._set_mode(MODE_BRUSH))
        tb.addWidget(self._brush_btn)

        tb.addSeparator()
        tb.addWidget(QLabel(" Brush: "))
        self._brush_combo = QComboBox()
        self._brush_combo.addItem("0 — GND", 0)
        self._brush_combo.addItem("1 — MTL", 1)
        self._brush_combo.currentIndexChanged.connect(self._on_brush_change)
        tb.addWidget(self._brush_combo)

        tb.addSeparator()
        tb.addWidget(QLabel(" Zoom: "))
        for label, fn, tip in [
            ("＋",   self._zoom_in,    "Zoom in  (Ctrl+scroll)"),
            ("－",   self._zoom_out,   "Zoom out (Ctrl+scroll)"),
            ("Fit",  self._zoom_fit,   "Fit grid to window"),
            ("1:1",  self._zoom_reset, "Reset to default size"),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(fn)
            btn.setMaximumWidth(38 if len(label) <= 2 else 48)
            tb.addWidget(btn)

        tb.addSeparator()
        tb.addWidget(QLabel(" Grid: "))
        for label, fn in [
            ("+Row",  self._add_row),
            ("–Row",  self._del_row),
            ("+Col",  self._add_col),
            ("–Col",  self._del_col),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(fn)
            tb.addWidget(btn)

        tb.addSeparator()
        self._path_label = QLabel("  (default design — not saved)")
        self._path_label.setStyleSheet("color: #888;")
        tb.addWidget(self._path_label)

        # ── Central area ─────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(4, 4, 4, 4)
        main_lay.setSpacing(4)

        # Horizontal split: grid | states table
        h_split = QSplitter(Qt.Horizontal)

        # Left: scrollable pixel grid
        self._grid = PixelGridWidget(self.model)
        self._grid.pixel_clicked.connect(self._on_pixel_click)
        self._grid.pixel_painted.connect(self._on_pixel_paint)
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._grid)
        self._scroll.setWidgetResizable(True)   # stretch grid to fill viewport
        self._scroll.setMinimumWidth(380)
        h_split.addWidget(self._scroll)

        # Right: states table
        self._states_table = StatesTableWidget(self.model)
        self._states_table.changed.connect(self._on_data_changed)
        self._states_table.setMinimumWidth(260)
        h_split.addWidget(self._states_table)

        h_split.setSizes([780, 320])
        main_lay.addWidget(h_split, stretch=1)

        # Bottom: property panel
        self._prop = PropertyPanel(self.model)
        self._prop.changed.connect(self._on_data_changed)
        self._prop.setMaximumHeight(155)
        self._prop.setMinimumHeight(120)
        main_lay.addWidget(self._prop)

        self.statusBar().showMessage("Ready — open a config.yaml or start designing.")

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget   { background:#1e1e1e; color:#dcdcdc; }
            QToolBar               { background:#252525; border-bottom:1px solid #3a3a3a;
                                     padding:3px; spacing:3px; }
            QPushButton            { background:#353535; color:#dcdcdc;
                                     border:1px solid #505050; border-radius:3px;
                                     padding:3px 8px; }
            QPushButton:hover      { background:#454545; }
            QPushButton:checked    { background:#1a6496; border-color:#1a6496; }
            QComboBox              { background:#353535; color:#dcdcdc;
                                     border:1px solid #505050; border-radius:3px;
                                     padding:2px 6px; }
            QComboBox QAbstractItemView { background:#2d2d2d; color:#dcdcdc; }
            QGroupBox              { border:1px solid #404040; border-radius:4px;
                                     margin-top:10px; padding-top:4px; font-weight:bold; }
            QGroupBox::title       { subcontrol-position:top left; padding:0 4px; }
            QDoubleSpinBox, QSpinBox { background:#353535; color:#dcdcdc;
                                       border:1px solid #505050; border-radius:3px; }
            QTableWidget           { background:#252525; color:#dcdcdc;
                                     gridline-color:#3a3a3a; border:1px solid #3a3a3a; }
            QHeaderView::section   { background:#2d2d2d; color:#aaaaaa;
                                     border:1px solid #3a3a3a; padding:4px; }
            QScrollArea            { border:none; }
            QSplitter::handle      { background:#3a3a3a; }
            QStatusBar             { background:#252525; color:#888; }
            QCheckBox              { color:#dcdcdc; }
            QLabel                 { color:#dcdcdc; }
            QDialog                { background:#1e1e1e; }
            QLineEdit              { background:#353535; color:#dcdcdc;
                                     border:1px solid #505050; border-radius:3px;
                                     padding:2px 6px; }
            QPlainTextEdit         { background:#1a1a1a; color:#d4d4d4;
                                     border:1px solid #3a3a3a; border-radius:3px; }
            QTabWidget::pane       { border:1px solid #404040; }
            QTabBar::tab           { background:#2d2d2d; color:#aaaaaa;
                                     padding:5px 14px; border:1px solid #404040; }
            QTabBar::tab:selected  { background:#1e1e1e; color:#dcdcdc;
                                     border-bottom:2px solid #1a6496; }
        """)

    # ── Load / Save ───────────────────────────────────────────────────────────

    def _load_file(self, path: str) -> None:
        try:
            self.model.load(path)
            name = Path(path).name
            self._path_label.setText(f"  {name}")
            self.setWindowTitle(f"Pixel Filter Designer — {name}")
            self._full_refresh()
            self.statusBar().showMessage(f"Loaded: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Cannot load file:\n{exc}")

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open config.yaml", "",
            "YAML files (*.yaml *.yml);;All files (*)")
        if path:
            self._load_file(path)

    def _on_save(self) -> None:
        if not self.model.file_path:
            self._on_save_as(); return
        try:
            self.model.save()
            self.statusBar().showMessage(f"Saved: {self.model.file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", f"Cannot save:\n{exc}")

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "",
            "YAML files (*.yaml *.yml);;All files (*)")
        if not path:
            return
        try:
            self.model.save(path)
            name = Path(path).name
            self._path_label.setText(f"  {name}")
            self.setWindowTitle(f"Pixel Filter Designer — {name}")
            self.statusBar().showMessage(f"Saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", f"Cannot save:\n{exc}")

    # ── Zoom helpers ──────────────────────────────────────────────────────────

    def _zoom_in(self) -> None:
        # Switch to manual mode so the scroll area shows scrollbars at larger size
        self._scroll.setWidgetResizable(False)
        self._grid.zoom_in()

    def _zoom_out(self) -> None:
        self._scroll.setWidgetResizable(False)
        self._grid.zoom_out()

    def _zoom_fit(self) -> None:
        # Back to fit mode: widget fills viewport, cell_size computed automatically
        self._grid.enter_fit_mode()
        self._scroll.setWidgetResizable(True)

    def _zoom_reset(self) -> None:
        self._scroll.setWidgetResizable(False)
        self._grid.zoom_reset()

    # ── Mode / brush ──────────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        self._grid.set_mode(mode)
        self._sel_btn.setChecked(mode == MODE_SELECT)
        self._brush_btn.setChecked(mode == MODE_BRUSH)
        self._brush_combo.setEnabled(mode == MODE_BRUSH)
        if mode == MODE_SELECT:
            self.statusBar().showMessage("Selection mode — click a pixel to inspect it.")
        else:
            t = self._brush_combo.currentData()
            self.statusBar().showMessage(
                f"Brush mode — painting type {t} ({PIXEL_INFO[t][0]}). "
                "Click/drag on the grid.")

    def _on_brush_change(self, _idx: int) -> None:
        t = self._brush_combo.currentData()
        self._grid.set_brush_type(t)

    # ── Grid operations ───────────────────────────────────────────────────────

    def _add_row(self) -> None:
        nr = self.model.num_rows()
        idx, ok = QInputDialog.getInt(self, "Insert Row",
                                      f"Insert before row (0 = top, {nr} = bottom):",
                                      nr, 0, nr)
        if ok:
            self.model.add_row(idx)
            self._full_refresh()

    def _add_col(self) -> None:
        nc = self.model.num_cols()
        idx, ok = QInputDialog.getInt(self, "Insert Column",
                                      f"Insert before column (0 = left, {nc} = right):",
                                      nc, 0, nc)
        if ok:
            self.model.add_col(idx)
            self._full_refresh()

    def _del_row(self) -> None:
        nr = self.model.num_rows()
        if nr <= 1:
            QMessageBox.warning(self, "Grid", "Cannot remove the last row."); return
        idx, ok = QInputDialog.getInt(self, "Remove Row",
                                      "Row index to remove:", nr - 1, 0, nr - 1)
        if ok:
            self.model.remove_row(idx)
            self._full_refresh()

    def _del_col(self) -> None:
        nc = self.model.num_cols()
        if nc <= 1:
            QMessageBox.warning(self, "Grid", "Cannot remove the last column."); return
        idx, ok = QInputDialog.getInt(self, "Remove Column",
                                      "Column index to remove:", nc - 1, 0, nc - 1)
        if ok:
            self.model.remove_col(idx)
            self._full_refresh()

    # ── Pixel events ──────────────────────────────────────────────────────────

    def _on_pixel_click(self, r: int, c: int) -> None:
        self._grid.set_selected(r, c)
        self._prop.select_pixel(r, c)

        ptype   = self.model.get_pixel(r, c)
        label   = PIXEL_INFO.get(ptype, ("?", None))[0]
        extras  = []
        si      = self.model.sw_index_at(r, c)
        ii      = self.model.ind_index_at(r, c)
        ci      = self.model.scap_index_at(r, c)
        pi      = self.model.cap_pair_index_at(r, c)
        if si  is not None: extras.append(f"SW_{si}")
        if ii  is not None:
            iv = (self.model.inductor_values[ii]
                  if ii < len(self.model.inductor_values) else 0)
            extras.append(f"IND_{ii} ({INDUCTOR_LABELS[iv]})")
        if ci  is not None:
            cv = (self.model.shunt_cap_values[ci]
                  if ci < len(self.model.shunt_cap_values) else DEFAULT_SCAP_F)
            extras.append(f"SCAP_{ci} ({cv*1e15:.0f} fF)")
        if pi  is not None: extras.append(f"cap-pair {pi}")
        tail = "  [" + ", ".join(extras) + "]" if extras else ""
        self.statusBar().showMessage(f"({r},{c}) → {label}{tail}")

    def _on_pixel_paint(self, _r: int, _c: int) -> None:
        self._states_table.refresh()

    def _on_data_changed(self) -> None:
        self._grid.refresh()
        self._states_table.refresh()

    # ── Simulation & Settings ─────────────────────────────────────────────────

    def _on_run_sim(self) -> None:
        dlg = SimulationWindow(self.model, self.gui_settings, self)
        dlg.show()   # non-blocking — user can keep editing while it runs

    def _on_settings(self) -> None:
        dlg = SettingsDialog(self.gui_settings, self.model, self)
        if dlg.exec_() == QDialog.Accepted:
            self.statusBar().showMessage(
                "Settings saved — changes to design paths take effect on next Save.")

    # ── Full refresh ──────────────────────────────────────────────────────────

    def _full_refresh(self) -> None:
        # Always enter fit mode on load/refresh so grid fills available space
        self._grid.enter_fit_mode()
        self._scroll.setWidgetResizable(True)
        self._grid.clear_selection()
        self._grid.refresh()
        self._prop.clear_selection()
        self._states_table.refresh()

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key_S and not ev.modifiers():
            self._set_mode(MODE_SELECT)
        elif ev.key() == Qt.Key_B and not ev.modifiers():
            self._set_mode(MODE_BRUSH)
        elif ev.key() == Qt.Key_Escape:
            self._set_mode(MODE_SELECT)
        else:
            super().keyPressEvent(ev)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Pixel Filter Designer")

    path = sys.argv[1] if len(sys.argv) > 1 else None
    win  = MainWindow(path)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
