"""
cost.py — Cost function for the switchable-filter SA optimiser.

Each filter state has a target spec that defines:
  - A passband region with a minimum acceptable S21 (in dB).
  - One or more stopband regions with a maximum acceptable S21 (in dB).

The cost is asymmetric: we only penalise when the response is *worse* than
the target.  Being better than the target adds no penalty.

  passband  penalty: max(0,  passband_min_db  - S21(f)) ** 2
  stopband  penalty: max(0,  S21(f)  - stopband_max_db) ** 2

The total cost is the sum of all per-frequency penalties across all states,
optionally weighted by ``state_weight``.

S-parameter loading
-------------------
Uses scikit-rf (skrf) to load Touchstone files, the same library used by the
plotter in virtuoso_engine/plotter.py.  This guarantees identical parsing of
files produced by Spectre and EMX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import skrf as rf
except ImportError as exc:
    raise ImportError(
        "scikit-rf is required for S-parameter loading.  Install it with:\n"
        "    pip install scikit-rf"
    ) from exc


# ---------------------------------------------------------------------------
# S-parameter loading via skrf  (mirrors virtuoso_engine/plotter.py)
# ---------------------------------------------------------------------------

def load_s2p(path: str | Path):
    """
    Load a 2-port Touchstone file using scikit-rf.

    Returns
    -------
    ntwk : skrf.Network
        The loaded network.  Key attributes:
          ntwk.f          — numpy array of frequencies (Hz)
          ntwk.s_db[:, 1, 0]  — S21 magnitude in dB  (row=1, col=0)
          ntwk.s_db[:, 0, 0]  — S11 magnitude in dB
    """
    return rf.Network(str(path))


def get_s21_db(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: load an s2p file and return (freqs_Hz, S21_dB).

    Returns numpy arrays so they work directly in cost computations.
    """
    ntwk = load_s2p(path)
    return ntwk.f, ntwk.s_db[:, 1, 0]


# ---------------------------------------------------------------------------
# Filter specification dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilterStateSpec:
    """
    Performance target for one filter state.

    Attributes
    ----------
    state_index   : which entry in state_list / s2p_files this spec applies to.
    passband      : (f_start, f_stop) in Hz — desired pass region.
    passband_min_db : minimum acceptable S21 in passband (dB).  Penalty if below.
    stopband_regions : list of (f_start, f_stop) tuples in Hz — reject regions.
    stopband_max_db  : maximum acceptable S21 in stopband (dB).  Penalty if above.
    weight           : scale factor for this state's contribution to total cost.
    cross_stopband_regions : passbands of *other* states — this state must attenuate
                             there to encourage fully decoupled operation.  Penalty if
                             S21 exceeds cross_stopband_max_db in any of these regions.
    cross_stopband_max_db  : attenuation target for cross-state regions (dB).
    cross_stopband_weight  : extra weight applied only to the cross-state penalty term.
    """
    state_index:           int
    passband:              Tuple[float, float]
    passband_min_db:       float
    stopband_regions:      List[Tuple[float, float]] = field(default_factory=list)
    stopband_max_db:       float = -15.0
    weight:                float = 1.0
    cross_stopband_regions: List[Tuple[float, float]] = field(default_factory=list)
    cross_stopband_max_db:  float = -20.0
    cross_stopband_weight:  float = 1.0


# ---------------------------------------------------------------------------
# Per-state cost
# ---------------------------------------------------------------------------

def compute_state_cost(
    freqs:    List[float],
    s21_db_values: List[float],
    spec:     FilterStateSpec,
) -> float:
    """
    Compute asymmetric mean-absolute-error cost for one state.

    Only penalise responses *worse* than the target:
      - passband:  penalty when S21(f) <  passband_min_db
      - stopband:  penalty when S21(f) >  stopband_max_db
    The penalty at each frequency point is the absolute violation (dB),
    averaged over all evaluated points.
    """
    total    = 0.0
    n_points = 0
    pb_lo, pb_hi = spec.passband

    for f, s21 in zip(freqs, s21_db_values):
        # Passband penalty
        if pb_lo <= f <= pb_hi:
            shortfall = spec.passband_min_db - s21
            if shortfall > 0:
                total += 5 * shortfall
            n_points += 1

        # Stopband penalty
        for sb_lo, sb_hi in spec.stopband_regions:
            if sb_lo <= f <= sb_hi:
                excess = s21 - spec.stopband_max_db
                if excess > 0:
                    total += excess
                n_points += 1

        # Cross-state decoupling penalty
        for cs_lo, cs_hi in spec.cross_stopband_regions:
            if cs_lo <= f <= cs_hi:
                excess = s21 - spec.cross_stopband_max_db
                if excess > 0:
                    total += excess * spec.cross_stopband_weight
                n_points += 1

    mean_abs = total / n_points if n_points > 0 else 0.0
    return mean_abs * spec.weight


# ---------------------------------------------------------------------------
# Total cost over all states
# ---------------------------------------------------------------------------

def compute_total_cost(
    s2p_paths: List[Optional[Path]],
    specs:     List[FilterStateSpec],
) -> float:
    """
    Compute the total cost for a design given the .s2p files for all states.

    Parameters
    ----------
    s2p_paths : list of Path or None
        One entry per filter state.  None means the simulation failed for that
        state — a large penalty is returned instead.
    specs     : list of FilterStateSpec, one per state.

    Returns
    -------
    float total cost (lower is better).
    """
    _FAIL_PENALTY = 1e9   # penalty for a failed simulation

    total = 0.0
    spec_by_idx: Dict[int, FilterStateSpec] = {s.state_index: s for s in specs}

    for state_idx, s2p_path in enumerate(s2p_paths):
        spec = spec_by_idx.get(state_idx)
        if spec is None:
            continue  # no target for this state → skip

        if s2p_path is None or not Path(s2p_path).exists():
            total += _FAIL_PENALTY * spec.weight
            continue

        try:
            freqs, s21_db_vals = get_s21_db(s2p_path)
            total += compute_state_cost(freqs, s21_db_vals, spec)
        except Exception as exc:  # noqa: BLE001
            print(f"  [cost] WARNING: failed to parse {s2p_path}: {exc}")
            total += _FAIL_PENALTY * spec.weight

    return total


# ---------------------------------------------------------------------------
# Build FilterStateSpec objects from config dicts
# ---------------------------------------------------------------------------

def specs_from_config(filter_specs_cfg: list) -> List[FilterStateSpec]:
    """
    Convert the ``filter_specs`` list from the SA config YAML into
    :class:`FilterStateSpec` objects.

    Expected YAML structure per entry::

        - state_index: 0
          passband: [6.0e9, 8.0e9]
          passband_min_db: -3.0
          stopband_regions:
            - [0.0, 5.5e9]
            - [8.5e9, 40.0e9]
          stopband_max_db: -15.0
          weight: 1.0
    """
    specs: List[FilterStateSpec] = []
    for entry in filter_specs_cfg:
        pb = entry["passband"]
        # Explicitly cast to float — PyYAML parses values like `11e9` (no
        # decimal point, no explicit exponent sign) as strings, not floats.
        sb_regions = [
            (float(r[0]), float(r[1]))
            for r in entry.get("stopband_regions", [])
        ]
        cs_regions = [
            (float(r[0]), float(r[1]))
            for r in entry.get("cross_stopband_regions", [])
        ]
        specs.append(FilterStateSpec(
            state_index            = int(entry["state_index"]),
            passband               = (float(pb[0]), float(pb[1])),
            passband_min_db        = float(entry.get("passband_min_db",       -3.0)),
            stopband_regions       = sb_regions,
            stopband_max_db        = float(entry.get("stopband_max_db",      -15.0)),
            weight                 = float(entry.get("weight",                 1.0)),
            cross_stopband_regions = cs_regions,
            cross_stopband_max_db  = float(entry.get("cross_stopband_max_db", -20.0)),
            cross_stopband_weight  = float(entry.get("cross_stopband_weight",  1.0)),
        ))
    return specs
