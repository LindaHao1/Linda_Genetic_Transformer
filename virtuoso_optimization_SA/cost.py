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
from itertools import permutations as _permutations
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
                total += 2 * shortfall
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
                    total += 0.5 * excess * spec.cross_stopband_weight
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
# Dynamic state assignment
# ---------------------------------------------------------------------------

def _build_cost_matrix(
    s2p_paths: List[Optional[Path]],
    specs:     List[FilterStateSpec],
) -> np.ndarray:
    """
    Build an N×N cost matrix where entry [sim_idx, spec_idx] is the cost of
    evaluating simulated state ``sim_idx`` against spec ``spec_idx``.

    N = number of specs = number of s2p_paths.  A failed simulation row is
    filled with _FAIL_PENALTY so the solver naturally avoids assigning it.
    """
    _FAIL_PENALTY = 1e9
    n = len(specs)
    matrix = np.full((n, n), _FAIL_PENALTY)

    loaded: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    for path in s2p_paths[:n]:
        if path is None or not Path(path).exists():
            loaded.append(None)
        else:
            try:
                loaded.append(get_s21_db(path))
            except Exception:  # noqa: BLE001
                loaded.append(None)

    for sim_idx, data in enumerate(loaded):
        if data is None:
            continue
        freqs, s21_db_vals = data
        for spec_idx, spec in enumerate(specs):
            matrix[sim_idx, spec_idx] = compute_state_cost(freqs, s21_db_vals, spec)

    return matrix


def compute_total_cost_dynamic(
    s2p_paths: List[Optional[Path]],
    specs:     List[FilterStateSpec],
) -> Tuple[float, List[int]]:
    """
    Find the bijective assignment of simulated states to specs that minimises
    total cost, then return ``(total_cost, assignment)`` where
    ``assignment[spec_idx] = sim_idx``.

    This solves the linear sum assignment problem (a.k.a. the Hungarian
    algorithm).  scipy is used when available; otherwise all N! permutations
    are tried (safe for N ≤ 8 which covers all practical cases).

    Parameters
    ----------
    s2p_paths : one .s2p file per simulated state, in simulation order.
    specs     : one FilterStateSpec per target band, in spec order.

    Returns
    -------
    total_cost : float — sum of per-state costs under the optimal assignment.
    assignment : list of length len(specs) — assignment[spec_idx] = sim_idx.
    """
    n = len(specs)
    if n == 0:
        return 0.0, []

    # Pad s2p_paths to length n if shorter (treat missing entries as failures)
    padded = list(s2p_paths) + [None] * max(0, n - len(s2p_paths))

    cost_matrix = _build_cost_matrix(padded, specs)

    # --- solve assignment ---
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # row_ind are sim indices, col_ind are spec indices
        # Cast to plain int — scipy returns numpy.int64 which is not JSON-serialisable
        assignment = [0] * n
        for sim_idx, spec_idx in zip(row_ind, col_ind):
            assignment[int(spec_idx)] = int(sim_idx)
        total_cost = float(cost_matrix[row_ind, col_ind].sum())
    except ImportError:
        # Brute-force over all N! permutations of sim indices
        best_cost = float("inf")
        best_perm = list(range(n))
        for perm in _permutations(range(n)):
            c = sum(cost_matrix[perm[spec_idx], spec_idx] for spec_idx in range(n))
            if c < best_cost:
                best_cost = c
                best_perm = list(perm)
        assignment = best_perm   # assignment[spec_idx] = sim_idx
        total_cost = best_cost

    return total_cost, assignment


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
