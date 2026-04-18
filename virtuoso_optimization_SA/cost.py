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


def _worst_k_mean(penalties: list, k: int) -> float:
    """Return the mean of the k largest values in *penalties*.

    Frequencies that do not violate their target contribute 0.0.  If there are
    fewer than *k* values the mean is taken over all of them.  Returns 0.0 for
    an empty list.
    """
    if not penalties:
        return 0.0
    k_eff = min(k, len(penalties))
    top_k = sorted(penalties, reverse=True)[:k_eff]
    return sum(top_k) / k_eff


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


def get_s11_db(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: load an s2p file and return (freqs_Hz, S11_dB).

    S11 is the reflection coefficient at port 1 (row=0, col=0).
    """
    ntwk = load_s2p(path)
    return ntwk.f, ntwk.s_db[:, 0, 0]


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
    passband_weight  : fusion weight for the passband penalty component.
    stopband_weight  : fusion weight for the stopband penalty component.
    cross_stopband_regions : passbands of *other* states — this state must attenuate
                             there to encourage fully decoupled operation.  Penalty if
                             S21 exceeds cross_stopband_max_db in any of these regions.
    cross_stopband_max_db  : attenuation target for cross-state regions (dB).
    cross_stopband_weight  : fusion weight for the cross-state penalty component.
    enable_s11_cost        : when True, add S11-based reflection penalties to the cost.
    s11_passband_max_db    : S11 in the passband should be at or below this (dB).
                             Penalty if S11 > s11_passband_max_db (want low reflection).
    s11_stopband_min_db    : S11 in stopbands should be at or above this (dB).
                             Penalty if S11 < s11_stopband_min_db (want high reflection).
    s11_cross_stopband_min_db : same as s11_stopband_min_db but for cross-state bands.
    """
    state_index:           int
    passband:              Tuple[float, float]
    passband_min_db:       float
    stopband_regions:      List[Tuple[float, float]] = field(default_factory=list)
    stopband_max_db:       float = -15.0
    weight:                float = 1.0
    passband_weight:       float = 1.0
    stopband_weight:       float = 1.0
    cross_stopband_regions: List[Tuple[float, float]] = field(default_factory=list)
    cross_stopband_max_db:  float = -20.0
    cross_stopband_weight:  float = 1.0
    # S11 reflection-coefficient targets
    enable_s11_cost:           bool  = False
    s11_passband_max_db:       float = -30.0
    s11_stopband_min_db:       float = -10.0
    s11_cross_stopband_min_db: float = -10.0

    # Lp norm order for penalty accumulation (1.0 = L1, 1.5 = L1.5, 2.0 = L2, etc.)
    cost_order: float = 1.0

    # Worst-k average: instead of averaging ALL per-frequency penalties (which
    # dilutes a few large violations when there are many frequency points), only
    # average the k largest penalty values.  Set to a large number (e.g. 9999)
    # to fall back to a plain mean.
    worst_k: int = 10

    # Peak matching: find the global S21 maximum over the full simulated frequency
    # range, shift all frequency ranges (passband/stopband/cross-stopband) to
    # follow it, then multiply the total cost by a linear penalty based on how
    # far the peak shifted relative to peak_matching_threshold * centre_freq.
    # multiplier = 1.0 + shift_ratio * (peak_matching_max_multiplier - 1.0)
    # When disabled (default), behaviour is identical to the original code.
    enable_peak_matching:         bool  = False
    peak_matching_threshold:      float = 0.20   # fraction of centre freq, e.g. 0.20 = ±20%
    peak_matching_max_multiplier: float = 2.0    # multiplier at full threshold (must be >= 1.0)

    # Spurious peak penalty: penalise any S21 value that exceeds spurious_peak_max_db
    # outside the (shifted) passband.  Intended to discourage secondary passbands —
    # each state should have exactly one transmission peak.
    # Uses worst-k mean aggregation (same worst_k as other penalties).
    enable_spurious_peak_penalty: bool  = False
    spurious_peak_max_db:         float = -10.0  # S21 above this outside passband = spurious
    spurious_peak_weight:         float = 1.0    # fusion weight, scaled by spec.weight


# ---------------------------------------------------------------------------
# Peak matching helper
# ---------------------------------------------------------------------------

def _find_passband_peak(
    freqs:         np.ndarray,
    s21_db_values: np.ndarray,
) -> Optional[float]:
    """
    Find the global S21 peak frequency over the entire simulated frequency range.

    Returns the frequency (Hz) of the maximum S21, or None if freqs is empty.
    """
    if len(freqs) == 0:
        return None
    return float(freqs[int(np.argmax(s21_db_values))])


# ---------------------------------------------------------------------------
# Per-state cost
# ---------------------------------------------------------------------------

def compute_state_cost(
    freqs:         List[float],
    s21_db_values: List[float],
    spec:          FilterStateSpec,
    s11_db_values: Optional[List[float]] = None,
) -> float:
    """
    Compute asymmetric weighted-composite cost for one state.

    S21 penalties (always active):
      - passband:  penalty when S21(f) <  passband_min_db
      - stopband:  penalty when S21(f) >  stopband_max_db
      - cross-state: penalty when S21(f) > cross_stopband_max_db

    S11 penalties (active when spec.enable_s11_cost is True and s11_db_values
    is provided).  Reuses the same per-state fusion weights:
      - passband:  penalty when S11(f) >  s11_passband_max_db   (too much reflection)
      - stopband:  penalty when S11(f) <  s11_stopband_min_db   (not enough reflection)
      - cross-state: penalty when S11(f) < s11_cross_stopband_min_db

    Each penalty class is normalised by its own point count.  The S21 and S11
    components are summed to form the final state cost.
    """
    # --- Peak matching preamble ---
    # When enabled: find the S21 peak near the passband, shift all frequency
    # ranges to follow it, then add a linear multiplier penalty proportional
    # to the shift magnitude.  When disabled (default) shift=0 and the
    # behaviour is identical to the original code.
    shift = 0.0          # actual measured peak offset from spec centre (diagnostic)
    applied_shift = 0.0  # shift applied to ranges — clamped to ±threshold so ranges
                         # never drift far from the spec; unclamped shift causes wrong
                         # assignments to score artificially well in dynamic assignment.
    shift_ratio = 0.0
    shift_penalty_multiplier = 1.0
    if spec.enable_peak_matching:
        _freqs_arr = np.asarray(freqs)
        _s21_arr   = np.asarray(s21_db_values)
        peak_freq  = _find_passband_peak(_freqs_arr, _s21_arr)
        if peak_freq is not None:
            pb_lo_nom, pb_hi_nom = spec.passband
            centre       = (pb_lo_nom + pb_hi_nom) / 2.0
            shift        = peak_freq - centre
            threshold_hz = spec.peak_matching_threshold * centre
            shift_ratio  = abs(shift) / threshold_hz
            if shift_ratio <= 1.0:
                # Peak is within threshold: shift ranges to follow it
                applied_shift = shift
            else:
                # Peak is beyond threshold: do NOT shift ranges.  The filter's
                # peak is too far from this spec — keeping original ranges ensures
                # the base cost is high (poor transmission / rejection in the
                # unshifted bands), correctly flagging a bad state-to-spec match.
                applied_shift = 0.0
                shift_ratio   = 1.0  # cap for multiplier calculation
            shift_penalty_multiplier = 1.0 + shift_ratio * (spec.peak_matching_max_multiplier - 1.0)
    # --- end preamble ---

    pb_lo = spec.passband[0] + applied_shift
    pb_hi = spec.passband[1] + applied_shift
    sb_regions_eff = [(lo + applied_shift, hi + applied_shift) for lo, hi in spec.stopband_regions]
    cs_regions_eff = [(lo + applied_shift, hi + applied_shift) for lo, hi in spec.cross_stopband_regions]

    pb_penalties: List[float] = []
    sb_penalties: List[float] = []
    cs_penalties: List[float] = []
    sp_penalties: List[float] = []   # spurious peak penalties

    for f, s21 in zip(freqs, s21_db_values):
        # Passband penalty
        if pb_lo <= f <= pb_hi:
            shortfall = spec.passband_min_db - s21
            pb_penalties.append(shortfall ** spec.cost_order if shortfall > 0 else 0.0)

        # Stopband penalty
        for sb_lo, sb_hi in sb_regions_eff:
            if sb_lo <= f <= sb_hi:
                excess = s21 - spec.stopband_max_db
                sb_penalties.append(excess ** spec.cost_order if excess > 0 else 0.0)

        # Cross-state decoupling penalty
        for cs_lo, cs_hi in cs_regions_eff:
            if cs_lo <= f <= cs_hi:
                excess = s21 - spec.cross_stopband_max_db
                cs_penalties.append(excess ** spec.cost_order if excess > 0 else 0.0)

        # Spurious peak penalty: any S21 above threshold outside the passband
        if spec.enable_spurious_peak_penalty and not (pb_lo <= f <= pb_hi):
            excess = s21 - spec.spurious_peak_max_db
            sp_penalties.append(excess ** spec.cost_order if excess > 0 else 0.0)

    # Use worst-k mean so that a handful of large violations are not diluted by
    # many compliant frequency points (especially important for wide cross-bands).
    mean_pb = _worst_k_mean(pb_penalties, spec.worst_k)
    mean_sb = _worst_k_mean(sb_penalties, spec.worst_k)
    mean_cs = _worst_k_mean(cs_penalties, spec.worst_k)
    mean_sp = _worst_k_mean(sp_penalties, spec.worst_k) if spec.enable_spurious_peak_penalty else 0.0

    s21_cost = (
        spec.passband_weight * mean_pb
        + spec.stopband_weight * mean_sb
        + spec.cross_stopband_weight * mean_cs
        + spec.spurious_peak_weight * mean_sp
    ) * spec.weight

    # --- S11 cost (optional) ---
    s11_cost = 0.0
    if spec.enable_s11_cost and s11_db_values is not None:
        s11_pb_penalties: List[float] = []
        s11_sb_penalties: List[float] = []
        s11_cs_penalties: List[float] = []
        for f, s11 in zip(freqs, s11_db_values):
            # Passband: penalise if S11 is too high (too much reflection in pass band)
            if pb_lo <= f <= pb_hi:
                excess = s11 - spec.s11_passband_max_db
                s11_pb_penalties.append(excess ** spec.cost_order if excess > 0 else 0.0)
            # Stopband: penalise if S11 is too low (not enough reflection in stop band)
            for sb_lo, sb_hi in sb_regions_eff:
                if sb_lo <= f <= sb_hi:
                    shortfall = spec.s11_stopband_min_db - s11
                    s11_sb_penalties.append(shortfall ** spec.cost_order if shortfall > 0 else 0.0)
            # Cross-state band: penalise if S11 is too low
            for cs_lo, cs_hi in cs_regions_eff:
                if cs_lo <= f <= cs_hi:
                    shortfall = spec.s11_cross_stopband_min_db - s11
                    s11_cs_penalties.append(shortfall ** spec.cost_order if shortfall > 0 else 0.0)
        s11_mean_pb = _worst_k_mean(s11_pb_penalties, spec.worst_k)
        s11_mean_sb = _worst_k_mean(s11_sb_penalties, spec.worst_k)
        s11_mean_cs = _worst_k_mean(s11_cs_penalties, spec.worst_k)
        s11_cost = (
            spec.passband_weight * s11_mean_pb
            + spec.stopband_weight * s11_mean_sb
            + spec.cross_stopband_weight * s11_mean_cs
        ) * spec.weight

    return (s21_cost + s11_cost) * shift_penalty_multiplier


# ---------------------------------------------------------------------------
# Per-state cost breakdown (individual components)
# ---------------------------------------------------------------------------

def compute_state_cost_breakdown(
    freqs:         List[float],
    s21_db_values: List[float],
    spec:          FilterStateSpec,
    s11_db_values: Optional[List[float]] = None,
) -> dict:
    """
    Like compute_state_cost but returns a dict with every sub-component instead
    of a single scalar.  Keys:

      s21_passband_raw      — worst-k mean Lp S21 shortfall below passband_min_db  (unweighted; drives cost)
      s21_stopband_raw      — worst-k mean Lp S21 excess  above stopband_max_db    (unweighted; drives cost)
      s21_crossband_raw     — worst-k mean Lp S21 excess  above cross_stopband_max_db (unweighted; drives cost)
      s21_passband_all_mean — simple mean over all in-band frequencies (unweighted; for reference)
      s21_stopband_all_mean — simple mean over all in-band frequencies (unweighted; for reference)
      s21_crossband_all_mean— simple mean over all in-band frequencies (unweighted; for reference)
      s21_passband          — weighted passband penalty   (raw * passband_weight * weight)
      s21_stopband          — weighted stopband penalty   (raw * stopband_weight * weight)
      s21_crossband         — weighted crossband penalty  (raw * cross_stopband_weight * weight)
      s11_passband_raw      — worst-k mean Lp S11 excess  above s11_passband_max_db (unweighted, 0 if disabled)
      s11_stopband_raw      — worst-k mean Lp S11 shortfall below s11_stopband_min_db (unweighted, 0 if disabled)
      s11_crossband_raw     — worst-k mean Lp S11 shortfall below s11_cross_stopband_min_db (unweighted, 0 if disabled)
      s11_passband_all_mean — simple mean S11 passband penalty (unweighted; for reference)
      s11_stopband_all_mean — simple mean S11 stopband penalty (unweighted; for reference)
      s11_crossband_all_mean— simple mean S11 crossband penalty (unweighted; for reference)
      s11_passband          — weighted S11 passband penalty
      s11_stopband          — weighted S11 stopband penalty
      s11_crossband         — weighted S11 crossband penalty
      total                 — sum of all weighted components * shift_penalty_multiplier (== compute_state_cost(...))
      peak_shift_hz         — S21 peak shift from passband centre in Hz (0 if peak matching disabled)
      peak_shift_ratio      — shift / (threshold * centre), clamped to [0,1] (0 if disabled)
      shift_penalty_multiplier — total cost multiplier from peak shift, in [1.0, 2.0] (1.0 if disabled)
    """
    # --- Peak matching preamble (identical to compute_state_cost) ---
    shift = 0.0
    applied_shift = 0.0
    shift_ratio = 0.0
    shift_penalty_multiplier = 1.0
    if spec.enable_peak_matching:
        _freqs_arr = np.asarray(freqs)
        _s21_arr   = np.asarray(s21_db_values)
        peak_freq  = _find_passband_peak(_freqs_arr, _s21_arr)
        if peak_freq is not None:
            pb_lo_nom, pb_hi_nom = spec.passband
            centre       = (pb_lo_nom + pb_hi_nom) / 2.0
            shift        = peak_freq - centre
            threshold_hz = spec.peak_matching_threshold * centre
            shift_ratio  = abs(shift) / threshold_hz
            if shift_ratio <= 1.0:
                applied_shift = shift
            else:
                applied_shift = 0.0
                shift_ratio   = 1.0
            shift_penalty_multiplier = 1.0 + shift_ratio * (spec.peak_matching_max_multiplier - 1.0)
    # --- end preamble ---

    pb_lo = spec.passband[0] + applied_shift
    pb_hi = spec.passband[1] + applied_shift
    sb_regions_eff = [(lo + applied_shift, hi + applied_shift) for lo, hi in spec.stopband_regions]
    cs_regions_eff = [(lo + applied_shift, hi + applied_shift) for lo, hi in spec.cross_stopband_regions]

    pb_penalties: List[float] = []
    sb_penalties: List[float] = []
    cs_penalties: List[float] = []
    sp_penalties: List[float] = []   # spurious peak penalties

    for f, s21 in zip(freqs, s21_db_values):
        if pb_lo <= f <= pb_hi:
            shortfall = spec.passband_min_db - s21
            pb_penalties.append(shortfall ** spec.cost_order if shortfall > 0 else 0.0)
        for sb_lo, sb_hi in sb_regions_eff:
            if sb_lo <= f <= sb_hi:
                excess = s21 - spec.stopband_max_db
                sb_penalties.append(excess ** spec.cost_order if excess > 0 else 0.0)
        for cs_lo, cs_hi in cs_regions_eff:
            if cs_lo <= f <= cs_hi:
                excess = s21 - spec.cross_stopband_max_db
                cs_penalties.append(excess ** spec.cost_order if excess > 0 else 0.0)
        if spec.enable_spurious_peak_penalty and not (pb_lo <= f <= pb_hi):
            excess = s21 - spec.spurious_peak_max_db
            sp_penalties.append(excess ** spec.cost_order if excess > 0 else 0.0)

    # worst-k mean — drives the actual cost
    mean_pb = _worst_k_mean(pb_penalties, spec.worst_k)
    mean_sb = _worst_k_mean(sb_penalties, spec.worst_k)
    mean_cs = _worst_k_mean(cs_penalties, spec.worst_k)
    mean_sp = _worst_k_mean(sp_penalties, spec.worst_k) if spec.enable_spurious_peak_penalty else 0.0
    # simple all-mean — kept for reference / diagnostics
    all_mean_pb = (sum(pb_penalties) / len(pb_penalties)) if pb_penalties else 0.0
    all_mean_sb = (sum(sb_penalties) / len(sb_penalties)) if sb_penalties else 0.0
    all_mean_cs = (sum(cs_penalties) / len(cs_penalties)) if cs_penalties else 0.0
    all_mean_sp = (sum(sp_penalties) / len(sp_penalties)) if sp_penalties else 0.0

    s21_pb = spec.passband_weight        * mean_pb * spec.weight
    s21_sb = spec.stopband_weight        * mean_sb * spec.weight
    s21_cs = spec.cross_stopband_weight  * mean_cs * spec.weight
    s21_sp = spec.spurious_peak_weight   * mean_sp * spec.weight

    # --- S11 breakdown ---
    s11_pb_raw = s11_sb_raw = s11_cs_raw = 0.0
    s11_pb_all = s11_sb_all = s11_cs_all = 0.0
    s11_pb_w   = s11_sb_w   = s11_cs_w   = 0.0
    if spec.enable_s11_cost and s11_db_values is not None:
        s11_pb_penalties: List[float] = []
        s11_sb_penalties: List[float] = []
        s11_cs_penalties: List[float] = []
        for f, s11 in zip(freqs, s11_db_values):
            if pb_lo <= f <= pb_hi:
                excess = s11 - spec.s11_passband_max_db
                s11_pb_penalties.append(excess ** spec.cost_order if excess > 0 else 0.0)
            for sb_lo, sb_hi in sb_regions_eff:
                if sb_lo <= f <= sb_hi:
                    shortfall = spec.s11_stopband_min_db - s11
                    s11_sb_penalties.append(shortfall ** spec.cost_order if shortfall > 0 else 0.0)
            for cs_lo, cs_hi in cs_regions_eff:
                if cs_lo <= f <= cs_hi:
                    shortfall = spec.s11_cross_stopband_min_db - s11
                    s11_cs_penalties.append(shortfall ** spec.cost_order if shortfall > 0 else 0.0)
        s11_pb_raw = _worst_k_mean(s11_pb_penalties, spec.worst_k)
        s11_sb_raw = _worst_k_mean(s11_sb_penalties, spec.worst_k)
        s11_cs_raw = _worst_k_mean(s11_cs_penalties, spec.worst_k)
        s11_pb_all = (sum(s11_pb_penalties) / len(s11_pb_penalties)) if s11_pb_penalties else 0.0
        s11_sb_all = (sum(s11_sb_penalties) / len(s11_sb_penalties)) if s11_sb_penalties else 0.0
        s11_cs_all = (sum(s11_cs_penalties) / len(s11_cs_penalties)) if s11_cs_penalties else 0.0
        s11_pb_w = spec.passband_weight       * s11_pb_raw * spec.weight
        s11_sb_w = spec.stopband_weight       * s11_sb_raw * spec.weight
        s11_cs_w = spec.cross_stopband_weight * s11_cs_raw * spec.weight

    total = (s21_pb + s21_sb + s21_cs + s21_sp + s11_pb_w + s11_sb_w + s11_cs_w) * shift_penalty_multiplier
    return {
        # worst-k mean: what actually drives the cost (consistent with compute_state_cost)
        "s21_passband_raw":          mean_pb,
        "s21_stopband_raw":          mean_sb,
        "s21_crossband_raw":         mean_cs,
        "s21_spurious_raw":          mean_sp,
        # simple all-mean: for reference / plotting
        "s21_passband_all_mean":     all_mean_pb,
        "s21_stopband_all_mean":     all_mean_sb,
        "s21_crossband_all_mean":    all_mean_cs,
        "s21_spurious_all_mean":     all_mean_sp,
        # weighted cost components
        "s21_passband":              s21_pb,
        "s21_stopband":              s21_sb,
        "s21_crossband":             s21_cs,
        "s21_spurious":              s21_sp,
        # S11 worst-k mean
        "s11_passband_raw":          s11_pb_raw,
        "s11_stopband_raw":          s11_sb_raw,
        "s11_crossband_raw":         s11_cs_raw,
        # S11 simple all-mean
        "s11_passband_all_mean":     s11_pb_all,
        "s11_stopband_all_mean":     s11_sb_all,
        "s11_crossband_all_mean":    s11_cs_all,
        # weighted S11 cost components
        "s11_passband":              s11_pb_w,
        "s11_stopband":              s11_sb_w,
        "s11_crossband":             s11_cs_w,
        "total":                     total,
        # peak matching diagnostics
        "peak_shift_hz":             shift,          # actual peak offset from spec centre
        "peak_applied_shift_hz":     applied_shift,  # shift used for ranges (0 if |shift|>threshold)
        "peak_shift_ratio":          shift_ratio,
        "shift_penalty_multiplier":  shift_penalty_multiplier,
    }


def compute_cost_breakdown(
    s2p_paths:  List[Optional[Path]],
    specs:      List[FilterStateSpec],
    assignment: Optional[List[int]] = None,
) -> List[dict]:
    """
    Compute per-spec cost breakdowns for a design.

    Parameters
    ----------
    s2p_paths  : one s2p file per *simulated* state (in simulation order).
    specs      : FilterStateSpec list (in spec order).
    assignment : assignment[spec_idx] = sim_idx from dynamic assignment,
                 or None to use spec.state_index (static).

    Returns
    -------
    List of dicts (one per spec, in spec order) with keys from
    compute_state_cost_breakdown plus ``spec_index`` and ``passband_hz``.
    Returns an entry with ``total=1e9`` for failed/missing simulations.
    """
    _FAIL = 1e9
    results = []
    for spec_idx, spec in enumerate(specs):
        sim_idx = assignment[spec_idx] if assignment is not None else spec.state_index
        entry: dict = {
            "spec_idx":   spec_idx,
            "passband_hz": list(spec.passband),
        }
        if sim_idx >= len(s2p_paths) or s2p_paths[sim_idx] is None:
            entry.update({k: _FAIL for k in (
                "s21_passband_raw", "s21_stopband_raw", "s21_crossband_raw",
                "s21_passband_all_mean", "s21_stopband_all_mean", "s21_crossband_all_mean",
                "s21_passband", "s21_stopband", "s21_crossband",
                "s11_passband_raw", "s11_stopband_raw", "s11_crossband_raw",
                "s11_passband_all_mean", "s11_stopband_all_mean", "s11_crossband_all_mean",
                "s11_passband", "s11_stopband", "s11_crossband", "total",
            )})
        else:
            try:
                freqs, s21_db_vals = get_s21_db(s2p_paths[sim_idx])
                _, s11_db_vals     = get_s11_db(s2p_paths[sim_idx])
                bd = compute_state_cost_breakdown(freqs, s21_db_vals, spec,
                                                  s11_db_values=s11_db_vals)
                entry.update(bd)
            except Exception as exc:  # noqa: BLE001
                print(f"  [cost] WARNING: breakdown failed for {s2p_paths[sim_idx]}: {exc}")
                entry.update({k: _FAIL for k in (
                    "s21_passband_raw", "s21_stopband_raw", "s21_crossband_raw",
                    "s21_passband_all_mean", "s21_stopband_all_mean", "s21_crossband_all_mean",
                    "s21_passband", "s21_stopband", "s21_crossband",
                    "s11_passband_raw", "s11_stopband_raw", "s11_crossband_raw",
                    "s11_passband_all_mean", "s11_stopband_all_mean", "s11_crossband_all_mean",
                    "s11_passband", "s11_stopband", "s11_crossband", "total",
                )})
        results.append(entry)
    return results


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
            _, s11_db_vals     = get_s11_db(s2p_path)
            total += compute_state_cost(freqs, s21_db_vals, spec, s11_db_values=s11_db_vals)
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

    loaded: List[Optional[Tuple]] = []
    for path in s2p_paths[:n]:
        if path is None or not Path(path).exists():
            loaded.append(None)
        else:
            try:
                freqs, s21_db_vals = get_s21_db(path)
                _, s11_db_vals     = get_s11_db(path)
                loaded.append((freqs, s21_db_vals, s11_db_vals))
            except Exception:  # noqa: BLE001
                loaded.append(None)

    for sim_idx, data in enumerate(loaded):
        if data is None:
            continue
        freqs, s21_db_vals, s11_db_vals = data
        for spec_idx, spec in enumerate(specs):
            matrix[sim_idx, spec_idx] = compute_state_cost(
                freqs, s21_db_vals, spec, s11_db_values=s11_db_vals
            )

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
            passband_weight        = float(entry.get("passband_weight",        1.0)),
            stopband_weight        = float(entry.get("stopband_weight",        1.0)),
            cross_stopband_regions = cs_regions,
            cross_stopband_max_db  = float(entry.get("cross_stopband_max_db", -20.0)),
            cross_stopband_weight  = float(entry.get("cross_stopband_weight",  1.0)),
            enable_s11_cost           = bool(entry.get("enable_s11_cost",           False)),
            s11_passband_max_db       = float(entry.get("s11_passband_max_db",       -30.0)),
            s11_stopband_min_db       = float(entry.get("s11_stopband_min_db",       -10.0)),
            s11_cross_stopband_min_db = float(entry.get("s11_cross_stopband_min_db", -10.0)),
            cost_order                = float(entry.get("cost_order",                  1.0)),
            worst_k                   = int(  entry.get("worst_k",                        5)),
            enable_peak_matching          = bool( entry.get("enable_peak_matching",           False)),
            peak_matching_threshold       = float(entry.get("peak_matching_threshold",        0.20)),
            peak_matching_max_multiplier  = float(entry.get("peak_matching_max_multiplier",   2.0)),
            enable_spurious_peak_penalty  = bool( entry.get("enable_spurious_peak_penalty",  False)),
            spurious_peak_max_db          = float(entry.get("spurious_peak_max_db",          -10.0)),
            spurious_peak_weight          = float(entry.get("spurious_peak_weight",            1.0)),
        ))
    return specs
