"""
inductor_cost.py — Cost function for inductor optimization.

Objective: match a target inductance L (nH) across a frequency band while
maximizing the quality factor Q.  Used by InductorGA in ga_optimizer.py.

The L and Q are derived from the 2-port S-parameters of the simulated pixel
structure using the port-2-shorted input impedance formula (same as plotter.py):

    S_in = S11 - S21 * S12 / (1 + S22)
    Z_in = Z0 * (1 + S_in) / (1 - S_in)
    L    = Im(Z_in) / (2π · f) · 1e9   [nH]
    Q    = Im(Z_in) / Re(Z_in)

Cost formula
------------
    L_penalty = worst_k_mean( (L(f) - target_l_nh)^2 )  over the eval band
    Q_reward  = -mean(Q) / q_ref          (negative: minimising cost maximises Q)
    cost      = l_weight * L_penalty + q_weight * Q_reward
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from virtuoso_optimization_SA.cost import load_s2p


@dataclass
class InductorSpec:
    """Optimization target for an inductor design (single state)."""
    freq_start_hz: float
    freq_stop_hz:  float
    target_l_nh:   float
    l_weight:      float = 1.0
    q_weight:      float = 1.0
    q_ref:         float = 10.0
    worst_k:       int   = 10
    z0:            float = 50.0


def compute_lq_from_s2p(
    path: str | Path,
    z0: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a 2-port s2p and return (freqs_hz, L_nH, Q).

    Numerical guards are applied when denominators are near zero so that
    the result contains NaN rather than Inf at problematic frequency points.
    """
    ntwk = load_s2p(path)
    s = ntwk.s                        # shape (n_freq, 2, 2)
    S11 = s[:, 0, 0]
    S12 = s[:, 0, 1]
    S21 = s[:, 1, 0]
    S22 = s[:, 1, 1]
    freq_hz = ntwk.f

    denom_load = 1.0 + S22
    safe_load = np.abs(denom_load) > 1e-12
    s_in = np.where(
        safe_load,
        S11 - S21 * S12 / np.where(safe_load, denom_load, 1.0),
        np.nan + 0j,
    )

    denom_z = 1.0 - s_in
    safe_z = np.abs(denom_z) > 1e-12
    z_in = np.where(
        safe_z,
        z0 * (1.0 + s_in) / np.where(safe_z, denom_z, 1.0),
        np.nan + 0j,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        l_nh = np.where(
            freq_hz > 0,
            np.imag(z_in) / (2.0 * np.pi * freq_hz) * 1e9,
            np.nan,
        ).real
        q = np.where(
            np.real(z_in) > 1e-12,
            np.imag(z_in) / np.real(z_in),
            np.nan,
        ).real

    return freq_hz, l_nh, q


def _worst_k_mean(vals: np.ndarray, k: int) -> float:
    """Mean of the k largest finite values; 0.0 if none."""
    clean = vals[np.isfinite(vals)]
    if len(clean) == 0:
        return 0.0
    k_eff = min(k, len(clean))
    return float(np.mean(np.sort(clean)[-k_eff:]))


def compute_inductor_cost(s2p_path: str | Path | None, spec: InductorSpec) -> float:
    """Return the inductor optimization cost for a single simulated s2p file.

    Returns 1e9 on failure (missing file, parse error, no frequency points in band).
    """
    _FAIL = 1e9
    if s2p_path is None or not Path(s2p_path).exists():
        return _FAIL

    try:
        freq_hz, l_nh, q = compute_lq_from_s2p(s2p_path, spec.z0)
    except Exception:
        return _FAIL

    mask = (freq_hz >= spec.freq_start_hz) & (freq_hz <= spec.freq_stop_hz)
    if not np.any(mask):
        return _FAIL

    l_band = l_nh[mask]
    q_band = q[mask]

    l_errors = (l_band - spec.target_l_nh) ** 2
    l_penalty = _worst_k_mean(l_errors, spec.worst_k)

    q_clean = q_band[np.isfinite(q_band)]
    q_mean = float(np.mean(q_clean)) if len(q_clean) > 0 else 0.0
    q_reward = -q_mean / spec.q_ref

    return spec.l_weight * l_penalty + spec.q_weight * q_reward


def inductor_spec_from_config(cfg: dict) -> InductorSpec:
    """Parse the ``inductor_objective`` block from a raw config dict."""
    ind = cfg["inductor_objective"]
    return InductorSpec(
        freq_start_hz=float(ind["freq_start_hz"]),
        freq_stop_hz=float(ind["freq_stop_hz"]),
        target_l_nh=float(ind["target_l_nh"]),
        l_weight=float(ind.get("l_weight", 1.0)),
        q_weight=float(ind.get("q_weight", 1.0)),
        q_ref=float(ind.get("q_ref", 10.0)),
        worst_k=int(ind.get("worst_k", 10)),
        z0=float(ind.get("z0", 50.0)),
    )
