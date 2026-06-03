"""
GA PV-contrast scoring, reused (not reimplemented) from
virtuoso_optimization_SA/cost.py.

The LM is conditioned on a compact per-state score vector and candidates are
ranked by the negative aggregated GA cost (soft_max).  S21 (K x n_freq, dB)
comes from the world model / simulator; everything else is delegated to
cost.py with the same FilterStateSpec objects the GA uses.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import yaml

_SA_DIR = "/disk/mcao53/RFIC_RL/virtuoso_optimization_SA"
if _SA_DIR not in sys.path:
    sys.path.insert(0, _SA_DIR)

from cost import (  # noqa: E402
    FilterStateSpec, _aggregate_state_costs, compute_state_cost,
    compute_state_cost_breakdown, specs_from_config,
)

# Conditioning scalars (order matters; matches breakdown keys in cost.py:441).
SCORE_SCALARS: List[str] = ["pv_pb_level", "pv_contrast", "pv_pb_std"]

# Higher-is-better ('max') vs lower-is-better ('min') per scalar.
DIRECTION: Dict[str, str] = {
    "pv_pb_level": "max", "pv_pb_avg": "max", "pv_contrast": "max",
    "pv_sb_avg": "min", "pv_pb_std": "min",
}

# Physical clamps (passive filter: passband S21 <= 0 dB; ripple std >= 0).
PB_LEVEL_MAX = 0.0
PB_STD_MIN = 0.0


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def build_specs(filter_specs_cfg: list) -> List[FilterStateSpec]:
    return specs_from_config(filter_specs_cfg)


def load_specs(yaml_path: str) -> List[FilterStateSpec]:
    """Read `filter_specs` from a GA config (e.g. finetune_3_states.yaml).

    That file overrides `filter_specs` wholesale, so `_base_` need not be
    resolved for the spec list.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return build_specs(cfg["filter_specs"])


# ---------------------------------------------------------------------------
# Score vectors / cost
# ---------------------------------------------------------------------------

def state_score_vector(freqs: Sequence[float], s21_db: Sequence[float],
                        spec: FilterStateSpec,
                        scalars: Sequence[str] = SCORE_SCALARS) -> np.ndarray:
    bd = compute_state_cost_breakdown(list(freqs), list(s21_db), spec)
    return np.array([float(bd.get(k, 0.0)) for k in scalars], dtype=np.float32)


def design_score_matrix(freqs: Sequence[float], s21_K: np.ndarray,
                         specs: List[FilterStateSpec],
                         scalars: Sequence[str] = SCORE_SCALARS) -> np.ndarray:
    """(K, len(scalars)) per-state score vectors. specs indexed by state_index."""
    spec_by_idx = {s.state_index: s for s in specs}
    out = []
    for k in range(s21_K.shape[0]):
        spec = spec_by_idx.get(k, specs[min(k, len(specs) - 1)])
        out.append(state_score_vector(freqs, s21_K[k], spec, scalars))
    return np.stack(out, axis=0)


def design_cost(freqs: Sequence[float], s21_K: np.ndarray,
                specs: List[FilterStateSpec],
                agg: str = "soft_max", temp: float = 2.0) -> float:
    spec_by_idx = {s.state_index: s for s in specs}
    costs = []
    for k in range(s21_K.shape[0]):
        spec = spec_by_idx.get(k, specs[min(k, len(specs) - 1)])
        costs.append(compute_state_cost(list(freqs), list(s21_K[k]), spec))
    return float(_aggregate_state_costs(costs, agg, temp))


def design_score(freqs: Sequence[float], s21_K: np.ndarray,
                 specs: List[FilterStateSpec],
                 agg: str = "soft_max", temp: float = 2.0) -> float:
    """Ranking score (higher = better) = negative aggregated GA cost."""
    return -design_cost(freqs, s21_K, specs, agg, temp)


# ---------------------------------------------------------------------------
# Normalization for the conditioning MLP
# ---------------------------------------------------------------------------

@dataclass
class ScoreNorm:
    """Per-scalar affine normalization: (x - center) / scale ~ O(1)."""
    centers: np.ndarray   # (n_scalars,)
    scales: np.ndarray    # (n_scalars,)

    @classmethod
    def from_config(cls, norm_cfg: dict,
                    scalars: Sequence[str] = SCORE_SCALARS) -> "ScoreNorm":
        c, s = [], []
        for k in scalars:
            entry = norm_cfg.get(k, {"center": 0.0, "scale": 1.0})
            c.append(float(entry["center"]))
            s.append(float(entry["scale"]))
        return cls(np.array(c, np.float32), np.array(s, np.float32))

    def normalize(self, v: np.ndarray) -> np.ndarray:
        return (v - self.centers) / self.scales

    def denormalize(self, v: np.ndarray) -> np.ndarray:
        return v * self.scales + self.centers


# ---------------------------------------------------------------------------
# Per-state running best (from SIMULATED designs only)
# ---------------------------------------------------------------------------

class PerStateBest:
    """Tracks the best achieved value of each scalar, INDEPENDENTLY per state.

    pv_pb_level / pv_contrast: keep the max; pv_pb_std: keep the min.  Per-state
    bests are decoupled (may come from different designs) — they only form a
    target signal, never a realized design.
    """

    def __init__(self, K: int, scalars: Sequence[str] = SCORE_SCALARS) -> None:
        self.scalars = list(scalars)
        self.K = K
        self._best: np.ndarray | None = None  # (K, n_scalars)

    def update(self, score_matrix: np.ndarray) -> None:
        sm = np.asarray(score_matrix, dtype=np.float32)
        if self._best is None:
            self._best = sm.copy()
            return
        for j, name in enumerate(self.scalars):
            if DIRECTION.get(name, "max") == "max":
                self._best[:, j] = np.maximum(self._best[:, j], sm[:, j])
            else:
                self._best[:, j] = np.minimum(self._best[:, j], sm[:, j])

    @property
    def has_data(self) -> bool:
        return self._best is not None

    def best_matrix(self) -> np.ndarray | None:
        return None if self._best is None else self._best.copy()


# ---------------------------------------------------------------------------
# Training-time score tracker (per-state best + best overall design score)
# ---------------------------------------------------------------------------

class TrainScoreTracker:
    """Helper trackers updated per minibatch during SFT.

    Tracks two quantities derived from each example's achieved S21:

      1. **Per-state best** (`PerStateBest`): the best value of each scalar in
         each state, INDEPENDENT across states (max for higher-is-better
         scalars, min for ripple-style scalars).  Captures the "envelope" of
         what's been seen.
      2. **Best overall** (`design_score`): the highest aggregated GA score
         across all examples seen so far -- a single scalar.  Also remembers
         the per-state score matrix and the global step where that best landed.

    The tracker maintains running counts so the snapshot includes the number of
    examples processed (useful for spotting epoch-rollover effects).
    """

    def __init__(self, K: int, specs, freqs: Sequence[float],
                 scalars: Sequence[str] = SCORE_SCALARS,
                 agg: str = "soft_max", agg_temp: float = 2.0) -> None:
        self.K = K
        self.specs = specs
        self.freqs = np.asarray(freqs)
        self.scalars = list(scalars)
        self.agg = agg
        self.agg_temp = agg_temp
        self.per_state_best = PerStateBest(K, scalars)
        self.best_overall: float = float("-inf")
        self.best_overall_state_matrix: np.ndarray | None = None  # (K, n_sc)
        self.best_overall_step: int = -1
        self.n_examples: int = 0

    def update_from_s21(self, s21_K: np.ndarray, step: int) -> None:
        """One example: (K, n_freq) S21 in dB."""
        sm = design_score_matrix(self.freqs, np.asarray(s21_K),
                                 self.specs, self.scalars)
        self.per_state_best.update(sm)
        sc = design_score(self.freqs, np.asarray(s21_K),
                          self.specs, self.agg, self.agg_temp)
        if sc > self.best_overall:
            self.best_overall = float(sc)
            self.best_overall_state_matrix = sm.copy()
            self.best_overall_step = step
        self.n_examples += 1

    def update_batch_from_examples(self, examples, step: int) -> None:
        """Convenience: update from a batch of TrainingExample objects."""
        for ex in examples:
            self.update_from_s21(ex.sparams.s21_db, step)

    def snapshot(self) -> dict:
        """Plain-dict snapshot for printing / checkpointing."""
        psb = self.per_state_best.best_matrix()
        return {
            "n_examples": self.n_examples,
            "scalars": list(self.scalars),
            "per_state_best": None if psb is None else psb.tolist(),
            "best_overall": self.best_overall,
            "best_overall_step": self.best_overall_step,
            "best_overall_state_matrix": (
                None if self.best_overall_state_matrix is None
                else self.best_overall_state_matrix.tolist()),
        }

    def format_short(self) -> str:
        """One-line summary suitable for the SFT progress print."""
        if self.n_examples == 0:
            return "trk: no examples yet"
        psb = self.per_state_best.best_matrix()
        per_state = " | ".join(
            "S{0}({1})".format(k, ",".join(
                f"{psb[k, j]:+.2f}" for j in range(len(self.scalars))))
            for k in range(self.K))
        return (f"best_overall={self.best_overall:+.3f}@step{self.best_overall_step} "
                f"| per_state_best: {per_state}")
