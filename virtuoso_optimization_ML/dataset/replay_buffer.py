"""
replay_buffer.py — Experience replay buffer for the online ML optimizer.

Each entry stores an encoded design image, the associated S-parameter tensor
(real simulation or world model prediction), and metadata used for priority
sampling and world model correction.

Feasibility labeling
--------------------
Infeasible decoded designs are added with is_feasible=False and their s_params
set to INFEASIBLE_SENTINEL (all-ones tensor) — an impossible value for a
passive device.  This provides gradient signal to the generative model
discouraging infeasible regions without a separate loss term.

Priority sampling (for generative model training)
---------------------------------------------------
  priority = alpha_recency * exp(-decay * (current_iter - entry.iteration))
           + alpha_real    * real_bonus  (only if is_real=True)
  Infeasible entries receive an additional factor of infeasible_weight.

World model training uses only real + feasible entries (uniform sampling).
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import math
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Infeasibility sentinel
# ---------------------------------------------------------------------------

def make_infeasible_sentinel(n_states: int, n_freq: int) -> torch.Tensor:
    """
    Return an all-ones S-parameter tensor used to mark infeasible designs.
    +1 (linear scale) is impossible for a passive 2-port device where
    |S21|^2 ≤ 1, i.e. S21_dB ≤ 0 dB.  In the dB domain we use +1 dB as
    sentinel (still positive and well outside any valid range).
    """
    return torch.ones(n_states, n_freq, dtype=torch.float32)


INFEASIBLE_SENTINEL_VALUE = 1.0   # dB value used as sentinel


# ---------------------------------------------------------------------------
# Buffer entry
# ---------------------------------------------------------------------------

@dataclass
class BufferEntry:
    image:      torch.Tensor   # (1, H, W+S)  float32 encoded design image
    s_params:   torch.Tensor   # (n_states, n_freq)  S21 in dB (or sentinel)
    is_real:    bool           # True if labeled by real simulator
    is_feasible: bool          # False → s_params is sentinel; skip world model
    iteration:  int            # training iteration when added
    confidence: float          # world model confidence at insertion (0.0=real)
    entry_id:   int            # monotonically increasing unique ID


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer with priority sampling.

    Parameters
    ----------
    max_size        : maximum number of entries (oldest evicted when full).
    alpha_recency   : weight of recency term in priority.
    alpha_real      : weight of real-data bonus term in priority.
    real_bonus      : additive bonus for real simulation entries.
    decay_rate      : exponential decay rate for recency (per iteration).
    infeasible_weight : multiplier for infeasible entries in gen model sampling.
    n_states        : number of filter states (for sentinel construction).
    n_freq          : number of frequency points (for sentinel construction).
    seed            : optional RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        max_size:           int,
        alpha_recency:      float = 0.6,
        alpha_real:         float = 0.4,
        real_bonus:         float = 2.0,
        decay_rate:         float = 0.05,
        infeasible_weight:  float = 0.1,
        n_states:           int   = 2,
        n_freq:             int   = 201,
        seed:               Optional[int] = None,
    ) -> None:
        self.max_size          = max_size
        self.alpha_recency     = alpha_recency
        self.alpha_real        = alpha_real
        self.real_bonus        = real_bonus
        self.decay_rate        = decay_rate
        self.infeasible_weight = infeasible_weight
        self.n_states          = n_states
        self.n_freq            = n_freq
        self._rng              = random.Random(seed)
        self._entries: List[BufferEntry] = []
        self._next_id: int = 0
        self._id_to_idx: dict = {}   # entry_id → list index (updated on eviction)

    # ------------------------------------------------------------------
    # Adding entries
    # ------------------------------------------------------------------

    def add(
        self,
        image:       torch.Tensor,
        s_params:    torch.Tensor,
        is_real:     bool,
        is_feasible: bool,
        iteration:   int,
        confidence:  float = 0.0,
    ) -> int:
        """
        Add an entry to the buffer.  Returns the assigned entry_id.

        If the buffer is full, the oldest entry is evicted (FIFO).
        """
        entry_id = self._next_id
        self._next_id += 1

        entry = BufferEntry(
            image       = image.detach().cpu(),
            s_params    = s_params.detach().cpu(),
            is_real     = is_real,
            is_feasible = is_feasible,
            iteration   = iteration,
            confidence  = confidence,
            entry_id    = entry_id,
        )

        if len(self._entries) >= self.max_size:
            evicted = self._entries.pop(0)
            self._id_to_idx.pop(evicted.entry_id, None)

        idx = len(self._entries)
        self._entries.append(entry)
        self._id_to_idx[entry_id] = idx
        self._rebuild_id_map()   # rebuild after potential eviction shift
        return entry_id

    def _rebuild_id_map(self) -> None:
        self._id_to_idx = {e.entry_id: i for i, e in enumerate(self._entries)}

    # ------------------------------------------------------------------
    # Sampling — generative model (all entries, priority weighted)
    # ------------------------------------------------------------------

    def sample_for_generative(
        self,
        batch_size:   int,
        current_iter: int,
    ) -> List[BufferEntry]:
        """
        Sample with priority weighting for generative model training.
        Infeasible entries receive infeasible_weight × priority.
        Returns at most min(batch_size, len(buffer)) entries.
        """
        if not self._entries:
            return []

        priorities = np.array([
            self._priority(e, current_iter) for e in self._entries
        ], dtype=np.float64)

        total = priorities.sum()
        if total <= 0:
            probs = np.ones(len(self._entries)) / len(self._entries)
        else:
            probs = priorities / total

        n = min(batch_size, len(self._entries))
        indices = self._rng.choices(range(len(self._entries)), weights=probs.tolist(), k=n)
        return [self._entries[i] for i in indices]

    def _priority(self, entry: BufferEntry, current_iter: int) -> float:
        age    = max(current_iter - entry.iteration, 0)
        p      = self.alpha_recency * math.exp(-self.decay_rate * age)
        if entry.is_real:
            p += self.alpha_real * self.real_bonus
        if not entry.is_feasible:
            p *= self.infeasible_weight
        return max(p, 1e-9)

    # ------------------------------------------------------------------
    # Sampling — world model (real + feasible only, uniform)
    # ------------------------------------------------------------------

    def sample_for_world_model(self, batch_size: int) -> List[BufferEntry]:
        """
        Sample real + feasible entries uniformly for world model training.
        Returns empty list if no qualifying entries exist.
        """
        pool = [e for e in self._entries if e.is_real and e.is_feasible]
        if not pool:
            return []
        n = min(batch_size, len(pool))
        return self._rng.choices(pool, k=n)

    # ------------------------------------------------------------------
    # Sampling — correction step (high-confidence real entries)
    # ------------------------------------------------------------------

    def sample_high_confidence_real(
        self,
        n:                    int,
        confidence_threshold: float,
    ) -> List[BufferEntry]:
        """
        Return up to n entries that were labeled by the world model
        (is_real=False, is_feasible=True) with confidence >= threshold.
        Used in the periodic correction step to re-simulate and retrain.
        """
        pool = [
            e for e in self._entries
            if (not e.is_real) and e.is_feasible and e.confidence >= confidence_threshold
        ]
        if not pool:
            return []
        n = min(n, len(pool))
        return self._rng.choices(pool, k=n)

    # ------------------------------------------------------------------
    # Update entry after re-simulation (correction step)
    # ------------------------------------------------------------------

    def update_entry(
        self,
        entry_id:   int,
        s_params:   torch.Tensor,
        is_real:    bool,
        confidence: float,
    ) -> bool:
        """
        Overwrite the s_params label of an existing entry in-place.
        Returns True if the entry was found, False otherwise.
        """
        idx = self._id_to_idx.get(entry_id)
        if idx is None:
            return False
        e = self._entries[idx]
        e.s_params   = s_params.detach().cpu()
        e.is_real    = is_real
        e.confidence = confidence
        return True

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def n_real(self) -> int:
        return sum(1 for e in self._entries if e.is_real and e.is_feasible)

    def n_infeasible(self) -> int:
        return sum(1 for e in self._entries if not e.is_feasible)

    def n_world_model(self) -> int:
        return sum(1 for e in self._entries if not e.is_real and e.is_feasible)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with Path(path).open("rb") as f:
            return pickle.load(f)
