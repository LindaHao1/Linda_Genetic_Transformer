"""
replay_buffer.py — Experience replay buffer for the online ML optimizer.

Each entry stores an encoded design image, the associated S-parameter tensor
(real simulation or world model prediction), and metadata used for priority
sampling and world model correction.

Feasibility policy
------------------
Only feasible designs are stored in this buffer.  Infeasible decoded images
are collected separately in the trainer's ``infeasible_image_store`` (a plain
rolling deque) and used exclusively to train the FeasibilityClassifier.
The generative model is therefore never trained on infeasible data here.

Priority sampling (for generative model training)
---------------------------------------------------
Sampling is driven by three terms mixed via configurable weights:

  priority_i = alpha_perf   * perf_rank_i          (performance rank, 0–1)
             + alpha_real   * real_bonus            (if is_real=True)
             + alpha_random                         (uniform exploration floor)

where:
  perf_rank_i = (N - cost_rank_i) / (N - 1)
              = 1.0 for the best (lowest cost) scored entry,
                0.0 for the worst-scored entry,
                0.0 for entries with no finite score (score = inf).

This removes the previous recency bias that caused self-reinforcing behaviour
(recently generated designs dominating training) and instead prioritises:
  1. High-performance designs that were verified by the real simulator.
  2. A uniform exploration floor so all entries remain reachable.

World model training uses only real + feasible entries (uniform sampling).
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
    score:      float = float("inf")   # optimization cost (lower=better); inf if unknown


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer with priority sampling.

    Only feasible designs are stored.  Infeasible images are managed by the
    trainer's ``infeasible_image_store`` and never appear here.

    Parameters
    ----------
    max_size      : maximum number of entries (oldest evicted when full).
    alpha_recency : weight of recency term in priority.
    alpha_real    : weight of real-data bonus term in priority.
    real_bonus    : additive bonus for real simulation entries.
    decay_rate    : exponential decay rate for recency (per iteration).
    n_states      : number of filter states (for sentinel construction).
    n_freq        : number of frequency points (for sentinel construction).
    seed          : optional RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        max_size:      int,
        alpha_perf:    float = 0.6,
        alpha_real:    float = 0.4,
        real_bonus:    float = 2.0,
        alpha_random:  float = 0.2,
        n_states:      int   = 2,
        n_freq:        int   = 201,
        seed:          Optional[int] = None,
        # Legacy parameters kept for backward compatibility (no longer used)
        alpha_recency: float = 0.0,
        decay_rate:    float = 0.0,
    ) -> None:
        self.max_size     = max_size
        self.alpha_perf   = alpha_perf
        self.alpha_real   = alpha_real
        self.real_bonus   = real_bonus
        self.alpha_random = alpha_random
        self.n_states     = n_states
        self.n_freq       = n_freq
        self._rng         = random.Random(seed)
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
        score:       float = float("inf"),
    ) -> int:
        """
        Add an entry to the buffer.  Returns the assigned entry_id.

        Parameters
        ----------
        score : optimization cost for this design (lower is better).
                Pass the real simulator cost for real entries; leave as inf
                for world-model-labeled or infeasible entries.

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
            score       = score,
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
        Sample feasible entries with priority weighting for generative model training.

        Priority = alpha_perf   * perf_rank          (performance, 0–1)
                 + alpha_real   * real_bonus          (if is_real=True)
                 + alpha_random                       (uniform exploration floor)

        perf_rank is normalised rank in [0, 1] based on cost: 1.0 = best (lowest)
        cost entry, 0.0 = worst or unscored (score = inf) entry.  Entries with
        no finite score receive perf_rank = 0 but are still reachable via
        the alpha_random floor.

        current_iter is accepted for interface compatibility but is no longer used
        (recency bias has been removed).
        """
        feasible = [e for e in self._entries if e.is_feasible]
        if not feasible:
            return []

        m = len(feasible)

        # --- Performance rank: rank by cost, lower cost → higher rank ---
        finite_idx = [i for i, e in enumerate(feasible) if e.score < float("inf")]
        perf_rank  = [0.0] * m          # default: 0 for unscored entries
        if finite_idx:
            n_finite  = len(finite_idx)
            # Sort ascending by score (lowest cost = best = rank n_finite-1)
            sorted_fi = sorted(finite_idx, key=lambda i: feasible[i].score)
            for rank_pos, orig_idx in enumerate(sorted_fi):
                # rank_pos 0 = best → normalised rank 1.0
                perf_rank[orig_idx] = 1.0 - rank_pos / max(n_finite - 1, 1)

        # Read weights; fall back to defaults for buffers loaded from old checkpoints
        alpha_perf   = getattr(self, "alpha_perf",   0.6)
        alpha_real   = getattr(self, "alpha_real",   0.4)
        real_bonus   = getattr(self, "real_bonus",   2.0)
        alpha_random = getattr(self, "alpha_random", 0.2)

        # --- Combine terms ---
        priorities = np.array([
            alpha_perf   * perf_rank[i]
            + (alpha_real * real_bonus if feasible[i].is_real else 0.0)
            + alpha_random
            for i in range(m)
        ], dtype=np.float64)

        probs = priorities / priorities.sum()
        n       = min(batch_size, m)
        indices = self._rng.choices(range(m), weights=probs.tolist(), k=n)
        return [feasible[i] for i in indices]

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
    # Sampling — best known real designs (for WM correction priority)
    # ------------------------------------------------------------------

    def sample_best_real(self, n: int) -> List[BufferEntry]:
        """Return up to n real feasible entries with the lowest (best) scores.

        Used in the WM correction step to include known-best designs in the
        fine-tuning batch so the world model does not 'forget' them.
        Only entries with a finite score (i.e., scored by the real simulator)
        are considered.
        """
        pool = [
            e for e in self._entries
            if e.is_real and e.is_feasible and e.score < float("inf")
        ]
        if not pool:
            return []
        pool.sort(key=lambda e: e.score)
        return pool[:n]

    # ------------------------------------------------------------------
    # Update entry after re-simulation (correction step)
    # ------------------------------------------------------------------

    def update_entry(
        self,
        entry_id:   int,
        s_params:   torch.Tensor,
        is_real:    bool,
        confidence: float,
        score:      float = float("inf"),
    ) -> bool:
        """
        Overwrite the s_params label of an existing entry in-place.
        Also updates the score field if a finite value is provided.
        Returns True if the entry was found, False otherwise.
        """
        idx = self._id_to_idx.get(entry_id)
        if idx is None:
            return False
        e = self._entries[idx]
        e.s_params   = s_params.detach().cpu()
        e.is_real    = is_real
        e.confidence = confidence
        e.score      = score
        return True

    # ------------------------------------------------------------------
    # Sampling — correction step (priority-ranked WM-labeled entries)
    # ------------------------------------------------------------------

    def sample_for_correction(
        self,
        n:                    int,
        confidence_threshold: float,
    ) -> List[BufferEntry]:
        """
        Return up to n entries that were labeled by the world model
        (is_real=False, is_feasible=True) with confidence >= threshold,
        ranked by a combined priority that favours high confidence AND low cost.

        Priority = 0.5 * normalised_confidence_rank
                 + 0.5 * normalised_score_rank_inverted

        where lower score = better rank (i.e. rank 0 = lowest cost).
        Entries with score=inf are treated as having the worst score rank.
        Returns the top n entries (deterministic, not random).
        If the pool is smaller than n, all entries are returned.
        """
        pool = [
            e for e in self._entries
            if (not e.is_real) and e.is_feasible and e.confidence >= confidence_threshold
        ]
        if not pool:
            return []

        m = len(pool)

        # --- Confidence rank (higher confidence → higher rank value) ---
        conf_sorted_idx = sorted(range(m), key=lambda i: pool[i].confidence)
        conf_rank = [0.0] * m
        for rank_pos, pool_idx in enumerate(conf_sorted_idx):
            conf_rank[pool_idx] = rank_pos / max(m - 1, 1)  # normalised to [0, 1]

        # --- Score rank (lower score → higher rank value; inf = worst) ---
        def _score_key(idx: int) -> float:
            s = pool[idx].score
            return s if s < float("inf") else float("inf")

        score_sorted_idx = sorted(range(m), key=_score_key)  # best (lowest) first
        score_rank = [0.0] * m
        for rank_pos, pool_idx in enumerate(score_sorted_idx):
            # rank_pos 0 → best score → gets highest normalised value (1.0)
            score_rank[pool_idx] = 1.0 - rank_pos / max(m - 1, 1)

        # --- Combined priority ---
        priorities = [
            0.5 * conf_rank[i] + 0.5 * score_rank[i]
            for i in range(m)
        ]

        # Sort descending by priority; return top n
        ranked = sorted(range(m), key=lambda i: priorities[i], reverse=True)
        top_n  = min(n, m)
        return [pool[i] for i in ranked[:top_n]]

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
