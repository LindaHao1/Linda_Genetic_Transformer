"""
Dataset: TrainingExamples + parsing real sims + a synthetic generator.

Hindsight relabeling note: the SFT *spec* for an example is its OWN achieved
S21 (`TrainingExample.sparams.s21_db`), not an intended target.  The dataset
just stores examples; the training scripts read `sparams` as the spec.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import time
from typing import Iterator, List, Optional, Sequence

import numpy as np

from src.tokenizer import (
    P_GROUND, P_INDUCTOR, P_METAL, P_SWITCH, TokenizerConfig,
)
from src.types import Design, SParams, TrainingExample

log = logging.getLogger(__name__)

_WM_DIR = "/disk/mcao53/RFIC_RL/world_model_test"
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)


# ---------------------------------------------------------------------------
# Real sims -> TrainingExample
# ---------------------------------------------------------------------------

def _flatten_caps(cap_values) -> List[float]:
    """Real configs may bank caps ([n_sw][n_bank]); encode.py wants flat.
    TODO(user): banked switches -- we take bank[0] (assume bank size 1)."""
    out = []
    for v in cap_values:
        out.append(float(v[0]) if isinstance(v, (list, tuple)) else float(v))
    return out


def _reduce_states(switch_states, n_sw: int) -> List[List[int]]:
    """state_list may be [K][n_sw][n_bank]; reduce inner bank to one bit."""
    red = []
    for row in switch_states:
        bits = []
        for k in range(n_sw):
            cell = row[k] if k < len(row) else 0
            bits.append(int(any(cell)) if isinstance(cell, (list, tuple))
                        else int(cell))
        red.append(bits)
    return red


def build_from_sims(
    sims_dirs: Sequence[str],
    freq_points: np.ndarray,
    limit: Optional[int] = None,
    min_states: Optional[int] = None,
    seed: int = 0,
) -> "LMDataset":
    """Parse design folders into simulated TrainingExamples.

    Reuses world_model_test parsing/S21 loaders (single source of truth).

    All `K_avail` states per design are retained (switch/inductor states + the
    matching .s2p files), so that downstream `collate` can resample which `K`
    states the LM sees on each minibatch.  Designs with fewer than `min_states`
    states are skipped (default: keep all).

    `seed` is retained for backward compat (cache header); no longer used at
    build time now that no random subsample is taken here.
    """
    del seed  # no random subsample at build time anymore
    from process_dataset import _load_records          # noqa: E402
    from wm_dataset import _load_s21                    # noqa: E402

    examples: List[TrainingExample] = []
    n_skipped = 0
    for sd in sims_dirs:
        for rec in _load_records(sd):
            matrix = rec["matrix"]
            n_sw = sum(v == P_SWITCH for row in matrix for v in row)
            n_ind = sum(v == P_INDUCTOR for row in matrix for v in row)
            K_avail = len(rec["switch_states"])

            if min_states is not None and K_avail < min_states:
                n_skipped += 1
                continue

            sw_full = _reduce_states(rec["switch_states"], n_sw)
            isw_full = rec.get("inductor_switch_states", [])
            if not isw_full:                             # [] => all inductors ON
                isw_full = [[1] * n_ind for _ in range(K_avail)]
            s2p_files = rec["s_param_files"][:K_avail]   # file i == state i

            design = Design(
                pixel_grid=matrix,
                switch_caps=_flatten_caps(rec.get("cap_values", [])),
                inductor_values=list(rec.get("inductor_values", [])),
                inductor_cap_values=list(rec.get("inductor_cap_values", [])),
                cap_pixel_pairs=rec.get("cap_pixel_pairs", []),
                cap_pair_values=list(rec.get("cap_pair_values", [])),
                switch_states=sw_full,                   # ALL K_avail states
                inductor_switch_states=isw_full,         # ALL K_avail states
            )
            s21 = np.stack([_load_s21(s2p_files[i], freq_points)
                            for i in range(K_avail)], axis=0)
            examples.append(TrainingExample(
                design=design,
                sparams=SParams(freq_hz=freq_points, s21_db=s21),
                source="simulated",
                iteration_added=0,
            ))
            if limit and len(examples) >= limit:
                log.info("Loaded %d simulated examples (limit hit)",
                         len(examples))
                return LMDataset(examples)
    log.info("Loaded %d simulated examples (%d skipped: < %s states); "
             "all K_avail states retained for per-collate resampling",
             len(examples), n_skipped, min_states)
    return LMDataset(examples)


def build_from_sims_cached(
    sims_dirs: Sequence[str],
    freq_points: np.ndarray,
    cache_path: str,
    limit: Optional[int] = None,
    min_states: Optional[int] = None,
    refresh: bool = False,
) -> "LMDataset":
    """Same as `build_from_sims` but pickles the parsed dataset to `cache_path`.

    Cache layout (format_version=2): each cached example retains ALL `K_avail`
    states.  Per-minibatch `K`-state resampling happens in `src/training.py:
    collate` (`state_rng` arg), giving true epoch-to-epoch state diversity.

    On subsequent runs (or other scripts that read the same sims), the cache is
    loaded directly -- skipping the per-design YAML + s2p parse.  The cache is
    invalidated automatically if the header doesn't match (sims_dirs,
    freq_points, limit, min_states, format_version); set `refresh=True` to
    rebuild on demand.

    The cache is a pickle of `{"header": ..., "examples": [...]}`.
    """
    import hashlib
    header = {
        "sims_dirs": [str(p) for p in sims_dirs],
        "freq_points_hash": hashlib.sha1(
            np.asarray(freq_points, dtype=np.float64).tobytes()).hexdigest(),
        "limit": limit,
        "min_states": min_states,
        "format_version": 2,
    }
    if not refresh and os.path.isfile(cache_path):
        t0 = time.time()
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("header") == header:
            log.info("Loaded %d cached examples from %s in %.1fs",
                     len(blob["examples"]), cache_path, time.time() - t0)
            return LMDataset(blob["examples"])
        log.info("Cache header mismatch at %s -- rebuilding", cache_path)

    ds = build_from_sims(sims_dirs, freq_points, limit=limit,
                         min_states=min_states)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    t0 = time.time()
    with open(cache_path, "wb") as f:
        pickle.dump({"header": header, "examples": ds.examples}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Wrote %d examples to cache %s in %.1fs",
             len(ds), cache_path, time.time() - t0)
    return ds


# ---------------------------------------------------------------------------
# Synthetic feasible-by-construction designs (smoke tests / cold start)
# ---------------------------------------------------------------------------

def make_random_design(cfg: TokenizerConfig,
                        rng: Optional[np.random.Generator] = None) -> Design:
    """A grammar-valid design: metal backbone + 1 switch + 1 inductor + 1 cap
    pair (metal endpoints, all-ground interior).  Not EM-optimized; only used
    to bootstrap/smoke-test the pipeline.

    TODO(user): for real cold-start data use
    virtuoso_optimization_SA/feasibility.py's constrained generator.
    """
    rng = rng or np.random.default_rng()
    R, C, K = cfg.rows, cfg.cols, cfg.n_states
    m = [[P_GROUND] * C for _ in range(R)]
    mid = R // 2
    for c in range(C):                       # metal backbone (connectivity + ports)
        m[mid][c] = P_METAL
    sw_c = max(2, C // 3)
    ind_c = min(C - 3, C - 1 - C // 3)
    m[mid][sw_c] = P_SWITCH
    m[mid][ind_c] = P_INDUCTOR
    # cap pair on a separate row: metal head, ground interior, metal dest
    pr = 1 if mid != 1 else R - 2
    L = cfg.cap_pair_min_dist
    hc = 1
    if hc + L < C:
        m[pr][hc] = P_METAL
        m[pr][hc + L] = P_METAL
        cap_pairs = [[[pr, hc], [pr, hc + L]]]
        cap_pair_vals = [float(rng.uniform(cfg.cap_pair_min, cfg.cap_pair_max))]
    else:
        cap_pairs, cap_pair_vals = [], []

    sw_states = [[int(rng.integers(0, 2))] for _ in range(K)]
    ind_states = [[int(rng.integers(0, 2))] for _ in range(K)]
    return Design(
        pixel_grid=m,
        switch_caps=[float(rng.uniform(cfg.cap_min, cfg.cap_max))],
        inductor_values=[0],
        inductor_cap_values=[float(rng.uniform(cfg.inductor_cap_min,
                                               cfg.inductor_cap_max))],
        cap_pixel_pairs=cap_pairs,
        cap_pair_values=cap_pair_vals,
        switch_states=sw_states,
        inductor_switch_states=ind_states,
    )


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

class LMDataset:
    def __init__(self, examples: Optional[List[TrainingExample]] = None) -> None:
        self.examples: List[TrainingExample] = list(examples or [])

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self.examples)

    def __getitem__(self, i: int) -> TrainingExample:
        return self.examples[i]

    def add(self, ex: TrainingExample) -> None:
        self.examples.append(ex)

    def add_relabeled(self, designs: Sequence[Design],
                      true_sparams: Sequence[SParams],
                      iteration: int = 0) -> None:
        """Add simulator-relabeled designs to the cumulative real dataset."""
        for d, sp in zip(designs, true_sparams):
            self.examples.append(TrainingExample(
                design=d, sparams=sp, source="simulated",
                iteration_added=iteration))

    def real_only(self) -> "LMDataset":
        return LMDataset([e for e in self.examples if e.source == "simulated"])
