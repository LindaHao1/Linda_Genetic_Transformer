"""
Simulator interface — PLACEHOLDER.

The real EM simulator is wired via `ParallelVirtuosoEngine`
(see virtuoso_optimization_GA/ga_optimizer.py:1104, which builds the engine and
calls `engine.simulate_designs(...)` returning per-state .s2p files).

`run_simulator` raises NotImplementedError as required by the prompt.  For
development the surrogate-backed fallback lets the whole pipeline (incl. the
recalibration branch) run end to end.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from src.surrogate import Surrogate
from src.types import Design, SParams


def run_simulator(designs: Sequence[Design]) -> List[SParams]:
    """PLACEHOLDER — wire to the real EM simulator.

    TODO(user): construct ParallelVirtuosoEngine(config_file, output_dir,
    n_parallel) and call engine.simulate_designs(designs); read the per-state
    .s2p files (skrf, S21 dB) into SParams.  See ga_optimizer.py:1104+.
    """
    raise NotImplementedError("Wire to real simulator (see docstring / ga_optimizer.py)")


def run_simulator_dev(
    designs: Sequence[Design],
    surrogate: Surrogate,
    freq_hz: np.ndarray,
    noise_db: float = 0.5,
    seed: Optional[int] = None,
) -> List[SParams]:
    """Dev stand-in: surrogate prediction + small Gaussian noise (dB).

    Lets `test_pipeline_dryrun.py` exercise the recalibration branch without a
    real simulator.  NOT physically meaningful.
    """
    rng = np.random.default_rng(seed)
    means, _ = surrogate.predict_multistate_batch(designs)
    out: List[SParams] = []
    for m in means:
        noisy = m + rng.normal(0.0, noise_db, size=m.shape).astype(np.float32)
        out.append(SParams(freq_hz=np.asarray(freq_hz), s21_db=noisy))
    return out
