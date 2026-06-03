"""
Async EM-simulation manager for the Phase-2 recalibration loop.

`AsyncSimManager` wraps `virtuoso_engine.parallel_engine.ParallelVirtuosoEngine`
and runs it as a long-lived background pool: the engine is started once before
the iterated loop and kept alive for its lifetime.  At a recalibration trigger,
selected designs are `submit`ted non-blocking; the loop keeps generating + doing
pool SFT while the simulations run.  As results finish over the next iterations,
`collect()` returns the completed (Design, SParams) pairs (the *realized* design
read back from the engine output, since cap values are snapped to a grid).

Usage::

    with AsyncSimManager(cfg_path, fpts, num_workers=8, output_root=...) as mgr:
        for it in range(n_iters):
            ...
            if recal: mgr.submit(to_sim, it)        # non-blocking
            for d, sp, it_sub in mgr.collect():     # whatever finished
                ingest(d, sp, it_sub)
        for d, sp, it_sub in mgr.drain():           # flush in-flight at the end
            ingest(d, sp, it_sub)
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.design_yaml import (build_pixelated_override,
                             load_design_and_s21_from_run_dir)
from src.types import Design, SParams

log = logging.getLogger(__name__)


class _PendingJob:
    __slots__ = ("design", "iteration", "future")

    def __init__(self, design: Design, iteration: int, future: "Future") -> None:
        self.design = design
        self.iteration = iteration
        self.future = future


class AsyncSimManager:
    """Non-blocking wrapper over ParallelVirtuosoEngine for async recalibration."""

    def __init__(
        self,
        engine_config_path: str,
        freq_hz: np.ndarray,
        num_workers: int,
        output_root: str,
        log_path: Optional[str] = None,
        quiet: bool = True,
    ) -> None:
        self.engine_config_path = str(engine_config_path)
        self.freq_hz = np.asarray(freq_hz)
        self.num_workers = int(num_workers)
        # MUST be absolute: Virtuoso runs with cwd=virtuoso_dir, so a relative
        # output dir (e.g. ./runs) would resolve under virtuoso_dir instead of
        # the launch dir and Virtuoso would fail to create its session log
        # ("CIW Initialization failure").
        self.output_root = Path(output_root).resolve()
        self._log_path = str(Path(log_path).resolve()) if log_path else log_path
        self._quiet = quiet
        self._engine = None                      # ParallelVirtuosoEngine
        self._pending: List[_PendingJob] = []    # in-flight jobs
        self._job_counter = 0                    # for unique per-job output dirs

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> "AsyncSimManager":
        from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._engine = ParallelVirtuosoEngine(
            self.engine_config_path,
            num_workers=self.num_workers,
            log_path=self._log_path,
            quiet=self._quiet,
        )
        self._engine.start()
        return self

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def __enter__(self) -> "AsyncSimManager":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def n_workers(self) -> int:
        return len(self._engine._workers) if self._engine else self.num_workers

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    # ---- submit (non-blocking) ------------------------------------------
    def submit(self, designs: Sequence[Design], iteration: int) -> int:
        """Submit designs to the engine; store futures. Returns # submitted."""
        if self._engine is None:
            raise RuntimeError("AsyncSimManager not started.")
        for d in designs:
            job_dir = self.output_root / f"job_{self._job_counter:06d}"
            self._job_counter += 1
            override = build_pixelated_override(
                d, output_parent_dir=str(job_dir))
            fut = self._engine.submit(override)
            self._pending.append(_PendingJob(d, iteration, fut))
        return len(designs)

    # ---- collect (non-blocking) -----------------------------------------
    def collect(self) -> List[Tuple[Design, SParams, int]]:
        """Return completed (design, true_sparams, submit_iteration) for finished
        jobs; leave unfinished pending.  Failed/empty sims are dropped + logged."""
        done, still = [], []
        for job in self._pending:
            if not job.future.done():
                still.append(job)
                continue
            sp = self._parse_future(job)
            if sp is not None:
                design, sparams = sp
                done.append((design, sparams, job.iteration))
        self._pending = still
        return done

    # ---- drain (blocking) — call once at loop end -----------------------
    def drain(self, timeout: Optional[float] = None
              ) -> List[Tuple[Design, SParams, int]]:
        out = []
        for job in self._pending:
            sp = self._parse_future(job, block=True, timeout=timeout)
            if sp is not None:
                design, sparams = sp
                out.append((design, sparams, job.iteration))
        self._pending = []
        return out

    # ---- internal: RunResult -> (realized Design, SParams) --------------
    def _parse_future(self, job: _PendingJob, block: bool = False,
                      timeout: Optional[float] = None
                      ) -> Optional[Tuple[Design, SParams]]:
        try:
            if not block and not job.future.done():
                return None
            res = job.future.result(timeout=timeout)
        except Exception as exc:
            log.warning("sim future failed (iter %d): %s", job.iteration, exc)
            return None
        if (not res.success) or (not res.s_param_files):
            log.warning("sim unsuccessful/empty (iter %d, cell %s)",
                        job.iteration, getattr(res, "cell_name", "?"))
            return None
        # Prefer the realized design read back from the engine output (cap values
        # are grid-snapped, so it differs from the submitted design).
        try:
            return load_design_and_s21_from_run_dir(res.run_dir, self.freq_hz)
        except Exception as exc:
            log.warning("read-back failed for %s (%s); falling back to submitted "
                        "design + globbed s2p", res.run_dir, exc)
        # Fallback: submitted design + s2p files reported by the engine.
        try:
            from wm_dataset import _load_s21
            K = job.design.n_states
            files = [f for f in res.s_param_files if f is not None][:K]
            if len(files) < K:
                log.warning("fallback: %d/%d state files; dropping",
                            len(files), K)
                return None
            s21 = np.stack([_load_s21(str(f), self.freq_hz) for f in files],
                           axis=0)
            return job.design, SParams(freq_hz=self.freq_hz, s21_db=s21)
        except Exception as exc:
            log.warning("fallback parse failed (iter %d): %s", job.iteration, exc)
            return None
