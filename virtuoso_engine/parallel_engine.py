"""
Parallel Virtuoso simulation engine — producer/consumer design.

Creates N independent Virtuoso workers, each fully isolated with:
  - Its own library name: ``{base_lib}_w{id}``
  - Its own output folder: ``{base_output_dir}/worker_{id}/``
  - Its own Virtuoso process (PersistentVirtuosoRunner)

Tasks are submitted to a shared ``queue.Queue`` and consumed by worker
threads.  Each worker runs one task at a time; multiple workers run
concurrently.

Configuration
-------------
Add a ``parallel`` section to your config YAML::

    parallel:
      num_workers: 4   # number of concurrent Virtuoso instances

The number of workers can also be set at construction time via the
``num_workers`` argument (takes precedence over the config file value).

Usage::

    from virtuoso_engine import ParallelVirtuosoEngine

    # Context-manager (recommended — ensures clean shutdown)
    with ParallelVirtuosoEngine("config.yaml") as engine:
        results = engine.run_batch([
            {"pixelated": {"state_list": [[0, 0]]}},
            {"pixelated": {"state_list": [[1, 0]]}},
            {"pixelated": {"state_list": [[0, 1]]}},
            {"pixelated": {"state_list": [[1, 1]]}},
        ])

    # Submit individual tasks and collect futures
    with ParallelVirtuosoEngine("config.yaml", num_workers=2) as engine:
        futures = [engine.submit(ov) for ov in overrides]
        results = [f.result() for f in futures]
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .engine import VirtuosoPixelEngine, RunResult


# ---------------------------------------------------------------------------
# Internal task container
# ---------------------------------------------------------------------------

@dataclass
class _Task:
    config_override: Optional[Dict]
    future: "Future[RunResult]"


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class _WorkerThread(threading.Thread):
    """
    A single Virtuoso worker.  Pulls tasks from ``task_queue``, executes them
    with a dedicated ``VirtuosoPixelEngine`` (and therefore a dedicated
    Virtuoso process), and writes results to each task's Future.
    """

    def __init__(
        self,
        worker_id: int,
        config_path: Path,
        task_queue: "queue.Queue[Optional[_Task]]",
    ) -> None:
        super().__init__(name=f"virt-worker-{worker_id}", daemon=True)
        self.worker_id   = worker_id
        self.config_path = config_path
        self.task_queue  = task_queue
        self._engine: Optional[VirtuosoPixelEngine] = None

    def run(self) -> None:
        self._engine = VirtuosoPixelEngine(self.config_path, worker_id=self.worker_id)
        print(f"[Worker-{self.worker_id}] Ready.")
        try:
            while True:
                task = self.task_queue.get()
                if task is None:          # shutdown sentinel
                    break
                try:
                    result = self._engine.run(task.config_override)
                    task.future.set_result(result)
                except Exception as exc:  # noqa: BLE001
                    task.future.set_exception(exc)
        finally:
            if self._engine is not None:
                self._engine.close()
            print(f"[Worker-{self.worker_id}] Stopped.")


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------

class ParallelVirtuosoEngine:
    """
    Producer/consumer parallel Virtuoso simulation engine.

    Starts ``num_workers`` independent Virtuoso instances.  Tasks submitted
    via :meth:`submit` are processed concurrently across all workers.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file (same format as
        :class:`VirtuosoPixelEngine`).
    num_workers:
        Number of concurrent Virtuoso workers.  Overrides the
        ``parallel.num_workers`` value in the config file when provided.
    """

    def __init__(
        self,
        config_path: str | Path,
        num_workers: Optional[int] = None,
    ) -> None:
        self.config_path  = Path(config_path)
        self._num_workers = num_workers
        self._task_queue: queue.Queue[Optional[_Task]] = queue.Queue()
        self._workers: List[_WorkerThread] = []
        self._started = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ParallelVirtuosoEngine":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all worker threads (each launches its own Virtuoso process)."""
        if self._started:
            return

        with self.config_path.open() as f:
            cfg = yaml.safe_load(f)
        n = self._num_workers or cfg.get("parallel", {}).get("num_workers", 1)

        for worker_id in range(n):
            w = _WorkerThread(
                worker_id=worker_id,
                config_path=self.config_path,
                task_queue=self._task_queue,
            )
            w.start()
            self._workers.append(w)

        self._started = True
        print(f"[ParallelEngine] Started {n} worker(s).")

    def close(self) -> None:
        """Send shutdown sentinels and wait for all workers to stop."""
        for _ in self._workers:
            self._task_queue.put(None)
        for w in self._workers:
            w.join()
        self._workers.clear()
        self._started = False
        print("[ParallelEngine] All workers stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        config_override: Optional[Dict] = None,
    ) -> "Future[RunResult]":
        """
        Submit a single simulation task.

        Parameters
        ----------
        config_override:
            Nested dict deep-merged over the base config (same semantics as
            :meth:`VirtuosoPixelEngine.run`).

        Returns
        -------
        concurrent.futures.Future
            Resolves to a :class:`RunResult` when the task completes.
        """
        if not self._started:
            raise RuntimeError("Call start() or use as a context manager first.")
        fut: Future[RunResult] = Future()
        self._task_queue.put(_Task(config_override=config_override, future=fut))
        return fut

    def run_batch(
        self,
        config_overrides: List[Optional[Dict]],
    ) -> List[RunResult]:
        """
        Submit a list of tasks and block until all complete.

        Results are returned in the same order as ``config_overrides``.

        Parameters
        ----------
        config_overrides:
            List of config override dicts (``None`` means use the base config
            unchanged).

        Returns
        -------
        List[RunResult]
            One result per override, in submission order.
        """
        futures = [self.submit(ov) for ov in config_overrides]
        return [f.result() for f in futures]
