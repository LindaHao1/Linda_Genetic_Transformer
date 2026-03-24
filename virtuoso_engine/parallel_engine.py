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

import io
import queue
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import yaml

from .engine import VirtuosoPixelEngine, RunResult


# ---------------------------------------------------------------------------
# Thread-safe tee stream — mirrors stdout to a log file
# ---------------------------------------------------------------------------

class _TeeStream:
    """
    Wraps an output stream so that every write goes to both the original
    stream and a log file.  A lock makes it safe for concurrent threads.
    """

    def __init__(self, original: TextIO, log_file: TextIO) -> None:
        self._original = original
        self._log_file = log_file
        self._lock     = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            self._original.write(data)
            self._original.flush()
            self._log_file.write(data)
            self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            self._original.flush()
            self._log_file.flush()

    # Delegate everything else (e.g. .encoding, .fileno) to the original
    def __getattr__(self, name: str):
        return getattr(self._original, name)


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
    log_path:
        Where to save the full console log for this run.  Defaults to
        ``{parent_output_dir}/parallel_engine_{YYYYMMDD_HHMMSS}.log``.
        Pass ``None`` to disable file logging.
    """

    def __init__(
        self,
        config_path: str | Path,
        num_workers: Optional[int] = None,
        log_path: Optional[str | Path] = "auto",
    ) -> None:
        self.config_path  = Path(config_path)
        self._num_workers = num_workers
        self._log_path    = log_path      # "auto", a Path, or None
        self._task_queue: queue.Queue[Optional[_Task]] = queue.Queue()
        self._workers: List[_WorkerThread] = []
        self._started    = False
        self._log_file: Optional[io.TextIOWrapper] = None
        self._orig_stdout = None

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

        # --- Set up console logging ---
        if self._log_path is not None:
            if self._log_path == "auto":
                parent_dir = Path(cfg["project"]["parent_output_dir"])
                timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
                resolved_log = parent_dir / f"parallel_engine_{timestamp}.log"
            else:
                resolved_log = Path(self._log_path)
            resolved_log.parent.mkdir(parents=True, exist_ok=True)
            self._log_file   = resolved_log.open("w", buffering=1, encoding="utf-8")
            self._orig_stdout = sys.stdout
            sys.stdout        = _TeeStream(self._orig_stdout, self._log_file)
            # Write header so the log is self-describing
            sys.stdout.write(
                f"# ParallelVirtuosoEngine log\n"
                f"# Started : {datetime.now().isoformat(timespec='seconds')}\n"
                f"# Config  : {self.config_path}\n"
                f"# Workers : {n}\n"
                f"# Log file: {resolved_log}\n\n"
            )

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

        # --- Restore stdout and close log file ---
        if self._orig_stdout is not None:
            sys.stdout        = self._orig_stdout
            self._orig_stdout = None
        if self._log_file is not None:
            self._log_file.write(
                f"\n# Finished: {datetime.now().isoformat(timespec='seconds')}\n"
            )
            self._log_file.close()
            self._log_file = None

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
