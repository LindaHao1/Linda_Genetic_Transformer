"""
sa_optimizer.py — Simulated Annealing optimiser for the switchable-filter design.

Algorithm
---------
Each SA iteration:
  1. Generate N_parallel candidate designs from the current design using random
     SA actions.
  2. Simulate all N_parallel candidates in parallel using ParallelVirtuosoEngine.
  3. Evaluate the cost of each candidate (S-parameter vs. target specs).
  4. Pick the best-cost candidate.
  5. Accept or reject it using the Metropolis criterion:
       always accept if cost improves;
       accept with probability exp(-(new_cost - current_cost) / T) otherwise.
  6. Cool the temperature.

The optimizer saves:
  - A JSON progress log (``sa_progress.json``) updated after every iteration.
  - The best design found so far (``best_design.json``).
  - All Virtuoso simulation results inside ``{output_dir}/sa_iter_{i}/``.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .actions import random_action
from .cost import compute_total_cost, specs_from_config, FilterStateSpec
from .design import DesignState
from .feasibility import generate_random_design

# RunResult type hint only — the heavy virtuoso_engine import is deferred to
# _make_engine() so that importing this module does not require scikit-rf.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine
    from virtuoso_engine.engine import RunResult


# ---------------------------------------------------------------------------
# SA configuration container
# ---------------------------------------------------------------------------

@dataclass
class SAConfig:
    # Grid
    rows:              int
    cols:              int
    n_switches:        int
    max_inductors:     int
    n_states:          int
    cap_min:           float
    cap_max:           float
    n_inductor_values: int   # number of entries in the inductor_translation_table

    # SA algorithm
    T_start:        float
    T_end:          float
    n_iter:         int
    cooling:        str    # "geometric" or "linear"
    n_parallel:     int
    n_regular_flip: int
    cap_sigma:      float
    action_weights: List[float]  # 6-element weights: [flip_01, move_switch,
                                 #   flip_inductor, flip_inductor_value,
                                 #   perturb_caps, flip_switch_state]

    # Paths
    virtuoso_config: str   # path to the Virtuoso engine config YAML
    output_dir:      str   # base directory for all SA run outputs

    # Random seed
    seed: Optional[int]

    @classmethod
    def from_dict(cls, d: dict) -> "SAConfig":
        sa   = d["sa"]
        grid = d["grid"]
        return cls(
            rows               = int(grid["rows"]),
            cols               = int(grid["cols"]),
            n_switches         = int(grid["n_switches"]),
            max_inductors      = int(grid["max_inductors"]),
            n_states           = int(grid["n_states"]),
            cap_min            = float(grid["cap_min"]),
            cap_max            = float(grid["cap_max"]),
            n_inductor_values  = int(grid.get("n_inductor_values", 4)),

            T_start        = float(sa["T_start"]),
            T_end          = float(sa["T_end"]),
            n_iter         = int(sa["n_iter"]),
            cooling        = str(sa.get("cooling", "geometric")),
            n_parallel     = int(sa.get("n_parallel", 4)),
            n_regular_flip = int(sa.get("n_regular_flip", 3)),
            cap_sigma      = float(sa.get("cap_sigma", 50e-15)),
            action_weights = [float(w) for w in sa.get(
                "action_weights", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])],

            virtuoso_config = str(d["virtuoso_config"]),
            output_dir      = str(d["output_dir"]),

            seed = d.get("seed"),
        )


# ---------------------------------------------------------------------------
# Temperature schedules
# ---------------------------------------------------------------------------

def _temperature(T_start: float, T_end: float, n_iter: int,
                 step: int, cooling: str) -> float:
    if cooling == "linear":
        return T_start + (T_end - T_start) * step / max(n_iter - 1, 1)
    # geometric (default)
    ratio = (T_end / T_start) ** (1.0 / max(n_iter - 1, 1))
    return T_start * (ratio ** step)


# ---------------------------------------------------------------------------
# Main SA optimiser class
# ---------------------------------------------------------------------------

class SwitchableFilterSA:
    """
    Simulated annealing optimiser for the pixelated switchable filter.

    Usage::

        optimizer = SwitchableFilterSA("sa_config.yaml")
        best_design, best_cost = optimizer.run()
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with self.config_path.open() as f:
            raw = yaml.safe_load(f)

        self.cfg      = SAConfig.from_dict(raw)
        self.specs    = specs_from_config(raw["filter_specs"])
        self.rng      = random.Random(self.cfg.seed)

        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._progress: List[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Tuple[DesignState, float]:
        """
        Run the SA optimisation loop.

        Returns
        -------
        (best_design, best_cost)
        """
        cfg = self.cfg
        print(f"\n{'='*70}")
        print(f"[SA] Switchable Filter Optimiser")
        print(f"[SA] Grid: {cfg.rows}x{cfg.cols}, switches={cfg.n_switches}, "
              f"max_ind={cfg.max_inductors}, states={cfg.n_states}")
        print(f"[SA] T: {cfg.T_start:.1f} → {cfg.T_end:.4f}  "
              f"n_iter={cfg.n_iter}  n_parallel={cfg.n_parallel}")
        print(f"{'='*70}\n")

        # --- Initial design ---
        print("[SA] Generating initial random design ...")
        current = self._generate_initial_design()
        print(f"[SA] Initial design:\n{current}")

        # --- Single engine for initial evaluation + full SA loop ---
        start_time = time.time()

        with self._make_engine() as engine:
            current_cost = self._evaluate(engine, current, iter_idx=-1, tag="init")

            print(f"[SA] Initial cost: {current_cost:.4f}")

            best_design = current.copy()
            best_cost   = current_cost
            self._save_best(best_design, best_cost, iteration=-1)

            for step in range(cfg.n_iter):
                T = _temperature(cfg.T_start, cfg.T_end, cfg.n_iter, step, cfg.cooling)

                # Generate N_parallel candidates
                candidates = self._generate_candidates(current)
                if not candidates:
                    print(f"[SA] iter {step}: no valid candidates — skipping.")
                    continue

                # Evaluate all candidates in parallel
                costs = self._evaluate_batch(engine, candidates, step)

                # Pick best candidate
                best_cand_idx  = min(range(len(costs)), key=lambda i: costs[i])
                best_cand      = candidates[best_cand_idx]
                best_cand_cost = costs[best_cand_idx]

                # Metropolis acceptance
                delta = best_cand_cost - current_cost
                if delta < 0:
                    accept = True
                else:
                    accept_prob = math.exp(-delta / T) if T > 1e-30 else 0.0
                    accept = self.rng.random() < accept_prob

                if accept:
                    current      = best_cand
                    current_cost = best_cand_cost

                # Track global best
                if current_cost < best_cost:
                    best_design = current.copy()
                    best_cost   = current_cost
                    self._save_best(best_design, best_cost, step)

                # Progress logging
                elapsed = time.time() - start_time
                self._log_step(step, T, current_cost, best_cost, delta, accept, elapsed)
                self._save_progress()

                print(
                    f"[SA] iter {step:4d}/{cfg.n_iter}  T={T:.4f}  "
                    f"cost={current_cost:.4f}  best={best_cost:.4f}  "
                    f"{'ACCEPT' if accept else 'reject'}  "
                    f"(Δ={delta:+.4f})  {elapsed:.1f}s"
                )

        print(f"\n[SA] Finished. Best cost: {best_cost:.4f}")
        print(f"[SA] Best design:\n{best_design}")
        return best_design, best_cost

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    def _make_engine(self):
        # Deferred import: scikit-rf (required by virtuoso_engine.plotter) may not
        # be installed in all environments.  Import only when simulation is needed.
        from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine  # noqa: PLC0415
        return ParallelVirtuosoEngine(
            config_path = self.cfg.virtuoso_config,
            num_workers = self.cfg.n_parallel,
            log_path    = self.output_dir / "parallel_engine.log",
        )

    # ------------------------------------------------------------------
    # Initial design generation
    # ------------------------------------------------------------------

    def _generate_initial_design(self) -> DesignState:
        cfg = self.cfg
        design = generate_random_design(
            rows               = cfg.rows,
            cols               = cfg.cols,
            n_switches         = cfg.n_switches,
            max_inductors      = cfg.max_inductors,
            n_states           = cfg.n_states,
            cap_min            = cfg.cap_min,
            cap_max            = cfg.cap_max,
            rng                = self.rng,
            n_inductor_values  = cfg.n_inductor_values,
        )
        if design is None:
            raise RuntimeError(
                "Failed to generate a valid initial design after 10,000 attempts. "
                "Relax grid parameters (rows, cols, n_switches, max_inductors)."
            )
        return design

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(self, current: DesignState) -> List[DesignState]:
        """Generate up to N_parallel mutated candidates from the current design."""
        cfg = self.cfg
        candidates: List[DesignState] = []
        seen: set = set()

        max_tries = cfg.n_parallel * 20
        for _ in range(max_tries):
            if len(candidates) >= cfg.n_parallel:
                break
            candidate = random_action(
                state              = current,
                n_regular_flip     = cfg.n_regular_flip,
                max_inductors      = cfg.max_inductors,
                cap_min            = cfg.cap_min,
                cap_max            = cfg.cap_max,
                cap_sigma          = cfg.cap_sigma,
                rng                = self.rng,
                n_inductor_values  = cfg.n_inductor_values,
                action_weights     = cfg.action_weights,
            )
            if candidate is None:
                continue
            # Deduplicate by matrix + inductor_values fingerprint
            key = (
                tuple(v for row in candidate.matrix for v in row)
                + tuple(candidate.inductor_values)
            )
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

        return candidates

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _build_override(
        self,
        design: DesignState,
        iter_idx: int,
        cand_idx: int,
    ) -> dict:
        """
        Build a config override dict for VirtuosoPixelEngine.run().

        We route each simulation to its own subdirectory to avoid cell-name
        collisions between different designs that share grid dimensions.
        """
        tag        = f"iter{iter_idx}_cand{cand_idx}"
        output_dir = str(self.output_dir / "sims" / tag)
        return {
            "mode":      "pixelated_design_simple",
            "pixelated": design.to_pixelated_config(),
            "project":   {"parent_output_dir": output_dir},
        }

    def _evaluate(
        self,
        engine: ParallelVirtuosoEngine,
        design: DesignState,
        iter_idx: int,
        tag: str = "eval",
    ) -> float:
        """Simulate one design synchronously and return its cost."""
        override = self._build_override(design, iter_idx, 0)
        override["project"]["parent_output_dir"] = str(
            self.output_dir / "sims" / tag
        )
        result = engine.run_batch([override])[0]
        return self._cost_from_result(result)

    def _evaluate_batch(
        self,
        engine: ParallelVirtuosoEngine,
        candidates: List[DesignState],
        iter_idx: int,
    ) -> List[float]:
        """Simulate all candidates in parallel and return their costs."""
        overrides = [
            self._build_override(d, iter_idx, i)
            for i, d in enumerate(candidates)
        ]
        results = engine.run_batch(overrides)
        return [self._cost_from_result(r) for r in results]

    def _cost_from_result(self, result: "RunResult") -> float:
        """Extract cost from a RunResult using the S-parameter files."""
        if not result.success:
            return 1e9
        # s_param_files is one .s2p per state, in state order
        return compute_total_cost(result.s_param_files, self.specs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_best(
        self,
        design: DesignState,
        cost: float,
        iteration: int,
    ) -> None:
        data = {
            "iteration":      iteration,
            "cost":           cost,
            "matrix":         design.matrix,
            "switch_states":  design.switch_states,
            "cap_values":     design.cap_values,
            "inductor_values": design.inductor_values,
        }
        path = self.output_dir / "best_design.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"  [SA] Best design saved → {path} (cost={cost:.4f})")

    def _log_step(
        self,
        step: int,
        T: float,
        current_cost: float,
        best_cost: float,
        delta: float,
        accepted: bool,
        elapsed: float,
    ) -> None:
        self._progress.append({
            "step":         step,
            "T":            T,
            "current_cost": current_cost,
            "best_cost":    best_cost,
            "delta":        delta,
            "accepted":     accepted,
            "elapsed_s":    elapsed,
        })

    def _save_progress(self) -> None:
        path = self.output_dir / "sa_progress.json"
        path.write_text(json.dumps(self._progress, indent=2))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_sa(config_path: str | Path) -> Tuple[DesignState, float]:
    """Run the full SA optimisation from a config file path."""
    optimizer = SwitchableFilterSA(config_path)
    return optimizer.run()
