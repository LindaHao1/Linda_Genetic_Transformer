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

from virtuoso_engine.config_loader import load_config

from .actions import random_action
from .cost import compute_total_cost, compute_total_cost_dynamic, specs_from_config, FilterStateSpec
from .design import DesignState
from .feasibility import generate_random_design, is_feasible

# RunResult type hint only — the heavy virtuoso_engine import is deferred to
# _make_engine() so that importing this module does not require scikit-rf.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine
    from virtuoso_engine.engine import RunResult


# ---------------------------------------------------------------------------
# Helper: load a DesignState from a pixelated config.yaml
# ---------------------------------------------------------------------------

def _load_design_from_config(config_path: str | Path) -> DesignState:
    """Parse a pixelated config.yaml and return the embedded DesignState."""
    from virtuoso_engine.spectre_runner import _normalize_cap_values, _normalize_state_list
    raw = load_config(config_path)
    pix = raw["pixelated"]
    cap_values   = _normalize_cap_values(pix["cap_values"])
    switch_states = _normalize_state_list(pix["state_list"])
    return DesignState(
        matrix                 = pix["pixel_state_matrix"],
        switch_states          = switch_states,
        cap_values             = cap_values,
        inductor_values        = pix.get("inductor_values", []),
        inductor_switch_states = pix.get("inductor_switch_states", []),
        cap_pixel_pairs        = pix.get("cap_pixel_pairs", []),
        cap_pair_values        = pix.get("cap_pair_values", []),
        inductor_cap_values    = pix.get("inductor_cap_values", []),
        pixel_sizes            = pix.get("pixel_sizes", []),
    )


# ---------------------------------------------------------------------------
# SA configuration container
# ---------------------------------------------------------------------------

@dataclass
class SAConfig:
    # Grid
    rows:              int
    cols:              int
    n_switches:          int
    max_inductors:       int
    n_initial_inductors: int   # exact number of inductors to place in the initial design
    n_states:            int
    cap_min:             float
    cap_max:             float
    n_inductor_values:   int   # number of entries in the inductor_translation_table

    # Cap pixel pairs
    enable_cap_pairs:  bool   # False = cap pairs disabled everywhere (init + SA actions)
    max_cap_pairs:     int
    cap_pair_min:      float
    cap_pair_max:      float
    cap_pair_sigma:    float
    cap_pair_min_dist: int
    cap_pair_max_dist: int

    # SA algorithm
    T_start:        float
    T_end:          float
    n_iter:         int
    cooling:        str    # "geometric" or "linear"
    n_parallel:     int
    n_regular_flip:           int
    cap_sigma:                float
    max_switch_state_flips:   int   # max bits to flip per flip_switch_state action
    max_inductor_state_flips: int   # max bits to flip per flip_inductor_state action
    action_weights: List[float]  # 11-element weights: [flip_01, modify_switch,
                                 #   flip_inductor, flip_inductor_value,
                                 #   perturb_caps, flip_switch_state,
                                 #   add_cap_pair, remove_cap_pair, perturb_pair_caps,
                                 #   flip_inductor_state, flip_via]

    # Paths
    virtuoso_config: str   # path to the Virtuoso engine config YAML
    output_dir:      str   # base directory for all SA run outputs

    # Random seed
    seed: Optional[int]

    # Connectivity mode for initial design generation
    require_4_connected: bool = False

    # Dynamic state assignment: find the optimal bijection of simulated states
    # to filter specs before computing cost.  Helps when the optimiser produces
    # a design whose states are "swapped" relative to the spec ordering.
    dynamic_state_assignment: bool = True

    # Optional path to a pixelated config.yaml to load as the starting design.
    # When set, the random initial design generation is skipped entirely.
    initial_design_path: Optional[str] = None

    # Via-ground pixel hyperparameters
    n_max_vias:         int = 0  # maximum number of type-4 (via) pixels allowed
    n_init_vias:        int = 0  # via pixels to place in the initial design
    via_flip_max_count: int = 3  # max vias to add or remove per flip_via action

    # Random-range overrides for initial design generation.
    # When set, the actual initial counts are sampled uniformly from [min, max]
    # instead of using the fixed point values above.  -1 = use the fixed value.
    n_switches_min:          int = -1
    n_switches_max:          int = -1
    n_initial_inductors_min: int = -1
    n_initial_inductors_max: int = -1
    n_init_vias_min:         int = -1
    n_init_vias_max:         int = -1

    # Capacitor bank size: number of cap+switch pairs per switch pixel.
    # 1 = legacy single-cap model (backward-compatible default).
    n_caps_per_switch: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "SAConfig":
        sa   = d["sa"]
        grid = d["grid"]
        return cls(
            rows               = int(grid["rows"]),
            cols               = int(grid["cols"]),
            n_switches           = int(grid["n_switches"]),
            max_inductors        = int(grid["max_inductors"]),
            n_initial_inductors  = int(grid.get("n_initial_inductors", 0)),
            n_states             = int(grid["n_states"]),
            cap_min              = float(grid["cap_min"]),
            cap_max              = float(grid["cap_max"]),
            n_inductor_values    = int(grid.get("n_inductor_values", 4)),

            enable_cap_pairs   = bool(grid.get("enable_cap_pairs", True)),
            max_cap_pairs      = int(grid.get("max_cap_pairs", 0)),
            cap_pair_min       = float(grid.get("cap_pair_min", 50e-15)),
            cap_pair_max       = float(grid.get("cap_pair_max", 500e-15)),
            cap_pair_sigma     = float(grid.get("cap_pair_sigma", 20e-15)),
            cap_pair_min_dist  = int(grid.get("cap_pair_min_dist", 2)),
            cap_pair_max_dist  = int(grid.get("cap_pair_max_dist", 3)),

            T_start        = float(sa["T_start"]),
            T_end          = float(sa["T_end"]),
            n_iter         = int(sa["n_iter"]),
            cooling        = str(sa.get("cooling", "geometric")),
            n_parallel     = int(sa.get("n_parallel", 4)),
            n_regular_flip           = int(sa.get("n_regular_flip", 3)),
            cap_sigma                = float(sa.get("cap_sigma", 50e-15)),
            max_switch_state_flips   = int(sa.get("max_switch_state_flips", 1)),
            max_inductor_state_flips = int(sa.get("max_inductor_state_flips", 1)),
            action_weights = [float(w) for w in sa.get(
                "action_weights", [1.0] * 11)],

            virtuoso_config = str(d["virtuoso_config"]),
            output_dir      = str(d["output_dir"]),

            seed                      = d.get("seed"),
            require_4_connected       = bool(d.get("require_4_connected", False)),
            dynamic_state_assignment  = bool(d.get("dynamic_state_assignment", True)),

            n_max_vias         = int(grid.get("n_max_vias", 0)),
            n_init_vias        = int(grid.get("n_init_vias", 0)),
            via_flip_max_count = int(sa.get("via_flip_max_count", 3)),

            n_switches_min          = int(grid.get("n_switches_min",          -1)),
            n_switches_max          = int(grid.get("n_switches_max",          -1)),
            n_initial_inductors_min = int(grid.get("n_initial_inductors_min", -1)),
            n_initial_inductors_max = int(grid.get("n_initial_inductors_max", -1)),
            n_init_vias_min         = int(grid.get("n_init_vias_min",         -1)),
            n_init_vias_max         = int(grid.get("n_init_vias_max",         -1)),

            n_caps_per_switch = int(grid.get("n_caps_per_switch", 1)),

            initial_design_path = str(d["initial_design_path"]) if d.get("initial_design_path") else None,
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
        raw = load_config(self.config_path)

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
            current_cost, current_assignment = self._evaluate(engine, current, iter_idx=-1, tag="init")

            print(f"[SA] Initial cost: {current_cost:.4f}"
                  + (f"  assignment={current_assignment}" if current_assignment is not None else ""))

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
                cost_assignments = self._evaluate_batch(engine, candidates, step)
                costs = [c for c, _ in cost_assignments]

                # Pick best candidate
                best_cand_idx  = min(range(len(costs)), key=lambda i: costs[i])
                best_cand, _   = candidates[best_cand_idx]
                best_cand_cost = costs[best_cand_idx]
                best_cand_assignment = cost_assignments[best_cand_idx][1]

                # Metropolis acceptance
                delta = best_cand_cost - current_cost
                if delta < 0:
                    accept = True
                else:
                    accept_prob = math.exp(-delta / T) if T > 1e-30 else 0.0
                    accept = self.rng.random() < accept_prob

                if accept:
                    current            = best_cand
                    current_cost       = best_cand_cost
                    current_assignment = best_cand_assignment

                # Track global best
                if current_cost < best_cost:
                    best_design = current.copy()
                    best_cost   = current_cost
                    self._save_best(best_design, best_cost, step)

                # Progress logging
                elapsed = time.time() - start_time
                self._log_step(step, T, current_cost, best_cost, delta, accept, elapsed,
                               assignment=current_assignment)
                self._save_progress()

                print(
                    f"[SA] iter {step:4d}/{cfg.n_iter}  T={T:.4f}  "
                    f"cost={current_cost:.4f}  best={best_cost:.4f}  "
                    f"{'ACCEPT' if accept else 'reject'}  "
                    f"(Δ={delta:+.4f})  {elapsed:.1f}s"
                    + (f"  assign={current_assignment}" if current_assignment is not None else "")
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

        # If an initial design path is provided, load it directly.
        if cfg.initial_design_path:
            print(f"[SA] Loading initial design from: {cfg.initial_design_path}")
            return _load_design_from_config(cfg.initial_design_path)

        effective_max_cap_pairs = cfg.max_cap_pairs if cfg.enable_cap_pairs else 0

        # Randomly sample initial counts when a range is configured.
        sw_lo  = cfg.n_switches_min          if cfg.n_switches_min          >= 0 else cfg.n_switches
        sw_hi  = cfg.n_switches_max          if cfg.n_switches_max          >= 0 else cfg.n_switches
        ind_lo = cfg.n_initial_inductors_min if cfg.n_initial_inductors_min >= 0 else cfg.n_initial_inductors
        ind_hi = cfg.n_initial_inductors_max if cfg.n_initial_inductors_max >= 0 else cfg.n_initial_inductors
        via_lo = cfg.n_init_vias_min         if cfg.n_init_vias_min         >= 0 else cfg.n_init_vias
        via_hi = cfg.n_init_vias_max         if cfg.n_init_vias_max         >= 0 else cfg.n_init_vias

        init_n_switches    = self.rng.randint(sw_lo,  sw_hi)
        init_n_inductors   = self.rng.randint(ind_lo, ind_hi)
        init_n_vias        = self.rng.randint(via_lo, via_hi)

        design = generate_random_design(
            rows                = cfg.rows,
            cols                = cfg.cols,
            n_switches          = init_n_switches,
            max_inductors       = cfg.max_inductors,
            n_initial_inductors = init_n_inductors,
            n_states            = cfg.n_states,
            cap_min             = cfg.cap_min,
            cap_max             = cfg.cap_max,
            rng                 = self.rng,
            n_inductor_values   = cfg.n_inductor_values,
            require_4_connected = cfg.require_4_connected,
            max_cap_pairs       = effective_max_cap_pairs,
            cap_pair_min        = cfg.cap_pair_min,
            cap_pair_max        = cfg.cap_pair_max,
            cap_pair_min_dist   = cfg.cap_pair_min_dist,
            cap_pair_max_dist   = cfg.cap_pair_max_dist,
            n_init_vias         = init_n_vias,
            max_vias            = cfg.n_max_vias,
            n_caps_per_switch   = cfg.n_caps_per_switch,
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

    def _generate_candidates(self, current: DesignState) -> List[Tuple[DesignState, dict]]:
        """Generate up to N_parallel mutated candidates from the current design.

        Returns a list of (DesignState, action_info) tuples.
        """
        cfg = self.cfg
        candidates: List[Tuple[DesignState, dict]] = []
        seen: set = set()

        max_tries = cfg.n_parallel * 40
        for _ in range(max_tries):
            if len(candidates) >= cfg.n_parallel:
                break
            candidate, action_info = random_action(
                state              = current,
                n_regular_flip     = cfg.n_regular_flip,
                max_inductors      = cfg.max_inductors,
                cap_min            = cfg.cap_min,
                cap_max            = cfg.cap_max,
                cap_sigma          = cfg.cap_sigma,
                rng                = self.rng,
                n_inductor_values  = cfg.n_inductor_values,
                action_weights     = cfg.action_weights,
                max_cap_pairs             = cfg.max_cap_pairs if cfg.enable_cap_pairs else 0,
                cap_pair_min              = cfg.cap_pair_min,
                cap_pair_max              = cfg.cap_pair_max,
                cap_pair_sigma            = cfg.cap_pair_sigma,
                cap_pair_min_dist         = cfg.cap_pair_min_dist,
                cap_pair_max_dist         = cfg.cap_pair_max_dist,
                max_switch_state_flips    = cfg.max_switch_state_flips,
                max_inductor_state_flips  = cfg.max_inductor_state_flips,
                max_vias                  = cfg.n_max_vias,
                via_flip_max_count        = cfg.via_flip_max_count,
            )
            if candidate is None:
                continue
            if not is_feasible(candidate.matrix, candidate.n_switches, cfg.max_inductors, max_vias=cfg.n_max_vias):
                continue
            # Deduplicate by matrix + inductor_values fingerprint
            key = (
                tuple(v for row in candidate.matrix for v in row)
                + tuple(candidate.inductor_values)
            )
            if key not in seen:
                seen.add(key)
                candidates.append((candidate, action_info))

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
    ) -> Tuple[float, Optional[List[int]]]:
        """Simulate one design synchronously and return (cost, assignment)."""
        override = self._build_override(design, iter_idx, 0)
        override["project"]["parent_output_dir"] = str(
            self.output_dir / "sims" / tag
        )
        result = engine.run_batch([override])[0]
        return self._cost_from_result(result)

    def _evaluate_batch(
        self,
        engine: ParallelVirtuosoEngine,
        candidates: List[Tuple[DesignState, dict]],
        iter_idx: int,
    ) -> List[Tuple[float, Optional[List[int]]]]:
        """Simulate all candidates in parallel and return (cost, assignment) per candidate."""
        overrides = []
        for i, (d, action_info) in enumerate(candidates):
            override = self._build_override(d, iter_idx, i)
            overrides.append(override)
            cand_dir = Path(override["project"]["parent_output_dir"])
            cand_dir.mkdir(parents=True, exist_ok=True)
            summary = dict(action_info)
            summary["iteration"] = iter_idx
            summary["candidate_index"] = i
            (cand_dir / "action_summary.json").write_text(json.dumps(summary, indent=2))
        results = engine.run_batch(overrides)
        cost_assignments = [self._cost_from_result(r) for r in results]
        # Write assignment back into each candidate's action_summary.json
        for i, (cost, assignment) in enumerate(cost_assignments):
            if assignment is not None:
                cand_dir = Path(overrides[i]["project"]["parent_output_dir"])
                summary_path = cand_dir / "action_summary.json"
                try:
                    summary = json.loads(summary_path.read_text())
                    summary["cost"] = cost
                    summary["state_assignment"] = assignment
                    summary_path.write_text(json.dumps(summary, indent=2))
                except Exception:
                    pass
        return cost_assignments

    def _cost_from_result(self, result: "RunResult") -> Tuple[float, Optional[List[int]]]:
        """Extract (cost, assignment) from a RunResult."""
        if not result.success:
            return 1e9, None
        if self.cfg.dynamic_state_assignment:
            cost, assignment = compute_total_cost_dynamic(result.s_param_files, self.specs)
            return cost, assignment
        return compute_total_cost(result.s_param_files, self.specs), None

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
        assignment: Optional[List[int]] = None,
    ) -> None:
        entry = {
            "step":         step,
            "T":            T,
            "current_cost": current_cost,
            "best_cost":    best_cost,
            "delta":        delta,
            "accepted":     accepted,
            "elapsed_s":    elapsed,
        }
        if assignment is not None:
            entry["state_assignment"] = assignment
        self._progress.append(entry)

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
