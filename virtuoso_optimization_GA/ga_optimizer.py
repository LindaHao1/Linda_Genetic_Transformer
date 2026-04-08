"""
ga_optimizer.py — Genetic Algorithm optimiser for the switchable-filter design.

Algorithm
---------
Initialisation
  Generate N_pop random feasible designs and evaluate them all.

Each generation:
  1. Preserve the top n_elite individuals unchanged (elitism).
  2. Fill the remaining pop_size − n_elite slots with offspring:
       a. Select two parents via tournament selection.
       b. With probability p_crossover apply zig-zag crossover; otherwise
          clone the fitter parent.
       c. With probability p_mutation apply one random mutation action.
  3. Evaluate every newly created individual in parallel.
  4. Log generation statistics (best / mean / worst cost, diversity).
  5. Save best_design.json, ga_progress.json, per-generation population
     snapshot, and update ga_progress.png.

Diversity metric
  Hamming distance between pixel matrices, averaged over all pairs, then
  divided by total interior cells.  Ranges from 0 (identical population)
  to 1 (maximally diverse).
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

from virtuoso_optimization_SA.actions import random_action
from virtuoso_optimization_SA.cost import (
    compute_total_cost, compute_state_cost, get_s21_db,
    specs_from_config, FilterStateSpec,
)
from virtuoso_optimization_SA.design import DesignState
from virtuoso_optimization_SA.feasibility import generate_random_design, is_feasible

from .crossover import crossover_zig_zag

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine
    from virtuoso_engine.engine import RunResult


# ---------------------------------------------------------------------------
# GA configuration container
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    # Grid
    rows:              int
    cols:              int
    n_switches:        int
    max_inductors:       int
    n_initial_inductors: int
    n_states:            int
    cap_min:             float
    cap_max:             float
    n_inductor_values:   int

    # GA algorithm
    pop_size:        int
    n_gen:           int
    n_elite:         int
    n_immigrants:    int   # fresh random designs per generation
    n_keep_bad:      int   # worst non-elites preserved per generation
    p_crossover:     float
    p_mutation:      float
    selection:       str    # "tournament"
    tournament_size: int
    n_parallel:      int
    n_regular_flip:  int
    cap_sigma:       float
    action_weights:  List[float]
    use_state_champions: bool

    # Paths
    virtuoso_config: str
    output_dir:      str

    # Random seed
    seed: Optional[int]

    @classmethod
    def from_dict(cls, d: dict) -> "GAConfig":
        ga   = d["ga"]
        grid = d["grid"]
        return cls(
            rows               = int(grid["rows"]),
            cols               = int(grid["cols"]),
            n_switches         = int(grid["n_switches"]),
            max_inductors        = int(grid["max_inductors"]),
            n_initial_inductors  = int(grid.get("n_initial_inductors", 4)),
            n_states             = int(grid["n_states"]),
            cap_min            = float(grid["cap_min"]),
            cap_max            = float(grid["cap_max"]),
            n_inductor_values  = int(grid.get("n_inductor_values", 4)),

            pop_size        = int(ga["pop_size"]),
            n_gen           = int(ga["n_gen"]),
            n_elite         = int(ga.get("n_elite", 2)),
            n_immigrants    = int(ga.get("n_immigrants", 0)),
            n_keep_bad      = int(ga.get("n_keep_bad", 0)),
            p_crossover     = float(ga.get("p_crossover", 0.7)),
            p_mutation      = float(ga.get("p_mutation", 0.4)),
            selection       = str(ga.get("selection", "tournament")),
            tournament_size = int(ga.get("tournament_size", 3)),
            n_parallel      = int(ga.get("n_parallel", 4)),
            n_regular_flip  = int(ga.get("n_regular_flip", 3)),
            cap_sigma       = float(ga.get("cap_sigma", 50e-15)),
            action_weights  = [float(w) for w in ga.get(
                "action_weights", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])],
            use_state_champions = bool(ga.get("use_state_champions", True)),

            virtuoso_config = str(d["virtuoso_config"]),
            output_dir      = str(d["output_dir"]),

            seed = d.get("seed"),
        )


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _tournament_select(
    population: List[DesignState],
    costs: List[float],
    k: int,
    rng: random.Random,
) -> DesignState:
    """Return the fittest individual from a random tournament of size k."""
    indices = rng.sample(range(len(population)), min(k, len(population)))
    best_idx = min(indices, key=lambda i: costs[i])
    return population[best_idx]


# ---------------------------------------------------------------------------
# Diversity metric
# ---------------------------------------------------------------------------

def _population_diversity(population: List[DesignState]) -> float:
    """
    Mean normalised Hamming distance between all pairs of pixel matrices.
    Returns 0.0 if the population has fewer than 2 individuals.
    """
    n = len(population)
    if n < 2:
        return 0.0

    rows = population[0].rows
    cols = population[0].cols
    interior_cells = rows * max(cols - 2, 1)  # exclude feed columns
    if interior_cells == 0:
        return 0.0

    total_pairs = 0
    total_hamming = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = sum(
                1
                for r in range(rows)
                for c in range(1, cols - 1)
                if population[i].matrix[r][c] != population[j].matrix[r][c]
            )
            total_hamming += diff / interior_cells
            total_pairs += 1

    return total_hamming / total_pairs


# ---------------------------------------------------------------------------
# Main GA optimiser class
# ---------------------------------------------------------------------------

class SwitchableFilterGA:
    """
    Genetic Algorithm optimiser for the pixelated switchable filter.

    Usage::

        optimizer = SwitchableFilterGA("ga_config.yaml")
        best_design, best_cost = optimizer.run()
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with self.config_path.open() as f:
            raw = yaml.safe_load(f)

        self.cfg   = GAConfig.from_dict(raw)
        self.specs = specs_from_config(raw["filter_specs"])
        self.rng   = random.Random(self.cfg.seed)

        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "population").mkdir(exist_ok=True)

        self._progress: List[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Tuple[DesignState, float]:
        """Run the GA optimisation loop. Returns (best_design, best_cost)."""
        cfg = self.cfg
        print(f"\n{'='*70}")
        print(f"[GA] Switchable Filter Optimiser")
        print(f"[GA] Grid: {cfg.rows}x{cfg.cols}, switches={cfg.n_switches}, "
              f"max_ind={cfg.max_inductors}, states={cfg.n_states}")
        print(f"[GA] pop={cfg.pop_size}, n_gen={cfg.n_gen}, "
              f"elite={cfg.n_elite}, immigrants={cfg.n_immigrants}, keep_bad={cfg.n_keep_bad}, "
              f"p_cross={cfg.p_crossover}, p_mut={cfg.p_mutation}")
        print(f"{'='*70}\n")

        start_time = time.time()

        with self._make_engine() as engine:
            # --- Initialise population ---
            print(f"[GA] Generating initial population of {cfg.pop_size} ...")
            population = self._init_population()
            print(f"[GA] Evaluating initial population ...")
            if cfg.use_state_champions:
                costs, init_per_state_costs = self._evaluate_batch_with_state_costs(
                    engine, population, gen_idx=-1
                )
            else:
                costs = self._evaluate_batch(engine, population, gen_idx=-1)
                init_per_state_costs = []

            best_idx    = min(range(len(costs)), key=lambda i: costs[i])
            best_design = population[best_idx].copy()
            best_cost   = costs[best_idx]
            best_sim_tag = f"gen-1_ind{best_idx}"
            self._save_best(best_design, best_cost, generation=-1, sim_tag=best_sim_tag)
            self._save_population_snapshot(population, costs, gen=-1)

            # --- State champion initialisation (reuses simulation data, no extra runs) ---
            # state_champion_costs      : per-state cost (used only to decide which champion wins)
            # state_champion_total_costs: total cost of the champion (used in population costs list)
            n_specs = len(self.specs)
            state_champions: List[Optional[DesignState]] = [None] * n_specs
            state_champion_costs: List[float]       = [1e9] * n_specs   # per-state
            state_champion_total_costs: List[float] = [1e9] * n_specs   # total
            if cfg.use_state_champions and init_per_state_costs:
                for ind_idx, psc in enumerate(init_per_state_costs):
                    for si, sc in enumerate(psc):
                        if sc < state_champion_costs[si]:
                            state_champion_costs[si]       = sc
                            state_champion_total_costs[si] = costs[ind_idx]
                            state_champions[si] = population[ind_idx].copy()

            print(f"[GA] Initial best cost: {best_cost:.4f}")

            # --- Generational loop ---
            for gen in range(cfg.n_gen):
                # Sort population by cost (ascending = best first)
                sorted_indices = sorted(range(len(costs)), key=lambda i: costs[i])

                # 1. Elite preservation
                elites      = [population[i].copy() for i in sorted_indices[:cfg.n_elite]]
                elite_costs = [costs[i]             for i in sorted_indices[:cfg.n_elite]]

                # 2. Keep-bad: randomly sample from the bottom 50% of non-elites
                kept_bad: List[DesignState] = []
                kept_bad_costs: List[float] = []
                if cfg.n_keep_bad > 0:
                    non_elite   = sorted_indices[cfg.n_elite:]
                    bottom_half = non_elite[len(non_elite) // 2:]
                    if bottom_half:
                        chosen         = self.rng.sample(bottom_half, min(cfg.n_keep_bad, len(bottom_half)))
                        kept_bad       = [population[i].copy() for i in chosen]
                        kept_bad_costs = [costs[i]             for i in chosen]

                # 2b. State champion pool (deduplicated against global elites)
                sc_pool: List[DesignState] = []
                sc_pool_costs: List[float] = []
                if cfg.use_state_champions:
                    sc_budget = max(0, cfg.pop_size - cfg.n_elite - len(kept_bad))
                    for si, champ in enumerate(state_champions):
                        if champ is None or len(sc_pool) >= sc_budget:
                            break
                        if not any(champ.matrix == e.matrix for e in elites + sc_pool):
                            sc_pool.append(champ.copy())
                            # Use total cost so the population costs list stays on the same scale
                            sc_pool_costs.append(state_champion_total_costs[si])

                # 3. Produce regular offspring via crossover + mutation
                offspring: List[DesignState] = []
                n_offspring = cfg.pop_size - cfg.n_elite - len(kept_bad) - len(sc_pool)
                n_regular   = n_offspring - cfg.n_immigrants
                max_tries   = n_offspring * 10
                attempts    = 0
                while len(offspring) < n_regular and attempts < max_tries:
                    attempts += 1
                    p1 = _tournament_select(population, costs, cfg.tournament_size, self.rng)
                    p2 = _tournament_select(population, costs, cfg.tournament_size, self.rng)

                    # Crossover
                    if self.rng.random() < cfg.p_crossover:
                        child = crossover_zig_zag(
                            p1, p2,
                            rng                 = self.rng,
                            n_switches_required = cfg.n_switches,
                            max_inductors       = cfg.max_inductors,
                            n_states            = cfg.n_states,
                            cap_min             = cfg.cap_min,
                            cap_max             = cfg.cap_max,
                            n_inductor_values   = cfg.n_inductor_values,
                        )
                        if child is None:
                            continue   # crossover failed — retry
                    else:
                        # Clone fitter parent
                        p1_cost = costs[population.index(p1)] if p1 in population else 1e9
                        p2_cost = costs[population.index(p2)] if p2 in population else 1e9
                        child = (p1 if p1_cost <= p2_cost else p2).copy()

                    # Mutation
                    if self.rng.random() < cfg.p_mutation:
                        mutated, _ = random_action(
                            state             = child,
                            n_regular_flip    = cfg.n_regular_flip,
                            max_inductors     = cfg.max_inductors,
                            cap_min           = cfg.cap_min,
                            cap_max           = cfg.cap_max,
                            cap_sigma         = cfg.cap_sigma,
                            rng               = self.rng,
                            n_inductor_values = cfg.n_inductor_values,
                            action_weights    = cfg.action_weights,
                        )
                        if mutated is not None:
                            child = mutated

                    if not is_feasible(child.matrix, cfg.n_switches, cfg.max_inductors):
                        continue

                    offspring.append(child)

                # 4. Random immigrants
                immigrants: List[DesignState] = []
                for _ in range(cfg.n_immigrants):
                    immigrant = generate_random_design(
                        rows                = cfg.rows,
                        cols                = cfg.cols,
                        n_switches          = cfg.n_switches,
                        max_inductors       = cfg.max_inductors,
                        n_initial_inductors = cfg.n_initial_inductors,
                        n_states            = cfg.n_states,
                        cap_min             = cfg.cap_min,
                        cap_max             = cfg.cap_max,
                        rng                 = self.rng,
                        n_inductor_values   = cfg.n_inductor_values,
                    )
                    if immigrant is not None:
                        immigrants.append(immigrant)

                # 5. Evaluate all new individuals in one batch
                to_evaluate = offspring + immigrants
                if cfg.use_state_champions:
                    new_costs, new_per_state = self._evaluate_batch_with_state_costs(
                        engine, to_evaluate, gen_idx=gen
                    )
                    # Update state champions from newly evaluated individuals
                    for local_idx, psc in enumerate(new_per_state):
                        for si, sc in enumerate(psc):
                            if sc < state_champion_costs[si]:
                                state_champion_costs[si]       = sc
                                state_champion_total_costs[si] = new_costs[local_idx]
                                state_champions[si] = to_evaluate[local_idx].copy()
                else:
                    new_costs = self._evaluate_batch(engine, to_evaluate, gen_idx=gen)
                offspring_costs = new_costs[:len(offspring)]
                immigrant_costs = new_costs[len(offspring):]

                # 6. Assemble new population: elites | state_champs | kept_bad | offspring | immigrants
                population = elites + sc_pool + kept_bad + offspring + immigrants
                costs      = elite_costs + sc_pool_costs + kept_bad_costs + offspring_costs + immigrant_costs

                # Track global best
                gen_best_idx = min(range(len(costs)), key=lambda i: costs[i])
                if costs[gen_best_idx] < best_cost:
                    best_design = population[gen_best_idx].copy()
                    best_cost   = costs[gen_best_idx]
                    # Compute population-slot offset to find the sim folder index.
                    # Slots: elites[0..n_elite) | sc_pool | kept_bad | to_evaluate[0..)
                    carried_count = cfg.n_elite + len(sc_pool) + len(kept_bad)
                    if gen_best_idx >= carried_count:
                        best_local_idx = gen_best_idx - carried_count
                        best_sim_tag   = f"gen{gen}_ind{best_local_idx}"
                    else:
                        best_sim_tag = f"elite/carried_gen{gen}_slot{gen_best_idx}"
                    self._save_best(best_design, best_cost, generation=gen,
                                    sim_tag=best_sim_tag)

                # Stats
                diversity = _population_diversity(population)
                mean_cost  = sum(costs) / len(costs)
                worst_cost = max(costs)
                elapsed    = time.time() - start_time

                self._log_generation(gen, best_cost, best_sim_tag, mean_cost,
                                     worst_cost, diversity, elapsed)
                self._save_progress()
                self._save_population_snapshot(population, costs, gen=gen)
                self._plot_progress()

                if cfg.use_state_champions:
                    sc_costs_str = "  ".join(
                        f"sc{si}={c:.4f}" for si, c in enumerate(state_champion_costs)
                    )
                    print(
                        f"[GA] gen {gen:4d}/{cfg.n_gen}  "
                        f"best={best_cost:.4f}  mean={mean_cost:.4f}  "
                        f"worst={worst_cost:.4f}  div={diversity:.3f}  "
                        f"{sc_costs_str}  "
                        f"{elapsed:.1f}s"
                    )
                else:
                    print(
                        f"[GA] gen {gen:4d}/{cfg.n_gen}  "
                        f"best={best_cost:.4f}  mean={mean_cost:.4f}  "
                        f"worst={worst_cost:.4f}  div={diversity:.3f}  "
                        f"{elapsed:.1f}s"
                    )

        print(f"\n[GA] Finished. Best cost: {best_cost:.4f}")
        print(f"[GA] Best design:\n{best_design}")
        return best_design, best_cost

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    def _make_engine(self):
        from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine
        return ParallelVirtuosoEngine(
            config_path = self.cfg.virtuoso_config,
            num_workers = self.cfg.n_parallel,
            log_path    = self.output_dir / "parallel_engine.log",
        )

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _init_population(self) -> List[DesignState]:
        cfg = self.cfg
        population: List[DesignState] = []
        for i in range(cfg.pop_size):
            design = generate_random_design(
                rows                = cfg.rows,
                cols                = cfg.cols,
                n_switches          = cfg.n_switches,
                max_inductors       = cfg.max_inductors,
                n_initial_inductors = cfg.n_initial_inductors,
                n_states            = cfg.n_states,
                cap_min             = cfg.cap_min,
                cap_max             = cfg.cap_max,
                rng                 = self.rng,
                n_inductor_values   = cfg.n_inductor_values,
            )
            if design is None:
                raise RuntimeError(
                    f"Failed to generate individual {i} of the initial population. "
                    "Relax grid parameters."
                )
            population.append(design)
        return population

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _build_override(
        self,
        design: DesignState,
        gen_idx: int,
        cand_idx: int,
    ) -> dict:
        tag        = f"gen{gen_idx}_ind{cand_idx}"
        output_dir = str(self.output_dir / "sims" / tag)
        return {
            "mode":      "pixelated_design_simple",
            "pixelated": design.to_pixelated_config(),
            "project":   {"parent_output_dir": output_dir},
        }

    def _evaluate_batch(
        self,
        engine: "ParallelVirtuosoEngine",
        individuals: List[DesignState],
        gen_idx: int,
    ) -> List[float]:
        if not individuals:
            return []
        overrides = [
            self._build_override(d, gen_idx, i)
            for i, d in enumerate(individuals)
        ]
        results = engine.run_batch(overrides)
        costs = [self._cost_from_result(r) for r in results]
        for r, cost in zip(results, costs):
            (r.run_dir / "cost.txt").write_text(f"{cost}\n")
        return costs

    def _cost_from_result(self, result: "RunResult") -> float:
        if not result.success:
            return 1e9
        return compute_total_cost(result.s_param_files, self.specs)

    def _per_state_costs_from_result(self, result: "RunResult") -> List[float]:
        """Return per-spec cost list (parallel to self.specs). 1e9 on failure."""
        n = len(self.specs)
        if not result.success:
            return [1e9] * n
        per_state: List[float] = []
        for spec in self.specs:
            si = spec.state_index
            if si >= len(result.s_param_files) or result.s_param_files[si] is None:
                per_state.append(1e9 * spec.weight)
                continue
            try:
                freqs, s21_db_vals = get_s21_db(result.s_param_files[si])
                per_state.append(compute_state_cost(freqs, s21_db_vals, spec))
            except Exception:
                per_state.append(1e9 * spec.weight)
        return per_state

    def _evaluate_batch_with_state_costs(
        self,
        engine: "ParallelVirtuosoEngine",
        individuals: List[DesignState],
        gen_idx: int,
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Like _evaluate_batch but also returns per-state costs.

        Returns
        -------
        costs           : List[float]         — total cost per individual
        per_state_costs : List[List[float]]   — [individual_idx][spec_idx]
        """
        if not individuals:
            return [], []
        overrides = [
            self._build_override(d, gen_idx, i)
            for i, d in enumerate(individuals)
        ]
        results = engine.run_batch(overrides)
        costs: List[float] = []
        per_state_costs: List[List[float]] = []
        for r in results:
            c = self._cost_from_result(r)
            costs.append(c)
            per_state_costs.append(self._per_state_costs_from_result(r))
            (r.run_dir / "cost.txt").write_text(f"{c}\n")
        return costs, per_state_costs

    # ------------------------------------------------------------------
    # Persistence & plotting
    # ------------------------------------------------------------------

    def _save_best(
        self,
        design: DesignState,
        cost: float,
        generation: int,
        sim_tag: str = "",
    ) -> None:
        sim_dir = str(self.output_dir / "sims" / sim_tag) if sim_tag else ""
        data = {
            "generation":            generation,
            "sim_tag":               sim_tag,
            "sim_dir":               sim_dir,
            "cost":                  cost,
            "matrix":                design.matrix,
            "switch_states":         design.switch_states,
            "cap_values":            design.cap_values,
            "inductor_values":       design.inductor_values,
            "inductor_switch_states": design.inductor_switch_states,
        }
        path = self.output_dir / "best_design.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"  [GA] Best design saved → {path}  cost={cost:.4f}  sim={sim_tag}")

    def _log_generation(
        self,
        gen: int,
        best_cost: float,
        best_sim_tag: str,
        mean_cost: float,
        worst_cost: float,
        diversity: float,
        elapsed: float,
    ) -> None:
        self._progress.append({
            "gen":          gen,
            "best_cost":    best_cost,
            "best_sim_tag": best_sim_tag,
            "mean_cost":    mean_cost,
            "worst_cost":   worst_cost,
            "diversity":    diversity,
            "elapsed_s":    elapsed,
        })

    def _save_progress(self) -> None:
        path = self.output_dir / "ga_progress.json"
        path.write_text(json.dumps(self._progress, indent=2))

    def _save_population_snapshot(
        self,
        population: List[DesignState],
        costs: List[float],
        gen: int,
    ) -> None:
        snapshot = [
            {
                "cost":           costs[i],
                "matrix":         population[i].matrix,
                "switch_states":  population[i].switch_states,
                "cap_values":     population[i].cap_values,
                "inductor_values": population[i].inductor_values,
            }
            for i in range(len(population))
        ]
        tag  = f"gen_{gen:04d}" if gen >= 0 else "gen_init"
        path = self.output_dir / "population" / f"{tag}.json"
        path.write_text(json.dumps(snapshot, indent=2))

    def _plot_progress(self) -> None:
        """Save a convergence plot (best / mean / worst cost vs generation)."""
        if not self._progress:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return   # matplotlib not available — skip silently

        gens       = [e["gen"]        for e in self._progress]
        best_costs = [e["best_cost"]  for e in self._progress]
        mean_costs = [e["mean_cost"]  for e in self._progress]
        worst_costs = [e["worst_cost"] for e in self._progress]
        diversity  = [e["diversity"]  for e in self._progress]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(gens, best_costs,  label="best",  linewidth=2, color="tab:green")
        ax1.plot(gens, mean_costs,  label="mean",  linewidth=1.5, color="tab:blue", linestyle="--")
        ax1.plot(gens, worst_costs, label="worst", linewidth=1, color="tab:red", linestyle=":")
        ax1.set_ylabel("Cost")
        ax1.set_title("GA Convergence — Switchable Filter")
        ax1.legend()
        ax1.grid(True, alpha=0.4)

        ax2.plot(gens, diversity, color="tab:orange", linewidth=1.5, label="diversity")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Diversity (norm. Hamming)")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.4)

        fig.tight_layout()
        fig.savefig(self.output_dir / "ga_progress.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_ga(config_path: str | Path) -> Tuple[DesignState, float]:
    """Run the full GA optimisation from a config file path."""
    optimizer = SwitchableFilterGA(config_path)
    return optimizer.run()
