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
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from virtuoso_optimization_SA.actions import random_action, _ACTION_NAMES
from virtuoso_optimization_SA.sa_optimizer import _load_design_from_config
from virtuoso_optimization_SA.cost import (
    compute_total_cost, compute_total_cost_dynamic, compute_state_cost, get_s21_db,
    compute_cost_breakdown,
    specs_from_config, FilterStateSpec,
)
from virtuoso_optimization_SA.design import DesignState
from virtuoso_optimization_SA.feasibility import generate_random_design, is_feasible

from .crossover import crossover_zig_zag
from .crossover_weighted_clustering import crossover_weighted_clustering
from .crossover_block import crossover_block

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine
    from virtuoso_engine.engine import RunResult


# ---------------------------------------------------------------------------
# Action-weight helpers
# ---------------------------------------------------------------------------

def _parse_action_weights(val, n_actions: int = len(_ACTION_NAMES)) -> List[float]:
    """
    Parse action weights from either a dict or a list.

    Dict form (preferred — readable in YAML):
        {"flip_01": 4.0, "move_switch": 2.0, ...}
    List form (legacy — positional, must match _ACTION_NAMES order):
        [4.0, 2.0, 1.0, ...]

    Dict keys that don't match any action name are silently ignored.
    Actions missing from the dict default to 0.0.
    Lists shorter than n_actions are zero-padded; longer lists are truncated.
    """
    if isinstance(val, dict):
        weights = [float(val.get(name, 0.0)) for name in _ACTION_NAMES]
    else:
        weights = [float(w) for w in val]
        if len(weights) < n_actions:
            weights += [0.0] * (n_actions - len(weights))
        elif len(weights) > n_actions:
            weights = weights[:n_actions]
    return weights


# ---------------------------------------------------------------------------
# GA configuration container
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    # Grid
    rows:              int
    cols:              int
    # Feasibility constraint ranges
    min_switches:        int
    max_switches:        int
    max_inductors:       int
    # Initialisation ranges (for random design generation)
    init_min_switches:   int
    init_max_switches:   int
    init_min_inductors:  int
    init_max_inductors:  int
    init_min_vias:       int
    init_max_vias:       int
    n_states:            int
    cap_min:             float
    cap_max:             float
    n_inductor_values:   int

    # GA algorithm
    pop_size_start:  int    # initial population size
    pop_size_end:    int    # final population size
    pop_size_step:   int    # decrement per schedule step
    elite_fraction:  float  # n_elite = max(3, round(current_pop * elite_fraction))
    n_gen:           int
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
    use_state_champions:          bool
    p_champion_parent:            float  # prob. p1 is forced to be a state champion
    dynamic_state_assignment:     bool   # use Hungarian algorithm to assign sim states → specs
    exploitation_n:               int    # exploitation mutations of best design per generation
    exploitation_action_weights:  List[float]  # switch-state-only action weights

    # Via-ground pixel hyperparameters
    n_max_vias:         int    # maximum number of type-4 (via) pixels allowed
    via_flip_max_count: int    # max vias to add or remove per flip_via action

    # Crossover method
    crossover_method: str   # "zig_zag" | "weighted_clustering" | "block"
    n_centroids:      int   # max cluster centroids for weighted_clustering (actual count drawn randomly in [2, n_centroids])
    block_min_size:   int   # minimum block width/height for block crossover
    block_max_size:   int   # maximum block width/height for block crossover

    # Paths
    virtuoso_config: str
    output_dir:      str

    # Random seed
    seed: Optional[int]

    # Cap pixel pairs
    enable_cap_pairs:   bool  = False
    max_cap_pairs:      int   = 0
    cap_pair_init_min:  int   = 0    # min cap pairs placed in a new random design
    cap_pair_init_max:  int   = -1   # max cap pairs placed in a new random design (-1 = use max_cap_pairs)
    cap_pair_min:       float = 50e-15
    cap_pair_max:       float = 1000e-15
    cap_pair_sigma:     float = 50e-15
    cap_pair_min_dist:  int   = 2
    cap_pair_max_dist:  int   = 3

    # Plotting: exclude high-cost (infeasible-performance) designs from mean/worst stats
    feasibility_cost_threshold: float = 1e6   # costs above this are excluded from filtered stats
    population_plot_top_k:      int   = 5     # number of top individuals to track per generation

    # Fine-tune: seed initial population from an existing design
    finetune_design_path: Optional[str] = None  # path to a pixelated config.yaml; None = disabled
    finetune_seed_ratio:  float = 0.7           # fraction of pop_size_start seeded by mutations of that design

    # Inductor constraint: force all inductors to share one value index
    single_inductor_value: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "GAConfig":
        ga   = d["ga"]
        grid = d["grid"]
        return cls(
            rows               = int(grid["rows"]),
            cols               = int(grid["cols"]),
            min_switches         = int(grid.get("min_switches", 2)),
            max_switches         = int(grid.get("max_switches", 6)),
            max_inductors        = int(grid["max_inductors"]),
            init_min_switches    = int(grid.get("init_min_switches", grid.get("min_switches", 2))),
            init_max_switches    = int(grid.get("init_max_switches", grid.get("max_switches", 6))),
            init_min_inductors   = int(grid.get("init_min_inductors", 0)),
            init_max_inductors   = int(grid.get("init_max_inductors", grid["max_inductors"])),
            init_min_vias        = int(grid.get("init_min_vias", 0)),
            init_max_vias        = int(grid.get("init_max_vias", grid.get("n_max_vias", 0))),
            n_states             = int(grid["n_states"]),
            cap_min            = float(grid["cap_min"]),
            cap_max            = float(grid["cap_max"]),
            n_inductor_values  = int(grid.get("n_inductor_values", 4)),

            pop_size_start  = int(ga.get("pop_size_start", ga.get("pop_size", 30))),
            pop_size_end    = int(ga.get("pop_size_end", 20)),
            pop_size_step   = int(ga.get("pop_size_step", 10)),
            elite_fraction  = float(ga.get("elite_fraction", 0.20)),
            n_gen           = int(ga["n_gen"]),
            n_immigrants    = int(ga.get("n_immigrants", 0)),
            n_keep_bad      = int(ga.get("n_keep_bad", 0)),
            p_crossover     = float(ga.get("p_crossover", 0.7)),
            p_mutation      = float(ga.get("p_mutation", 0.4)),
            selection       = str(ga.get("selection", "tournament")),
            tournament_size = int(ga.get("tournament_size", 3)),
            n_parallel      = int(ga.get("n_parallel", 4)),
            n_regular_flip  = int(ga.get("n_regular_flip", 3)),
            cap_sigma       = float(ga.get("cap_sigma", 50e-15)),
            action_weights  = _parse_action_weights(ga.get(
                "action_weights", {n: 1.0 for n in _ACTION_NAMES})),
            use_state_champions      = bool(ga.get("use_state_champions", True)),
            p_champion_parent        = float(ga.get("p_champion_parent", 0.0)),
            dynamic_state_assignment = bool(ga.get("dynamic_state_assignment", True)),
            exploitation_n               = int(ga.get("exploitation_n", 0)),
            exploitation_action_weights  = _parse_action_weights(ga.get(
                "exploitation_action_weights", {})),

            n_max_vias         = int(grid.get("n_max_vias", 0)),
            via_flip_max_count = int(ga.get("via_flip_max_count", 3)),

            crossover_method = str(ga.get("crossover_method", "zig_zag")),
            n_centroids      = int(ga.get("n_centroids", 3)),
            block_min_size   = int(ga.get("block_min_size", 5)),
            block_max_size   = int(ga.get("block_max_size", 15)),

            enable_cap_pairs   = bool(grid.get("enable_cap_pairs", False)),
            max_cap_pairs      = int(grid.get("max_cap_pairs", 0)),
            cap_pair_init_min  = int(grid.get("cap_pair_init_min", 0)),
            cap_pair_init_max  = int(grid.get("cap_pair_init_max", -1)),
            cap_pair_min       = float(grid.get("cap_pair_min", 50e-15)),
            cap_pair_max       = float(grid.get("cap_pair_max", 1000e-15)),
            cap_pair_sigma     = float(grid.get("cap_pair_sigma", 50e-15)),
            cap_pair_min_dist  = int(grid.get("cap_pair_min_dist", 2)),
            cap_pair_max_dist  = int(grid.get("cap_pair_max_dist", 3)),

            feasibility_cost_threshold = float(ga.get("feasibility_cost_threshold", 1e6)),
            population_plot_top_k      = int(ga.get("population_plot_top_k", 5)),

            finetune_design_path = str(ga["finetune_design_path"]) if ga.get("finetune_design_path") else None,
            finetune_seed_ratio  = float(ga.get("finetune_seed_ratio", 0.7)),

            single_inductor_value = bool(grid.get("single_inductor_value", False)),

            virtuoso_config = str(d["virtuoso_config"]),
            output_dir      = str(d["output_dir"]),

            seed = d.get("seed"),
        )


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _tournament_select_idx(
    population: List[DesignState],
    costs: List[float],
    k: int,
    rng: random.Random,
) -> int:
    """Return the index of the fittest individual from a random tournament of size k."""
    indices = rng.sample(range(len(population)), min(k, len(population)))
    return min(indices, key=lambda i: costs[i])


def _tournament_select(
    population: List[DesignState],
    costs: List[float],
    k: int,
    rng: random.Random,
) -> DesignState:
    """Return the fittest individual from a random tournament of size k."""
    return population[_tournament_select_idx(population, costs, k, rng)]


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
        (self.output_dir / "state_best").mkdir(exist_ok=True)

        # Save config snapshots for future reference
        shutil.copy2(self.config_path, self.output_dir / "ga_config_snapshot.yaml")
        shutil.copy2(self.cfg.virtuoso_config, self.output_dir / "virtuoso_config_snapshot.yaml")

        self._progress: List[dict] = []

    # ------------------------------------------------------------------
    # Population / elite schedule
    # ------------------------------------------------------------------

    def _scheduled_pop(self, gen: int) -> tuple:
        """Return (current_pop_size, current_n_elite) for the given generation."""
        cfg = self.cfg
        n_steps = max(1, (cfg.pop_size_start - cfg.pop_size_end) // cfg.pop_size_step)
        gens_per_step = max(1, cfg.n_gen // n_steps)
        step  = min(gen // gens_per_step, n_steps)
        pop   = cfg.pop_size_start - step * cfg.pop_size_step
        elite = max(3, round(pop * cfg.elite_fraction))
        return pop, elite

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Tuple[DesignState, float]:
        """Run the GA optimisation loop. Returns (best_design, best_cost)."""
        cfg = self.cfg
        print(f"\n{'='*70}")
        print(f"[GA] Switchable Filter Optimiser")
        print(f"[GA] Grid: {cfg.rows}x{cfg.cols}, switches={cfg.min_switches}–{cfg.max_switches}, "
              f"max_ind={cfg.max_inductors}, states={cfg.n_states}")
        print(f"[GA] pop={cfg.pop_size_start}→{cfg.pop_size_end} (step {cfg.pop_size_step}), "
              f"n_gen={cfg.n_gen}, elite_frac={cfg.elite_fraction:.0%}, "
              f"immigrants={cfg.n_immigrants}, keep_bad={cfg.n_keep_bad}, "
              f"p_cross={cfg.p_crossover}, p_mut={cfg.p_mutation}")
        print(f"{'='*70}\n")

        start_time = time.time()

        with self._make_engine() as engine:
            # --- Initialise population ---
            print(f"[GA] Generating initial population of {cfg.pop_size_start} ...")
            population = self._init_population()
            print(f"[GA] Evaluating initial population ...")
            if cfg.use_state_champions:
                costs, init_per_state_costs, init_sim_data = self._evaluate_batch_with_state_costs(
                    engine, population, gen_idx=-1
                )
            else:
                costs, init_sim_data = self._evaluate_batch(engine, population, gen_idx=-1)
                init_per_state_costs = []

            best_idx    = min(range(len(costs)), key=lambda i: costs[i])
            best_design = population[best_idx].copy()
            best_cost   = costs[best_idx]
            best_sim_tag = f"gen-1_ind{best_idx}"
            best_s2p_files, best_assignment = init_sim_data[best_idx]
            self._save_best(best_design, best_cost, generation=-1, sim_tag=best_sim_tag,
                            s2p_files=best_s2p_files, assignment=best_assignment)
            self._save_population_snapshot(population, costs, gen=-1)

            # Parallel list tracking the sim tag of each population member
            population_sim_tags: List[Optional[str]] = [
                f"gen-1_ind{i}" for i in range(len(population))
            ]

            # --- State champion initialisation (reuses simulation data, no extra runs) ---
            # state_champion_costs      : per-state cost (used only to decide which champion wins)
            # state_champion_total_costs: total cost of the champion (used in population costs list)
            n_specs = len(self.specs)
            state_champions: List[Optional[DesignState]] = [None] * n_specs
            state_champion_costs: List[float]       = [1e9] * n_specs   # per-state
            state_champion_total_costs: List[float] = [1e9] * n_specs   # total
            state_champion_sim_tags: List[Optional[str]] = [None] * n_specs
            if cfg.use_state_champions and init_per_state_costs:
                for ind_idx, psc in enumerate(init_per_state_costs):
                    for si, sc in enumerate(psc):
                        if sc < state_champion_costs[si]:
                            state_champion_costs[si]       = sc
                            state_champion_total_costs[si] = costs[ind_idx]
                            state_champions[si] = population[ind_idx].copy()
                            state_champion_sim_tags[si] = f"gen-1_ind{ind_idx}"
                            self._save_state_best(
                                state_idx      = si,
                                design         = population[ind_idx],
                                per_state_cost = sc,
                                total_cost     = costs[ind_idx],
                                generation     = -1,
                                sim_tag        = f"gen-1_ind{ind_idx}",
                                provenance     = {
                                    "generation":            -1,
                                    "sim_tag":               f"gen-1_ind{ind_idx}",
                                    "used_crossover":        False,
                                    "crossover_method":      None,
                                    "parent1_sim_tag":       None,
                                    "parent2_sim_tag":       None,
                                    "parent1_from_champion": None,
                                    "mutation_action":       None,
                                    "mutation_description":  "selected from random initial population",
                                },
                            )

            print(f"[GA] Initial best cost: {best_cost:.4f}")

            # --- Generational loop ---
            for gen in range(cfg.n_gen):
                current_pop_size, current_n_elite = self._scheduled_pop(gen)

                # Sort population by cost (ascending = best first)
                sorted_indices = sorted(range(len(costs)), key=lambda i: costs[i])

                # 1. Elite preservation
                elites      = [population[i].copy() for i in sorted_indices[:current_n_elite]]
                elite_costs = [costs[i]             for i in sorted_indices[:current_n_elite]]

                # 1b. Best-design exploitation: fire switch-state-only mutations of global best
                exploit_candidates: List[DesignState] = []
                if cfg.exploitation_n > 0:
                    for _ in range(cfg.exploitation_n):
                        mutated, _ = random_action(
                            state                    = best_design,
                            n_regular_flip           = cfg.n_regular_flip,
                            max_inductors            = cfg.max_inductors,
                            cap_min                  = cfg.cap_min,
                            cap_max                  = cfg.cap_max,
                            cap_sigma                = cfg.cap_sigma,
                            rng                      = self.rng,
                            n_inductor_values        = cfg.n_inductor_values,
                            action_weights           = cfg.exploitation_action_weights,
                            max_cap_pairs            = cfg.max_cap_pairs if cfg.enable_cap_pairs else 0,
                            cap_pair_min             = cfg.cap_pair_min,
                            cap_pair_max             = cfg.cap_pair_max,
                            cap_pair_sigma           = cfg.cap_pair_sigma,
                            cap_pair_min_dist        = cfg.cap_pair_min_dist,
                            cap_pair_max_dist        = cfg.cap_pair_max_dist,
                            max_vias                 = cfg.n_max_vias,
                            via_flip_max_count       = cfg.via_flip_max_count,
                            single_inductor_value    = cfg.single_inductor_value,
                        )
                        if mutated is not None and is_feasible(mutated.matrix, 0, cfg.max_inductors, min_switches=cfg.min_switches, max_switches=cfg.max_switches, max_vias=cfg.n_max_vias):
                            exploit_candidates.append(mutated)

                # 2. Keep-bad: randomly sample from the bottom 50% of non-elites
                kept_bad: List[DesignState] = []
                kept_bad_costs: List[float] = []
                if cfg.n_keep_bad > 0:
                    non_elite   = sorted_indices[current_n_elite:]
                    bottom_half = non_elite[len(non_elite) // 2:]
                    if bottom_half:
                        chosen         = self.rng.sample(bottom_half, min(cfg.n_keep_bad, len(bottom_half)))
                        kept_bad       = [population[i].copy() for i in chosen]
                        kept_bad_costs = [costs[i]             for i in chosen]

                # 2b. State champion pool (deduplicated against global elites)
                sc_pool: List[DesignState] = []
                sc_pool_costs: List[float] = []
                sc_pool_sim_tags: List[Optional[str]] = []
                if cfg.use_state_champions:
                    sc_budget = max(0, current_pop_size - current_n_elite - len(kept_bad))
                    for si, champ in enumerate(state_champions):
                        if champ is None or len(sc_pool) >= sc_budget:
                            break
                        if not any(champ.matrix == e.matrix for e in elites + sc_pool):
                            sc_pool.append(champ.copy())
                            # Use total cost so the population costs list stays on the same scale
                            sc_pool_costs.append(state_champion_total_costs[si])
                            sc_pool_sim_tags.append(state_champion_sim_tags[si])

                # 3. Produce regular offspring via crossover + mutation
                offspring: List[DesignState] = []
                offspring_mutation_log: List[dict] = []   # one entry per offspring
                n_offspring = current_pop_size - current_n_elite - len(kept_bad) - len(sc_pool) - len(exploit_candidates)
                n_regular   = n_offspring - cfg.n_immigrants
                max_tries   = n_offspring * 10
                attempts    = 0
                while len(offspring) < n_regular and attempts < max_tries:
                    attempts += 1

                    # Biased parent 1: with p_champion_parent, pick from state champions
                    available_champ_pairs = [
                        (si, c) for si, c in enumerate(state_champions) if c is not None
                    ]
                    if (cfg.p_champion_parent > 0
                            and available_champ_pairs
                            and self.rng.random() < cfg.p_champion_parent):
                        si_chosen, champ = self.rng.choice(available_champ_pairs)
                        p1 = champ.copy()
                        p1_from_champion = True
                        p1_cost_override = state_champion_total_costs[si_chosen]
                        p1_sim_tag = state_champion_sim_tags[si_chosen]
                    else:
                        p1_idx = _tournament_select_idx(population, costs, cfg.tournament_size, self.rng)
                        p1 = population[p1_idx]
                        p1_from_champion = False
                        p1_cost_override = None
                        p1_sim_tag = population_sim_tags[p1_idx]
                    p2_idx = _tournament_select_idx(population, costs, cfg.tournament_size, self.rng)
                    p2 = population[p2_idx]
                    p2_sim_tag = population_sim_tags[p2_idx]

                    # Crossover
                    did_crossover = self.rng.random() < cfg.p_crossover
                    if did_crossover:
                        _crossover_kwargs = dict(
                            parent1      = p1,
                            parent2      = p2,
                            rng          = self.rng,
                            min_switches = cfg.min_switches,
                            max_switches = cfg.max_switches,
                            max_inductors      = cfg.max_inductors,
                            n_states           = cfg.n_states,
                            cap_min            = cfg.cap_min,
                            cap_max            = cfg.cap_max,
                            n_inductor_values  = cfg.n_inductor_values,
                            max_vias           = cfg.n_max_vias,
                            max_cap_pairs      = cfg.max_cap_pairs if cfg.enable_cap_pairs else 0,
                            cap_pair_min_dist  = cfg.cap_pair_min_dist,
                            cap_pair_max_dist  = cfg.cap_pair_max_dist,
                            single_inductor_value = cfg.single_inductor_value,
                        )
                        if cfg.crossover_method == "weighted_clustering":
                            child = crossover_weighted_clustering(
                                **_crossover_kwargs,
                                n_centroids_max = cfg.n_centroids,
                            )
                        elif cfg.crossover_method == "block":
                            child = crossover_block(
                                **_crossover_kwargs,
                                block_min_size = cfg.block_min_size,
                                block_max_size = cfg.block_max_size,
                            )
                        else:
                            child = crossover_zig_zag(**_crossover_kwargs)
                        if child is None:
                            continue   # crossover failed — retry
                    else:
                        # Clone fitter parent — use override cost if p1 came from champion pool
                        if p1_from_champion and p1_cost_override is not None:
                            p1_cost = p1_cost_override
                        else:
                            p1_cost = costs[p1_idx] if not p1_from_champion else 1e9
                        p2_cost = costs[p2_idx]
                        child = (p1 if p1_cost <= p2_cost else p2).copy()

                    # Mutation
                    mutation_entry: dict = {
                        "offspring_idx":    len(offspring),
                        "crossover":        did_crossover,
                        "p1_from_champion": p1_from_champion,
                        "p1_sim_tag":       p1_sim_tag,
                        "p2_sim_tag":       p2_sim_tag,
                        "mutation":         None,
                    }
                    if self.rng.random() < cfg.p_mutation:
                        mutated, mut_info = random_action(
                            state                    = child,
                            n_regular_flip           = cfg.n_regular_flip,
                            max_inductors            = cfg.max_inductors,
                            cap_min                  = cfg.cap_min,
                            cap_max                  = cfg.cap_max,
                            cap_sigma                = cfg.cap_sigma,
                            rng                      = self.rng,
                            n_inductor_values        = cfg.n_inductor_values,
                            action_weights           = cfg.action_weights,
                            max_cap_pairs            = cfg.max_cap_pairs if cfg.enable_cap_pairs else 0,
                            cap_pair_min             = cfg.cap_pair_min,
                            cap_pair_max             = cfg.cap_pair_max,
                            cap_pair_sigma           = cfg.cap_pair_sigma,
                            cap_pair_min_dist        = cfg.cap_pair_min_dist,
                            cap_pair_max_dist        = cfg.cap_pair_max_dist,
                            max_vias                 = cfg.n_max_vias,
                            via_flip_max_count       = cfg.via_flip_max_count,
                            single_inductor_value    = cfg.single_inductor_value,
                        )
                        if mutated is not None:
                            child = mutated
                            mutation_entry["mutation"] = mut_info

                    if not is_feasible(child.matrix, 0, cfg.max_inductors, min_switches=cfg.min_switches, max_switches=cfg.max_switches, max_vias=cfg.n_max_vias):
                        continue

                    offspring.append(child)
                    offspring_mutation_log.append(mutation_entry)

                # 4. Random immigrants
                immigrants: List[DesignState] = []
                for _ in range(cfg.n_immigrants):
                    immigrant = generate_random_design(
                        rows                = cfg.rows,
                        cols                = cfg.cols,
                        n_switches          = self.rng.randint(cfg.init_min_switches, cfg.init_max_switches),
                        max_inductors       = cfg.max_inductors,
                        n_initial_inductors = self.rng.randint(cfg.init_min_inductors, cfg.init_max_inductors),
                        n_states            = cfg.n_states,
                        cap_min             = cfg.cap_min,
                        cap_max             = cfg.cap_max,
                        rng                 = self.rng,
                        n_inductor_values   = cfg.n_inductor_values,
                        n_init_vias         = self.rng.randint(cfg.init_min_vias, cfg.init_max_vias),
                        max_vias            = cfg.n_max_vias,
                        max_cap_pairs       = cfg.max_cap_pairs if cfg.enable_cap_pairs else 0,
                        cap_pair_init_min   = cfg.cap_pair_init_min if cfg.enable_cap_pairs else 0,
                        cap_pair_init_max   = cfg.cap_pair_init_max if cfg.enable_cap_pairs else 0,
                        cap_pair_min        = cfg.cap_pair_min,
                        cap_pair_max        = cfg.cap_pair_max,
                        cap_pair_min_dist   = cfg.cap_pair_min_dist,
                        cap_pair_max_dist   = cfg.cap_pair_max_dist,
                        single_inductor_value = cfg.single_inductor_value,
                    )
                    if immigrant is not None:
                        immigrants.append(immigrant)

                # 5. Evaluate all new individuals in one batch
                to_evaluate = offspring + immigrants + exploit_candidates
                if cfg.use_state_champions:
                    new_costs, new_per_state, new_sim_data = self._evaluate_batch_with_state_costs(
                        engine, to_evaluate, gen_idx=gen
                    )
                    # Update state champions from newly evaluated individuals
                    for local_idx, psc in enumerate(new_per_state):
                        for si, sc in enumerate(psc):
                            if sc < state_champion_costs[si]:
                                state_champion_costs[si]       = sc
                                state_champion_total_costs[si] = new_costs[local_idx]
                                state_champions[si] = to_evaluate[local_idx].copy()
                                new_sim_tag = f"gen{gen}_ind{local_idx}"
                                state_champion_sim_tags[si] = new_sim_tag
                                # Build provenance from offspring mutation log if available
                                if local_idx < len(offspring_mutation_log):
                                    entry = offspring_mutation_log[local_idx]
                                    _prov = {
                                        "generation":            gen,
                                        "sim_tag":               new_sim_tag,
                                        "used_crossover":        entry.get("crossover", False),
                                        "crossover_method":      cfg.crossover_method if entry.get("crossover") else None,
                                        "parent1_sim_tag":       entry.get("p1_sim_tag"),
                                        "parent2_sim_tag":       entry.get("p2_sim_tag"),
                                        "parent1_from_champion": entry.get("p1_from_champion"),
                                        "mutation_action":       entry["mutation"].get("action") if entry.get("mutation") else None,
                                        "mutation_description":  entry["mutation"].get("description") if entry.get("mutation") else None,
                                    }
                                else:
                                    _kind = "immigrant" if local_idx < len(offspring) + len(immigrants) else "exploit candidate"
                                    _prov = {
                                        "generation":            gen,
                                        "sim_tag":               new_sim_tag,
                                        "used_crossover":        False,
                                        "crossover_method":      None,
                                        "parent1_sim_tag":       None,
                                        "parent2_sim_tag":       None,
                                        "parent1_from_champion": None,
                                        "mutation_action":       None,
                                        "mutation_description":  _kind,
                                    }
                                self._save_state_best(
                                    state_idx      = si,
                                    design         = to_evaluate[local_idx],
                                    per_state_cost = sc,
                                    total_cost     = new_costs[local_idx],
                                    generation     = gen,
                                    sim_tag        = new_sim_tag,
                                    provenance     = _prov,
                                )
                else:
                    new_costs, new_sim_data = self._evaluate_batch(engine, to_evaluate, gen_idx=gen)
                offspring_costs  = new_costs[:len(offspring)]
                immigrant_costs  = new_costs[len(offspring):len(offspring) + len(immigrants)]
                exploit_costs    = new_costs[len(offspring) + len(immigrants):]

                # 6. Assemble new population: elites | state_champs | kept_bad | offspring | immigrants | exploit
                population = elites + sc_pool + kept_bad + offspring + immigrants + exploit_candidates
                costs      = elite_costs + sc_pool_costs + kept_bad_costs + offspring_costs + immigrant_costs + exploit_costs

                # Rebuild population_sim_tags in parallel with the same segment ordering
                _kept_bad_tags = [population_sim_tags[i] for i in (chosen if cfg.n_keep_bad > 0 and kept_bad else [])]
                _n_off = len(offspring)
                _n_imm = len(immigrants)
                _n_exp = len(exploit_candidates)
                population_sim_tags = (
                    [population_sim_tags[i] for i in sorted_indices[:current_n_elite]]
                    + sc_pool_sim_tags
                    + _kept_bad_tags
                    + [f"gen{gen}_ind{j}" for j in range(_n_off)]
                    + [f"gen{gen}_ind{_n_off + j}" for j in range(_n_imm)]
                    + [f"gen{gen}_ind{_n_off + _n_imm + j}" for j in range(_n_exp)]
                )

                if exploit_candidates:
                    print(f"[GA] gen {gen:4d}  exploit: {len(exploit_candidates)} candidates, "
                          f"best_exploit={min(exploit_costs):.4f}")

                # Track global best
                gen_best_idx = min(range(len(costs)), key=lambda i: costs[i])
                if costs[gen_best_idx] < best_cost:
                    best_design = population[gen_best_idx].copy()
                    best_cost   = costs[gen_best_idx]
                    # Compute population-slot offset to find the sim folder index.
                    # Slots: elites | sc_pool | kept_bad | to_evaluate[0..)
                    carried_count = current_n_elite + len(sc_pool) + len(kept_bad)
                    if gen_best_idx >= carried_count:
                        best_local_idx = gen_best_idx - carried_count
                        best_sim_tag   = f"gen{gen}_ind{best_local_idx}"
                        best_s2p_files, best_assignment = new_sim_data[best_local_idx]
                    else:
                        best_sim_tag = f"elite/carried_gen{gen}_slot{gen_best_idx}"
                        # Carried elite: s2p files not in new_sim_data; skip breakdown
                        best_s2p_files, best_assignment = [], None
                    self._save_best(best_design, best_cost, generation=gen,
                                    sim_tag=best_sim_tag,
                                    s2p_files=best_s2p_files, assignment=best_assignment)

                # Stats
                diversity = _population_diversity(population)
                mean_cost  = sum(costs) / len(costs)
                worst_cost = max(costs)
                elapsed    = time.time() - start_time

                # Filtered stats: exclude very high costs (poor RF performance)
                thresh = cfg.feasibility_cost_threshold
                feasible_costs = [c for c in costs if c < thresh]
                if feasible_costs:
                    feasible_mean_cost  = sum(feasible_costs) / len(feasible_costs)
                    feasible_worst_cost = max(feasible_costs)
                else:
                    feasible_mean_cost  = mean_cost
                    feasible_worst_cost = worst_cost

                self._log_generation(gen, best_cost, best_sim_tag, mean_cost,
                                     worst_cost, diversity, elapsed,
                                     offspring_mutation_log,
                                     pop_costs=costs,
                                     feasible_mean_cost=feasible_mean_cost,
                                     feasible_worst_cost=feasible_worst_cost)
                self._save_progress()
                self._save_population_snapshot(population, costs, gen=gen)
                self._plot_progress()
                self._plot_population_behavior()

                if cfg.use_state_champions:
                    sc_costs_str = "  ".join(
                        f"sc{si}={c:.4f}" for si, c in enumerate(state_champion_costs)
                    )
                    print(
                        f"[GA] gen {gen:4d}/{cfg.n_gen}  pop={current_pop_size}  elite={current_n_elite}  "
                        f"best={best_cost:.4f}  mean={mean_cost:.4f}  "
                        f"worst={worst_cost:.4f}  div={diversity:.3f}  "
                        f"{sc_costs_str}  "
                        f"{elapsed:.1f}s"
                    )
                else:
                    print(
                        f"[GA] gen {gen:4d}/{cfg.n_gen}  pop={current_pop_size}  elite={current_n_elite}  "
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
        effective_max_cap_pairs = cfg.max_cap_pairs if cfg.enable_cap_pairs else 0

        # --- Phase 1: fine-tune seeding (optional) ---
        if cfg.finetune_design_path:
            seed_design = _load_design_from_config(cfg.finetune_design_path)
            # Normalize seed design if single_inductor_value is enabled
            if cfg.single_inductor_value and seed_design.inductor_values:
                seed_design.inductor_values = [seed_design.inductor_values[0]] * len(seed_design.inductor_values)
            n_seeded = round(cfg.pop_size_start * cfg.finetune_seed_ratio)
            print(f"[GA] Fine-tune mode: seeding {n_seeded}/{cfg.pop_size_start} "
                  f"individuals by mutating {cfg.finetune_design_path}")
            _MAX_SEED_RETRIES = 200
            for i in range(n_seeded):
                mutant = None
                for _ in range(_MAX_SEED_RETRIES):
                    candidate, _ = random_action(
                        state                    = seed_design,
                        n_regular_flip           = cfg.n_regular_flip,
                        max_inductors            = cfg.max_inductors,
                        cap_min                  = cfg.cap_min,
                        cap_max                  = cfg.cap_max,
                        cap_sigma                = cfg.cap_sigma,
                        rng                      = self.rng,
                        n_inductor_values        = cfg.n_inductor_values,
                        action_weights           = cfg.action_weights,
                        max_cap_pairs            = effective_max_cap_pairs,
                        cap_pair_min             = cfg.cap_pair_min,
                        cap_pair_max             = cfg.cap_pair_max,
                        cap_pair_sigma           = cfg.cap_pair_sigma,
                        cap_pair_min_dist        = cfg.cap_pair_min_dist,
                        cap_pair_max_dist        = cfg.cap_pair_max_dist,
                        max_switch_state_flips   = 3,
                        max_inductor_state_flips = 3,
                        max_vias                 = cfg.n_max_vias,
                        via_flip_max_count       = cfg.via_flip_max_count,
                        single_inductor_value    = cfg.single_inductor_value,
                    )
                    if candidate is not None:
                        mutant = candidate
                        break
                if mutant is None:
                    raise RuntimeError(
                        f"Failed to generate seeded individual {i} after "
                        f"{_MAX_SEED_RETRIES} retries. Check action weights and seed design."
                    )
                population.append(mutant)
        else:
            n_seeded = 0

        # --- Phase 2: random individuals ---
        n_random = cfg.pop_size_start - n_seeded
        for i in range(n_random):
            design = generate_random_design(
                rows                = cfg.rows,
                cols                = cfg.cols,
                n_switches          = self.rng.randint(cfg.init_min_switches, cfg.init_max_switches),
                max_inductors       = cfg.max_inductors,
                n_initial_inductors = self.rng.randint(cfg.init_min_inductors, cfg.init_max_inductors),
                n_states            = cfg.n_states,
                cap_min             = cfg.cap_min,
                cap_max             = cfg.cap_max,
                rng                 = self.rng,
                n_inductor_values   = cfg.n_inductor_values,
                n_init_vias         = self.rng.randint(cfg.init_min_vias, cfg.init_max_vias),
                max_vias            = cfg.n_max_vias,
                max_cap_pairs       = effective_max_cap_pairs,
                cap_pair_init_min   = cfg.cap_pair_init_min if cfg.enable_cap_pairs else 0,
                cap_pair_init_max   = cfg.cap_pair_init_max if cfg.enable_cap_pairs else 0,
                cap_pair_min        = cfg.cap_pair_min,
                cap_pair_max        = cfg.cap_pair_max,
                cap_pair_min_dist   = cfg.cap_pair_min_dist,
                cap_pair_max_dist   = cfg.cap_pair_max_dist,
                single_inductor_value = cfg.single_inductor_value,
            )
            if design is None:
                raise RuntimeError(
                    f"Failed to generate random individual {i} of the initial population. "
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
    ) -> Tuple[List[float], List[Tuple[List, Optional[List[int]]]]]:
        """Returns (costs, sim_data) where sim_data[i] = (s2p_files, assignment)."""
        if not individuals:
            return [], []
        overrides = [
            self._build_override(d, gen_idx, i)
            for i, d in enumerate(individuals)
        ]
        results = engine.run_batch(overrides)
        costs: List[float] = []
        sim_data: List[Tuple[List, Optional[List[int]]]] = []
        for r in results:
            c, assignment = self._cost_from_result(r)
            costs.append(c)
            sim_data.append((r.s_param_files if r.success else [], assignment))
            self._write_sim_results(r, c, assignment)
        return costs, sim_data

    def _cost_from_result(
        self, result: "RunResult"
    ) -> Tuple[float, Optional[List[int]]]:
        """
        Returns (total_cost, assignment) where assignment[spec_idx] = sim_idx,
        or (1e9, None) on failure.  Uses the Hungarian algorithm when
        dynamic_state_assignment is enabled.
        """
        if not result.success:
            return 1e9, None
        if self.cfg.dynamic_state_assignment:
            cost, assignment = compute_total_cost_dynamic(result.s_param_files, self.specs)
            return cost, assignment
        return compute_total_cost(result.s_param_files, self.specs), None

    def _per_state_costs_from_result(
        self,
        result: "RunResult",
        assignment: Optional[List[int]],
    ) -> List[float]:
        """
        Return per-spec cost list (parallel to self.specs). 1e9 on failure.

        Parameters
        ----------
        assignment : assignment[spec_idx] = sim_idx from dynamic assignment,
                     or None to fall back to spec.state_index (static).
        """
        n = len(self.specs)
        if not result.success:
            return [1e9] * n
        per_state: List[float] = []
        for spec_idx, spec in enumerate(self.specs):
            # Use dynamic assignment if available, else fall back to state_index
            sim_idx = assignment[spec_idx] if assignment is not None else spec.state_index
            if sim_idx >= len(result.s_param_files) or result.s_param_files[sim_idx] is None:
                per_state.append(1e9 * spec.weight)
                continue
            try:
                freqs, s21_db_vals = get_s21_db(result.s_param_files[sim_idx])
                per_state.append(compute_state_cost(freqs, s21_db_vals, spec))
            except Exception:
                per_state.append(1e9 * spec.weight)
        return per_state

    def _write_sim_results(
        self,
        result: "RunResult",
        cost: float,
        assignment: Optional[List[int]],
    ) -> None:
        """Write cost.txt and (if dynamic assignment) state_assignment.json to the sim dir."""
        (result.run_dir / "cost.txt").write_text(f"{cost}\n")
        if assignment is not None:
            spec_labels = [
                {
                    "spec_idx":    si,
                    "state_index": spec.state_index,
                    "passband_hz": list(spec.passband),
                    "sim_idx":     assignment[si],
                }
                for si, spec in enumerate(self.specs)
            ]
            data = {
                "cost":       cost,
                "assignment": assignment,   # assignment[spec_idx] = sim_idx
                "details":    spec_labels,
            }
            (result.run_dir / "state_assignment.json").write_text(
                json.dumps(data, indent=2)
            )

    def _evaluate_batch_with_state_costs(
        self,
        engine: "ParallelVirtuosoEngine",
        individuals: List[DesignState],
        gen_idx: int,
    ) -> Tuple[List[float], List[List[float]], List[Tuple[List, Optional[List[int]]]]]:
        """
        Like _evaluate_batch but also returns per-state costs and sim data.

        Returns
        -------
        costs           : List[float]         — total cost per individual
        per_state_costs : List[List[float]]   — [individual_idx][spec_idx]
        sim_data        : List[(s2p_files, assignment)] — one entry per individual
        """
        if not individuals:
            return [], [], []
        overrides = [
            self._build_override(d, gen_idx, i)
            for i, d in enumerate(individuals)
        ]
        results = engine.run_batch(overrides)
        costs: List[float] = []
        per_state_costs: List[List[float]] = []
        sim_data: List[Tuple[List, Optional[List[int]]]] = []
        for r in results:
            c, assignment = self._cost_from_result(r)
            costs.append(c)
            per_state_costs.append(self._per_state_costs_from_result(r, assignment))
            sim_data.append((r.s_param_files if r.success else [], assignment))
            self._write_sim_results(r, c, assignment)
        return costs, per_state_costs, sim_data

    # ------------------------------------------------------------------
    # Persistence & plotting
    # ------------------------------------------------------------------

    def _save_best(
        self,
        design: DesignState,
        cost: float,
        generation: int,
        sim_tag: str = "",
        s2p_files: Optional[List] = None,
        assignment: Optional[List[int]] = None,
    ) -> None:
        sim_dir = str(self.output_dir / "sims" / sim_tag) if sim_tag else ""

        # Compute cost breakdown when s2p files are available
        breakdown: Optional[List[dict]] = None
        if s2p_files:
            try:
                breakdown = compute_cost_breakdown(s2p_files, self.specs, assignment)
            except Exception as exc:  # noqa: BLE001
                print(f"  [GA] WARNING: cost breakdown failed: {exc}")

        data = {
            "generation":            generation,
            "sim_tag":               sim_tag,
            "sim_dir":               sim_dir,
            "cost":                  cost,
            "cost_breakdown":        breakdown,
            "matrix":                design.matrix,
            "switch_states":         design.switch_states,
            "cap_values":            design.cap_values,
            "inductor_values":       design.inductor_values,
            "inductor_switch_states": design.inductor_switch_states,
        }
        path = self.output_dir / "best_design.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"  [GA] Best design saved → {path}  cost={cost:.4f}  sim={sim_tag}")

        # Print breakdown table
        if breakdown:
            self._print_cost_breakdown(breakdown)

    def _save_state_best(
        self,
        state_idx: int,
        design: "DesignState",
        per_state_cost: float,
        total_cost: float,
        generation: int,
        sim_tag: str,
        provenance: dict,
    ) -> None:
        """Save the best design found so far for a specific filter state index."""
        sim_dir = str(self.output_dir / "sims" / sim_tag) if sim_tag else ""
        data = {
            "state_idx":             state_idx,
            "generation":            generation,
            "sim_tag":               sim_tag,
            "sim_dir":               sim_dir,
            "per_state_cost":        per_state_cost,
            "total_cost":            total_cost,
            "provenance":            provenance,
            "matrix":                design.matrix,
            "switch_states":         design.switch_states,
            "cap_values":            design.cap_values,
            "inductor_values":       design.inductor_values,
            "inductor_switch_states": design.inductor_switch_states,
        }
        path = self.output_dir / "state_best" / f"state_{state_idx}_best.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"  [GA] State {state_idx} best saved → cost={per_state_cost:.4f}  sim={sim_tag}")

    @staticmethod
    def _print_cost_breakdown(breakdown: List[dict]) -> None:
        """Print a formatted cost breakdown table to stdout."""
        has_s11 = any(b.get("s11_passband", 0) + b.get("s11_stopband", 0) + b.get("s11_crossband", 0) > 0
                      for b in breakdown)
        header = (
            f"  {'Spec':>4}  {'Passband(GHz)':>14}  "
            f"{'S21-PB':>8}  {'S21-SB':>8}  {'S21-CB':>8}"
        )
        if has_s11:
            header += f"  {'S11-PB':>8}  {'S11-SB':>8}  {'S11-CB':>8}"
        header += f"  {'State':>8}"
        print(f"  {'─'*len(header.rstrip())}")
        print(header)
        print(f"  {'─'*len(header.rstrip())}")
        for b in breakdown:
            pb_ghz = f"{b['passband_hz'][0]/1e9:.0f}–{b['passband_hz'][1]/1e9:.0f}"
            row = (
                f"  {b['spec_idx']:>4}  {pb_ghz:>14}  "
                f"{b['s21_passband']:>8.4f}  {b['s21_stopband']:>8.4f}  {b['s21_crossband']:>8.4f}"
            )
            if has_s11:
                row += (
                    f"  {b['s11_passband']:>8.4f}  {b['s11_stopband']:>8.4f}  {b['s11_crossband']:>8.4f}"
                )
            row += f"  {b['total']:>8.4f}"
            print(row)
        total = sum(b["total"] for b in breakdown)
        print(f"  {'─'*len(header.rstrip())}")
        ncols = 3 + (3 if has_s11 else 0)
        pad = "  " + " "*4 + "  " + " "*14 + "  " + "  ".join([" "*8]*ncols) + "  "
        print(f"{pad}{total:>8.4f}  (total)")

    def _log_generation(
        self,
        gen: int,
        best_cost: float,
        best_sim_tag: str,
        mean_cost: float,
        worst_cost: float,
        diversity: float,
        elapsed: float,
        mutation_log: Optional[List[dict]] = None,
        pop_costs: Optional[List[float]] = None,
        feasible_mean_cost: Optional[float] = None,
        feasible_worst_cost: Optional[float] = None,
    ) -> None:
        # Summarise mutation action counts for compact reporting
        mutation_counts: Dict[str, int] = {}
        n_mutated = 0
        n_crossover = 0
        if mutation_log:
            for entry in mutation_log:
                if entry.get("crossover"):
                    n_crossover += 1
                if entry.get("mutation"):
                    n_mutated += 1
                    action_name = entry["mutation"].get("action", "unknown")
                    mutation_counts[action_name] = mutation_counts.get(action_name, 0) + 1

        self._progress.append({
            "gen":                  gen,
            "best_cost":            best_cost,
            "best_sim_tag":         best_sim_tag,
            "mean_cost":            mean_cost,
            "worst_cost":           worst_cost,
            "feasible_mean_cost":   feasible_mean_cost if feasible_mean_cost is not None else mean_cost,
            "feasible_worst_cost":  feasible_worst_cost if feasible_worst_cost is not None else worst_cost,
            "pop_costs":            list(pop_costs) if pop_costs is not None else [],
            "diversity":            diversity,
            "elapsed_s":            elapsed,
            "n_offspring":          len(mutation_log) if mutation_log is not None else None,
            "n_crossover":          n_crossover,
            "n_mutated":            n_mutated,
            "mutation_counts":      mutation_counts,
            "mutations":            mutation_log if mutation_log is not None else [],
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

        gens        = [e["gen"]                 for e in self._progress]
        best_costs  = [e["best_cost"]            for e in self._progress]
        mean_costs  = [e["feasible_mean_cost"]   for e in self._progress]
        worst_costs = [e["feasible_worst_cost"]  for e in self._progress]
        diversity   = [e["diversity"]            for e in self._progress]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        thresh = self.cfg.feasibility_cost_threshold
        ax1.plot(gens, best_costs,  label="best",  linewidth=2, color="tab:green")
        ax1.plot(gens, mean_costs,  label=f"mean (cost<{thresh:.0e})",
                 linewidth=1.5, color="tab:blue", linestyle="--")
        ax1.plot(gens, worst_costs, label=f"worst (cost<{thresh:.0e})",
                 linewidth=1, color="tab:red", linestyle=":")
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

    def _plot_population_behavior(self) -> None:
        """Save population behavior plots: cost density heatmap + top-k trajectories."""
        if not self._progress or not any(e.get("pop_costs") for e in self._progress):
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import numpy as np
        except ImportError:
            return

        gens = [e["gen"] for e in self._progress]
        all_pop_costs = [e.get("pop_costs", []) for e in self._progress]
        thresh = self.cfg.feasibility_cost_threshold
        k = self.cfg.population_plot_top_k

        # Clip costs at threshold for display
        all_clipped = [
            [min(c, thresh) for c in pc] for pc in all_pop_costs
        ]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

        # --- Panel 1: Cost density heatmap ---
        # Determine common y-axis range from non-empty generations
        all_vals = [c for pc in all_clipped for c in pc]
        if all_vals:
            y_min = min(all_vals)
            y_max = max(all_vals)
            n_bins = 40
            bin_edges = np.linspace(y_min, y_max, n_bins + 1)
            heatmap = np.zeros((n_bins, len(gens)))
            for gi, pc in enumerate(all_clipped):
                if pc:
                    counts, _ = np.histogram(pc, bins=bin_edges)
                    heatmap[:, gi] = counts

            # Use LogNorm to handle wide count ranges; add 1 to avoid log(0)
            norm = mcolors.LogNorm(vmin=1, vmax=max(heatmap.max(), 1))
            im = ax1.imshow(
                heatmap,
                aspect="auto",
                origin="lower",
                extent=[gens[0] - 0.5, gens[-1] + 0.5, y_min, y_max],
                norm=norm,
                cmap="YlOrRd",
            )
            fig.colorbar(im, ax=ax1, label="Individual count")
        ax1.set_ylabel("Cost")
        ax1.set_title("Population Cost Density per Generation")

        # --- Panel 2: Top-k trajectories ---
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, k))
        for rank in range(k):
            traj = []
            traj_gens = []
            for gi, pc in enumerate(all_pop_costs):
                if len(pc) > rank:
                    sorted_costs = sorted(pc)
                    traj.append(sorted_costs[rank])
                    traj_gens.append(gens[gi])
            if not traj:
                continue
            lw = 2.0 if rank == 0 else 1.2
            alpha = 1.0 if rank == 0 else 0.55
            label = f"Top-{rank+1}"
            ax2.plot(traj_gens, traj, color=colors[rank], linewidth=lw,
                     alpha=alpha, label=label)

        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Cost")
        ax2.set_title(f"Top-{k} Individual Trajectories")
        ax2.legend(loc="upper right", fontsize="small")
        ax2.grid(True, alpha=0.4)

        fig.tight_layout()
        fig.savefig(self.output_dir / "ga_population_behavior.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_ga(config_path: str | Path) -> Tuple[DesignState, float]:
    """Run the full GA optimisation from a config file path."""
    optimizer = SwitchableFilterGA(config_path)
    return optimizer.run()
