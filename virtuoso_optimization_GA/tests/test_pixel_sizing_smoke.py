#!/usr/bin/env python3
"""
Pure-Python logic smoke test for flexible per-pixel sizing.

Exercises (no Virtuoso/EMX):
  1. Sized connectivity vs legacy connectivity equivalence + disconnection.
  2. generate_random_design(enable_pixel_sizing=True) shape + feasibility.
  3. action_perturb_pixel_sizes behaviour (incl. no-op when disabled).
  4. Each crossover carries a shape-consistent sizing matrix.
  5. to_pixelated_config() emits pixel_sizes only when populated.
  6. build_skill_pixel_sizes_list defaults to uniform == legacy.
  7. Backward-compat: legacy (uniform) design path unchanged.

Run:  python virtuoso_optimization_GA/tests/test_pixel_sizing_smoke.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from virtuoso_optimization_SA.design import DesignState
from virtuoso_optimization_SA.feasibility import (
    check_connectivity,
    check_connectivity_sized,
    _patches_overlap,
    generate_random_design,
    is_feasible,
)
from virtuoso_optimization_SA.actions import action_perturb_pixel_sizes
from virtuoso_optimization_GA.crossover import crossover_zig_zag
from virtuoso_optimization_GA.crossover_block import crossover_block
from virtuoso_optimization_GA.crossover_weighted_clustering import (
    crossover_weighted_clustering,
)
from virtuoso_engine.pixel_matrix import build_skill_pixel_sizes_list

GRID = 25.0
MARGIN = 2.0
SMIN, SMAX = 25.0, 37.0

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def shape_ok(sizes, rows, cols) -> bool:
    return len(sizes) == rows and all(len(r) == cols for r in sizes)


# ---------------------------------------------------------------------------
# 1. Sized connectivity
# ---------------------------------------------------------------------------
print("[1] Sized connectivity")

# Simple horizontal bridge: feed(0)–metal–metal–feed
mat = [[1, 1, 1, 1]]
uniform = [[31.0] * 4]
check("overlap formula 31/31", _patches_overlap(31.0, 31.0, GRID, MARGIN))
check("no overlap 25/25", not _patches_overlap(25.0, 25.0, GRID, MARGIN))
check("uniform 31 == legacy connected",
      check_connectivity_sized(mat, uniform, GRID, MARGIN) == check_connectivity(mat))

# Shrink the interior bridge cell so overlap < margin → disconnected.
broken = [[31.0, 25.0, 25.0, 31.0]]
check("shrunk bridge disconnects",
      not check_connectivity_sized(mat, broken, GRID, MARGIN))

# Equivalence over random matrices when sizes are uniform & well-overlapping.
rng = random.Random(123)
equiv = True
for _ in range(200):
    R, C = 4, 5
    m = [[rng.choice([0, 1]) for _ in range(C)] for _ in range(R)]
    # ensure feed columns single non-zero
    for col in (0, C - 1):
        for r in range(R):
            m[r][col] = 0
        m[rng.randrange(R)][col] = 1
    u = [[31.0] * C for _ in range(R)]
    if check_connectivity_sized(m, u, GRID, MARGIN) != check_connectivity(m):
        equiv = False
        break
check("uniform-size connectivity matches legacy over 200 random matrices", equiv)


# ---------------------------------------------------------------------------
# 2. generate_random_design with sizing
# ---------------------------------------------------------------------------
print("[2] generate_random_design(enable_pixel_sizing=True)")
rng = random.Random(7)
d = generate_random_design(
    rows=6, cols=9, n_switches=2, max_inductors=1, n_states=3,
    cap_min=100e-15, cap_max=1000e-15, rng=rng,
    n_inductor_values=1, single_inductor_value=True, n_initial_inductors=1,
    enable_pixel_sizing=True, pixel_size_min=SMIN, pixel_size_max=SMAX,
    grid_size=GRID, connect_margin=MARGIN,
)
check("design generated", d is not None)
if d is not None:
    check("pixel_sizes shape == matrix", shape_ok(d.pixel_sizes, d.rows, d.cols))
    check("all sizes within [min,max]",
          all(SMIN <= v <= SMAX for row in d.pixel_sizes for v in row))
    check("sized feasibility holds",
          is_feasible(d.matrix, 0, 1, min_switches=1, max_switches=2,
                      pixel_sizes=d.pixel_sizes, grid_size=GRID, connect_margin=MARGIN))


# ---------------------------------------------------------------------------
# 3. action_perturb_pixel_sizes
# ---------------------------------------------------------------------------
print("[3] action_perturb_pixel_sizes")
rng = random.Random(11)
mutated, info = action_perturb_pixel_sizes(d, 4, 2.0, SMIN, SMAX, GRID, MARGIN, rng)
check("perturb returns a design", mutated is not None)
if mutated is not None:
    changed = sum(
        1 for r in range(d.rows) for c in range(d.cols)
        if abs(mutated.pixel_sizes[r][c] - d.pixel_sizes[r][c]) > 1e-9
    )
    check("at least one size changed", changed >= 1, f"changed={changed}")
    check("perturbed design still sized-feasible",
          is_feasible(mutated.matrix, 0, 1, min_switches=1, max_switches=2,
                      pixel_sizes=mutated.pixel_sizes, grid_size=GRID, connect_margin=MARGIN))
    check("original design unchanged (copy semantics)",
          d.pixel_sizes != mutated.pixel_sizes)

# No-op when sizing disabled
legacy = DesignState(matrix=[[1, 1, 1]], switch_states=[[], [], []], cap_values=[])
res, _ = action_perturb_pixel_sizes(legacy, 4, 2.0, SMIN, SMAX, GRID, MARGIN, random.Random(1))
check("perturb is no-op when pixel_sizes empty", res is None)


# ---------------------------------------------------------------------------
# 4. Crossover carries sizing matrix
# ---------------------------------------------------------------------------
print("[4] Crossover carries sizing matrix")
rng = random.Random(99)
parents = []
for _ in range(2):
    p = generate_random_design(
        rows=7, cols=9, n_switches=2, max_inductors=1, n_states=3,
        cap_min=100e-15, cap_max=1000e-15, rng=rng,
        n_inductor_values=1, single_inductor_value=True, n_initial_inductors=1,
        enable_pixel_sizing=True, pixel_size_min=SMIN, pixel_size_max=SMAX,
        grid_size=GRID, connect_margin=MARGIN,
    )
    assert p is not None
    parents.append(p)

xover_common = dict(
    rng=rng, min_switches=1, max_switches=2, max_inductors=1, n_states=3,
    cap_min=100e-15, cap_max=1000e-15, n_inductor_values=1,
    single_inductor_value=True, grid_size=GRID, connect_margin=MARGIN,
)
for name, fn, extra in [
    ("zig_zag", crossover_zig_zag, {}),
    ("weighted_clustering", crossover_weighted_clustering, {"n_centroids_max": 3}),
    ("block", crossover_block, {"block_min_size": 2, "block_max_size": 3}),
]:
    got_child = False
    for _ in range(60):  # sized feasibility may reject some children
        child, _params = fn(parents[0], parents[1], **xover_common, **extra)
        if child is not None:
            got_child = True
            check(f"{name}: child has sizing matrix", bool(child.pixel_sizes))
            check(f"{name}: sizing shape consistent",
                  shape_ok(child.pixel_sizes, child.rows, child.cols))
            check(f"{name}: child sized-feasible",
                  is_feasible(child.matrix, 0, 1, min_switches=1, max_switches=2,
                              pixel_sizes=child.pixel_sizes, grid_size=GRID, connect_margin=MARGIN))
            break
    check(f"{name}: produced at least one child in 60 tries", got_child)


# ---------------------------------------------------------------------------
# 5 & 6. Serialization + SKILL builder
# ---------------------------------------------------------------------------
print("[5/6] Serialization + SKILL list")
cfg_sized = d.to_pixelated_config()
check("sized design serializes pixel_sizes", "pixel_sizes" in cfg_sized)
cfg_legacy = legacy.to_pixelated_config()
check("legacy design omits pixel_sizes", "pixel_sizes" not in cfg_legacy)

# Uniform SKILL list == legacy (every cell = default size).
m2 = [[0, 1, 2], [3, 0, 1]]
skill_uniform = build_skill_pixel_sizes_list(m2, None, 31.0)
check("SKILL uniform fills default everywhere",
      skill_uniform == "list(" + " ".join(["31.0000"] * 6) + ")",
      skill_uniform)
skill_sized = build_skill_pixel_sizes_list(m2, [[25.0, 26.0, 27.0], [28.0, 29.0, 30.0]], 31.0)
check("SKILL sized is row-first floats",
      skill_sized == "list(25.0000 26.0000 27.0000 28.0000 29.0000 30.0000)",
      skill_sized)


# ---------------------------------------------------------------------------
# 7. Backward-compat: legacy generate_random_design unchanged
# ---------------------------------------------------------------------------
print("[7] Backward compatibility (uniform path)")
rng = random.Random(7)
d_legacy = generate_random_design(
    rows=6, cols=9, n_switches=2, max_inductors=1, n_states=3,
    cap_min=100e-15, cap_max=1000e-15, rng=rng,
    n_inductor_values=1, single_inductor_value=True, n_initial_inductors=1,
)  # enable_pixel_sizing defaults False
check("legacy design generated", d_legacy is not None)
if d_legacy is not None:
    check("legacy pixel_sizes empty", d_legacy.pixel_sizes == [])
    check("legacy serialization omits pixel_sizes",
          "pixel_sizes" not in d_legacy.to_pixelated_config())


# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"RESULT: {_passed} passed, {_failed} failed")
print("=" * 60)
sys.exit(1 if _failed else 0)
