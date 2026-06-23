#!/usr/bin/env python3
"""
Pure-Python logic smoke test for the differential multi-layer metal update.

Exercises (no Virtuoso/EMX):
  1. mirror_half_matrix dual flexible output (returns 5 values, two output rows).
  2. mirror_half_layers mirrors the layer channel identically to the type matrix.
  3. generate_random_design(allowed_layers=...) builds a shape-consistent layer
     matrix with layer variety, and honours the flexible right-feed constraint.
  4. action_flip_01 with allowed_layers performs the 4-way state choice and
     records layer transitions.
  5. Each crossover carries a shape-consistent layer matrix.
  6. to_pixelated_config() emits pixel_layers only when populated.
  7. DIFFERENTIAL_FIXED_PORTS / total ports reflect the 8-port dual output, and
     build_skill_pixel_layers_list emits row-first ints (0 when unset).
  8. Backward-compat: legacy paths leave pixel_layers empty and flip stays binary.

Run:  python virtuoso_optimization_GA/tests/test_multilayer_smoke.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from virtuoso_optimization_SA.feasibility import generate_random_design
from virtuoso_optimization_SA.actions import action_flip_01, action_regional_density
from virtuoso_optimization_GA.crossover import crossover_zig_zag
from virtuoso_optimization_GA.crossover_block import crossover_block
from virtuoso_optimization_GA.crossover_weighted_clustering import (
    crossover_weighted_clustering,
)
from virtuoso_engine.pixel_matrix import (
    DIFFERENTIAL_FIXED_PORTS,
    analyze_matrix,
    build_skill_pixel_layers_list,
    differential_total_ports,
    mirror_half_layers,
    mirror_half_matrix,
)

LAYERS = [0, 1, 2]
_passed = 0
_failed = 0


def check(label: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


def _mk(rng, rows=8, cols=7):
    return generate_random_design(
        rows=rows, cols=cols, n_switches=0, max_inductors=0, n_states=1,
        cap_min=0.0, cap_max=0.0, rng=rng, p0=0.4, p1=0.6,
        left_feed_row_max=rows - 3, right_feed_row_max=rows - 3,
        allowed_layers=LAYERS,
    )


def main() -> int:
    rng = random.Random(42)

    print("[1] mirror_half_matrix dual flexible output")
    half = [[1, 1, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 0]]
    out = mirror_half_matrix(half)
    check("returns 5 values", len(out) == 5)
    full, it, ib, ot, ob = out
    check("full has 2R-1 rows", len(full) == 7)
    check("input mirror rows", (it, ib) == (0, 6))
    check("output flexible + mirrored (not pinned to axis)", (ot, ob) == (1, 5))

    print("[2] mirror_half_layers")
    half_layers = [[0, 1, 2, 0], [0, 2, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    fl = mirror_half_layers(half_layers)
    check("layer matrix has 2R-1 rows", len(fl) == 7)
    check("bottom mirror equals top row", fl[6] == fl[0])
    check("empty passthrough", mirror_half_layers([]) == [])

    print("[3] generate_random_design with layers")
    d = _mk(rng)
    check("design generated", d is not None)
    check("pixel_layers shape matches matrix",
          len(d.pixel_layers) == d.rows and all(len(r) == d.cols for r in d.pixel_layers))
    left = [r for r in range(d.rows) if d.matrix[r][0] != 0]
    right = [r for r in range(d.rows) if d.matrix[r][d.cols - 1] != 0]
    check("both feeds avoid bottom two rows",
          all(r <= d.rows - 3 for r in left + right))
    interior = [d.pixel_layers[r][c] for r in range(d.rows)
                for c in range(1, d.cols - 1) if d.matrix[r][c] == 1]
    check("interior metal shows >1 layer value", len(set(interior)) > 1)

    print("[4] action_flip_01 4-way choice")
    nd = None
    for _ in range(30):
        nd, info = action_flip_01(d, n_flip=3, rng=rng, allowed_layers=LAYERS,
                                  flip_region_size=4)
        if nd is not None:
            break
    check("flip feasible", nd is not None)
    check("flip records layer transition", "from_layer" in info["flipped_cells"][0])
    check("flipped layers all valid", all(
        fc["to_layer"] in LAYERS or fc["to"] == 0 for fc in info["flipped_cells"]))

    print("[5] crossover carries layer matrix")
    base = dict(rng=rng, min_switches=0, max_switches=0, max_inductors=0, n_states=1,
                cap_min=0.0, cap_max=0.0, n_inductor_values=1, max_vias=0,
                max_cap_pairs=0, cap_pair_min_dist=2, cap_pair_max_dist=3, grid_size=10.0)
    for name, fn, extra in [
        ("zigzag", crossover_zig_zag, {}),
        ("block", crossover_block, dict(block_min_size=2, block_max_size=4)),
        ("cluster", crossover_weighted_clustering, dict(n_centroids_max=3)),
    ]:
        p1, p2 = _mk(rng), _mk(rng)
        child = None
        for _ in range(40):
            child, _i = fn(p1, p2, **base, **extra)
            if child is not None:
                break
        ok = (child is not None and bool(child.pixel_layers)
              and len(child.pixel_layers) == child.rows
              and all(len(r) == child.cols for r in child.pixel_layers))
        check(f"{name}: child carries shape-consistent layers", ok)

    print("[6] serialization")
    cfg = d.to_pixelated_config()
    check("emits pixel_layers when populated", "pixel_layers" in cfg)

    print("[7] ports + SKILL builder")
    check("8 fixed differential ports", len(DIFFERENTIAL_FIXED_PORTS) == 8)
    check("dual output port names present",
          {"PIX_OUT_T", "PIX_OUT_B"} <= set(DIFFERENTIAL_FIXED_PORTS))
    mi = analyze_matrix(full)
    check("total ports = 8 + n_via", differential_total_ports(mi) == 8)
    skill = build_skill_pixel_layers_list(full, fl)
    check("skill is row-first ints", skill.startswith("list(") and "2" in skill)
    check("skill defaults to 0 when layers None",
          set(build_skill_pixel_layers_list(full, None).strip("list()").split()) == {"0"})

    print("[8] action_regional_density multi-layer")
    rng8 = random.Random(42)
    ml_design = generate_random_design(rows=9, cols=8, n_switches=0, max_inductors=0,
                                       n_states=1, cap_min=0.0, cap_max=0.0, rng=rng8,
                                       p0=0.4, p1=0.6, allowed_layers=LAYERS)
    rd_result, rd_info = None, None
    for _ in range(100):
        rd_result, rd_info = action_regional_density(ml_design, rng8,
                                                     allowed_layers=LAYERS)
        if rd_result is not None:
            break
    check("regional_density multi-layer succeeds", rd_result is not None)
    if rd_result is not None:
        target_layer = rd_info["changed_cells"][0]["layer"]
        check("all changed cells carry the selected layer",
              all(cell["layer"] == target_layer for cell in rd_info["changed_cells"]))
        check("pixel_layers updated for changed cells",
              all(
                  (rd_result.pixel_layers[cell["row"]][cell["col"]] == target_layer
                   if cell["to"] == 1
                   else rd_result.pixel_layers[cell["row"]][cell["col"]] == 0)
                  for cell in rd_info["changed_cells"]
              ))
        check("selected layer is from allowed_layers", target_layer in LAYERS)
        check("description includes layer suffix", f"layer {target_layer}" in rd_info["description"])
    # single-layer path unchanged
    sl_design = generate_random_design(rows=9, cols=8, n_switches=0, max_inductors=0,
                                       n_states=1, cap_min=0.0, cap_max=0.0, rng=rng8,
                                       p0=0.4, p1=0.6)
    sl_result, sl_info = None, None
    for _ in range(100):
        sl_result, sl_info = action_regional_density(sl_design, rng8)
        if sl_result is not None:
            break
    check("regional_density single-layer still works", sl_result is not None)
    if sl_result is not None:
        check("single-layer changed_cells have no layer key",
              "layer" not in sl_info["changed_cells"][0])

    print("[9] backward compatibility")
    rng2 = random.Random(7)
    legacy = generate_random_design(rows=8, cols=7, n_switches=2, max_inductors=0,
                                    n_states=1, cap_min=0.0, cap_max=0.0, rng=rng2,
                                    p0=0.4, p1=0.6)
    check("legacy pixel_layers empty", legacy.pixel_layers == [])
    check("legacy serialization omits pixel_layers",
          "pixel_layers" not in legacy.to_pixelated_config())
    lnd = None
    for _ in range(30):
        lnd, linfo = action_flip_01(legacy, n_flip=3, rng=rng2)
        if lnd is not None:
            break
    check("legacy flip stays binary (no layers)",
          lnd is not None and lnd.pixel_layers == []
          and "from_layer" not in linfo["flipped_cells"][0])

    print("\n" + "=" * 60)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 60)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
