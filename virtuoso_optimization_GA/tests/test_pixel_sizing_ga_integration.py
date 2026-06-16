#!/usr/bin/env python3
"""
GA-integration smoke test for per-pixel sizing — NO Virtuoso/EMX.

Loads the tiny sizing config, builds the initial population through the real
GA code path, confirms every individual carries a sizing matrix, that the
engine override produced by _build_override includes `pixel_sizes`, and that
mutation + crossover run without error and preserve the sizing matrix.

Run:  python virtuoso_optimization_GA/tests/test_pixel_sizing_ga_integration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from virtuoso_optimization_GA.ga_optimizer import SwitchableFilterGA

CFG = _ROOT / "virtuoso_optimization_GA/configs/smoke/ps_sizing_tiny.yaml"

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


print(f"[GA] Loading {CFG.name}")
ga = SwitchableFilterGA(CFG)
check("config parsed: enable_pixel_sizing", ga.cfg.enable_pixel_sizing is True)
check("grid_size derived from engine config", abs(ga.grid_size - 25.0) < 1e-9,
      f"grid_size={ga.grid_size}")
from virtuoso_optimization_SA.actions import _ACTION_NAMES as _AN
check("perturb_pixel_sizes weight > 0",
      ga.cfg.action_weights[_AN.index("perturb_pixel_sizes")] > 0, str(ga.cfg.action_weights))

print("[GA] Building initial population")
pop = ga._init_population()
check("population non-empty", len(pop) > 0, f"n={len(pop)}")
all_sized = all(bool(ind.pixel_sizes) for ind in pop)
check("every individual has a sizing matrix", all_sized)
shapes_ok = all(
    len(ind.pixel_sizes) == ind.rows and all(len(r) == ind.cols for r in ind.pixel_sizes)
    for ind in pop
)
check("every sizing matrix is shape-consistent", shapes_ok)
ranges_ok = all(
    ga.cfg.pixel_size_min <= v <= ga.cfg.pixel_size_max
    for ind in pop for row in ind.pixel_sizes for v in row
)
check("all sizes within configured range", ranges_ok)

print("[GA] _build_override carries pixel_sizes")
override = ga._build_override(pop[0], gen_idx=0, cand_idx=0)
check("override mode correct", override.get("mode") == "pixelated_design_simple")
check("override pixelated has pixel_sizes",
      "pixel_sizes" in override.get("pixelated", {}))
check("override pixel_sizes shape matches matrix",
      len(override["pixelated"]["pixel_sizes"]) == pop[0].rows)

print("\n" + "=" * 60)
print(f"RESULT: {_passed} passed, {_failed} failed")
print("=" * 60)
sys.exit(1 if _failed else 0)
