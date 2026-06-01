"""
Convert an LM-generated `Design` into a config override dict / YAML file that
the Virtuoso pixel engine consumes (mode `pixelated_design_simple`).

`build_pixelated_override(design)` returns the inline override the engine's
`run_batch` takes directly (preferred for in-process simulation).

`write_simulation_yaml(design, base_config_path, out_yaml_path,
output_parent_dir)` writes a standalone YAML that inherits from the base
virtuoso config (e.g. `virtuoso_engine/config_ps31_gs25.yaml`) and embeds
the override.  Either path works with the engine; the YAML is the
artifact-on-disk version.

`resolve_engine_config_from_data_collection(data_cfg_path)` reads a
"data collection" GA config (e.g. `world_model_test/wm_data_collection.yaml`)
and returns the engine config path it pulls in via `virtuoso_config:`.  This
keeps the LM-generated design YAML inheriting from the SAME engine config
that produced the training data -- no parameter drift.

Field shapes mirror `virtuoso_optimization_SA/design.py:DesignState.to_pixelated_config`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.types import Design

_VE_DIR = "/disk/mcao53/RFIC_RL"
if _VE_DIR not in sys.path:
    sys.path.insert(0, _VE_DIR)

from virtuoso_engine.config_loader import load_config  # noqa: E402


def resolve_engine_config_from_data_collection(
    data_cfg_path: str,
) -> str:
    """Read a GA-style data-collection YAML and return the engine config path
    it points at via `virtuoso_config:`.

    Example: `world_model_test/wm_data_collection.yaml` -> through its
    `_base_:` chain -> `virtuoso_config: virtuoso_engine/config_ps31_gs25.yaml`.

    Paths in `virtuoso_config:` may be relative to the config file they were
    declared in; this helper resolves them against the data-collection file's
    parent.  Falls back to the absolute path when it's already absolute.
    """
    cfg = load_config(data_cfg_path)
    vc = cfg.get("virtuoso_config")
    if not vc:
        raise SystemExit(
            f"{data_cfg_path}: no `virtuoso_config:` field found in the "
            f"resolved config -- not a data-collection / GA config?")
    vc_path = Path(vc)
    if not vc_path.is_absolute():
        # Resolve relative to the data-collection file's parent (matches how
        # GA configs are written).
        vc_path = (Path(data_cfg_path).resolve().parent / vc_path).resolve()
    return str(vc_path)


def build_pixelated_override(
    design: Design,
    output_parent_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Inline override dict, ready for `ParallelVirtuosoEngine.run_batch`."""
    # bank-size-1 wrapping for engine: switch_states[s][i] -> [bit], cap[i] -> [val]
    state_list = [[[int(b)] for b in row] for row in design.switch_states]
    cap_values = [[float(v)] for v in design.switch_caps]

    pixelated: Dict[str, Any] = {
        "pixel_state_matrix": [list(row) for row in design.pixel_grid],
        "state_list":         state_list,
        "cap_values":         cap_values,
        "inductor_values":    list(design.inductor_values),
    }
    if design.inductor_switch_states:
        pixelated["inductor_switch_states"] = [
            list(s) for s in design.inductor_switch_states
        ]
    if design.inductor_cap_values:
        pixelated["inductor_cap_values"] = list(design.inductor_cap_values)
    if design.cap_pixel_pairs:
        pixelated["cap_pixel_pairs"] = [
            [list(p[0]), list(p[1])] for p in design.cap_pixel_pairs
        ]
        pixelated["cap_pair_values"] = list(design.cap_pair_values)

    override: Dict[str, Any] = {
        "mode":      "pixelated_design_simple",
        "pixelated": pixelated,
    }
    if output_parent_dir is not None:
        override["project"] = {"parent_output_dir": str(output_parent_dir)}
    return override


def write_simulation_yaml(
    design: Design,
    base_config_path: str,
    out_yaml_path: str,
    output_parent_dir: Optional[str] = None,
) -> str:
    """Write a standalone YAML that the engine can consume.

    The YAML uses the existing `_base_:` inheritance mechanism: it points at
    `base_config_path` (e.g. `virtuoso_engine/config_ps31_gs25.yaml`) and
    overlays the `pixelated` block for the LM-generated design.
    """
    override = build_pixelated_override(design, output_parent_dir)
    out: Dict[str, Any] = {"_base_": base_config_path, **override}
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return out_yaml_path
