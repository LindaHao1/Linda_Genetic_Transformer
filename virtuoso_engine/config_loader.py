"""
config_loader.py — YAML config loader with optional _base_ inheritance.

Usage
-----
Replace ``yaml.safe_load(open(path))`` with ``load_config(path)`` wherever
you want configs to support inheritance.

_base_ key
----------
Add ``_base_: relative/path.yaml`` to any config file to inherit from a base.
The base is loaded first, then the current file's values are deep-merged on
top (child wins).  The ``_base_`` key is stripped from the result.

Merge rules
-----------
- **Dicts** are recursively merged: only the keys that differ need to be
  specified in the child, allowing e.g. a single ``action_weights`` entry to
  be overridden without rewriting the whole dict.
- **Lists** are replaced wholesale: ``filter_specs``, ``inductor_translation_table``,
  etc. must be given in full if they need to change.
- Chains are supported: A → B → C (B inherits from C, A inherits from B).
- All ``_base_`` paths are resolved relative to the file that contains them.

Backward compatibility
----------------------
Files without a ``_base_`` key behave identically to plain ``yaml.safe_load``.
"""

from __future__ import annotations

import os
from pathlib import Path
import yaml


# Directory holding per-server overlay files (servers/<name>.yaml), relative to
# this module.  The active server is selected by the RFIC_SERVER env var, else
# the merged config's server.active, else "local".
_SERVERS_DIR = Path(__file__).resolve().parent / "servers"


def load_config(path: str | Path, _apply_server: bool = True) -> dict:
    """Load a YAML config file, resolving ``_base_`` inheritance and the active
    server overlay.

    Parameters
    ----------
    path:
        Absolute or relative path to the YAML config file.
    _apply_server:
        Internal flag.  When True (default) the active server overlay
        (``servers/<name>.yaml``) is deep-merged on top of the resolved config.
        Set False when loading a server overlay itself to avoid recursion.

    Returns
    -------
    dict
        The fully-merged configuration dictionary.
    """
    path = Path(path).resolve()
    raw: dict = yaml.safe_load(path.read_text()) or {}

    base_rel = raw.pop("_base_", None)
    if base_rel is None:
        merged = raw
    else:
        base_path = path.parent / base_rel
        # Defer the server overlay until the full _base_ chain is resolved.
        base = load_config(base_path, _apply_server=False)
        merged = _deep_merge(base, raw)

    if _apply_server:
        merged = _apply_server_overlay(merged)
    return merged


def _apply_server_overlay(cfg: dict) -> dict:
    """Deep-merge the active server overlay on top of ``cfg``.

    Active server = ``$RFIC_SERVER`` if set, else ``cfg["server"]["active"]``,
    else ``"local"``.  Server values win.  Missing overlay files are a no-op.
    """
    active = os.environ.get("RFIC_SERVER")
    if not active:
        active = (cfg.get("server") or {}).get("active", "local")

    overlay_path = _SERVERS_DIR / f"{active}.yaml"
    if not overlay_path.exists():
        return cfg

    overlay = load_config(overlay_path, _apply_server=False)
    return _deep_merge(cfg, overlay)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Dict values are merged recursively; all other types (including lists)
    are replaced by the override value.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
