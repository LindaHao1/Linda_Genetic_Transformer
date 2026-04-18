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

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file, resolving optional ``_base_`` inheritance.

    Parameters
    ----------
    path:
        Absolute or relative path to the YAML config file.

    Returns
    -------
    dict
        The fully-merged configuration dictionary.
    """
    path = Path(path).resolve()
    raw: dict = yaml.safe_load(path.read_text()) or {}

    base_rel = raw.pop("_base_", None)
    if base_rel is None:
        return raw

    base_path = path.parent / base_rel
    base = load_config(base_path)
    return _deep_merge(base, raw)


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
