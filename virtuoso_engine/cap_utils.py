"""
cap_utils.py — Shared capacitor dimension lookup utility.

Used by both engine.py (full_layout mode) and spectre_runner.py
(pixelated_design_simple TSMC model instantiation).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List


def _lookup_cap_dims(
    table_path: Path,
    cap_values: List[float],
) -> List[tuple]:
    """Return (length_um, width_um) for each capacitance value using a linear fit.

    Algorithm per value:
      1. Find the 3 table rows closest in cap value.
      2. Compute area = length * width for each; fit area = a * cap + b (least squares).
      3. Predict area at target cap value.
      4. Assume square: l = w = sqrt(area).
      5. If w >= max_width in the table, clamp w = max_width, l = area / max_width.
      6. Round both to 2 decimal places.

    CSV format: cap_value_f, length_um, width_um
    """
    if not table_path.exists():
        raise FileNotFoundError(
            f"Cap dimension table not found: {table_path}\n"
            "Check the switch_cap_table / cap_pair_table config keys."
        )
    rows: List[tuple] = []
    with table_path.open() as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for r in reader:
            rows.append((float(r["cap_value_f"]), float(r["length_um"]), float(r["width_um"])))
    if not rows:
        raise ValueError(f"Cap dimension table is empty: {table_path}")

    max_w = max(r[2] for r in rows)

    result = []
    for target_cv in cap_values:
        nearest = sorted(rows, key=lambda r: abs(r[0] - target_cv))[:3]
        caps   = [r[0] for r in nearest]
        areas  = [r[1] * r[2] for r in nearest]

        n   = len(caps)
        sx  = sum(caps)
        sy  = sum(areas)
        sxx = sum(c * c for c in caps)
        sxy = sum(c * a for c, a in zip(caps, areas))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-30:
            a_coef, b_coef = 0.0, sy / n
        else:
            a_coef = (n * sxy - sx * sy) / denom
            b_coef = (sy - a_coef * sx) / n

        pred_area = a_coef * target_cv + b_coef
        pred_area = max(pred_area, 0.0)

        side = pred_area ** 0.5
        if side >= max_w:
            w = max_w
            l = pred_area / max_w if max_w > 0 else side
        else:
            l = w = side

        result.append((round(l, 2), round(w, 2)))
    return result
