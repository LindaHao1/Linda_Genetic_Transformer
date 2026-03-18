"""
pixel_matrix.py — Pixel grid analysis helpers for pixelated_design_simple mode.

Pixel type values
-----------------
0 : ground layer only (M4 only, no top metal, no central port)
1 : top + ground layer (M9 + M4, no central port)
2 : top + ground + inductor central port (M9 + M4 + IND_<i> pin)
3 : top + ground + switch central port  (M9 + M4 + SW_<i> pin)

Port naming contract
--------------------
Ports are always generated in row-first (left-to-right, top-to-bottom) order:
  PIX_IN   — left feedline I/O port
  PIX_OUT  — right feedline I/O port
  IND_0, IND_1, ...  — inductor pixel central ports (row-first order)
  SW_0,  SW_1,  ...  — switch  pixel central ports  (row-first order)

This ordering is shared by both this module and the SKILL layout template;
changing one without the other will silently break the port-to-component mapping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Matrix analysis
# ---------------------------------------------------------------------------

def analyze_matrix(matrix: List[List[int]]) -> Dict[str, Any]:
    """
    Walk the pixel_state_matrix in row-first order and classify every cell.

    Returns a dict with:
        num_rows    (int)
        num_cols    (int)
        ind_pixels  list of (row, col, ind_index)  — type-2 cells
        sw_pixels   list of (row, col,  sw_index)  — type-3 cells
        num_ind     int
        num_sw      int
        all_pixels  list of (row, col, pixel_type)  — every cell
    """
    num_rows = len(matrix)
    if num_rows == 0:
        raise ValueError("pixel_state_matrix must not be empty.")
    num_cols = len(matrix[0])
    if num_cols == 0:
        raise ValueError("pixel_state_matrix rows must not be empty.")

    ind_pixels: List[Tuple[int, int, int]] = []
    sw_pixels:  List[Tuple[int, int, int]] = []
    all_pixels: List[Tuple[int, int, int]] = []

    ind_idx = 0
    sw_idx  = 0

    for row in range(num_rows):
        for col in range(num_cols):
            ptype = int(matrix[row][col])
            if ptype not in (0, 1, 2, 3):
                raise ValueError(
                    f"Invalid pixel type {ptype!r} at ({row}, {col}). "
                    "Valid types: 0 (ground), 1 (metal), 2 (inductor), 3 (switch)."
                )
            all_pixels.append((row, col, ptype))
            if ptype == 2:
                ind_pixels.append((row, col, ind_idx))
                ind_idx += 1
            elif ptype == 3:
                sw_pixels.append((row, col, sw_idx))
                sw_idx += 1

    return {
        "num_rows":   num_rows,
        "num_cols":   num_cols,
        "ind_pixels": ind_pixels,
        "sw_pixels":  sw_pixels,
        "num_ind":    ind_idx,
        "num_sw":     sw_idx,
        "all_pixels": all_pixels,
    }


# ---------------------------------------------------------------------------
# EMX port flag builder
# ---------------------------------------------------------------------------

def build_emx_port_flags(matrix_info: Dict[str, Any]) -> str:
    """
    Build the EMX CLI port flags string from analyzed matrix info.

    Port order is always:
        PIX_IN  PIX_OUT  PIX_IN_GND  PIX_OUT_GND  IND_0 ...  SW_0 ...

    The two GND ports are placed in the ground layer at the same x/y
    position as PIX_IN / PIX_OUT and are used as reference ground
    terminals in the Spectre testbench.

    Returns e.g.:
        "-p PIX_IN -p PIX_OUT -p PIX_IN_GND -p PIX_OUT_GND -p IND_0 -p SW_0"
    """
    flags = ["-p PIX_IN", "-p PIX_OUT", "-p PIX_IN_GND", "-p PIX_OUT_GND"]
    for _, _, idx in matrix_info["ind_pixels"]:
        flags.append(f"-p IND_{idx}")
    for _, _, idx in matrix_info["sw_pixels"]:
        flags.append(f"-p SW_{idx}")
    return " ".join(flags)


def total_ports(matrix_info: Dict[str, Any]) -> int:
    """Total EMX port count: PIX_IN + PIX_OUT + PIX_IN_GND + PIX_OUT_GND + num_ind + num_sw."""
    return 4 + matrix_info["num_ind"] + matrix_info["num_sw"]


# ---------------------------------------------------------------------------
# SKILL list builder
# ---------------------------------------------------------------------------

def build_skill_pixel_types_list(matrix: List[List[int]]) -> str:
    """
    Convert 2D matrix to a flat SKILL list literal (row-first order).

    Example: [[0,1],[3,2]] → "list(0 1 3 2)"
    """
    flat = [str(int(v)) for row in matrix for v in row]
    return "list(" + " ".join(flat) + ")"


def build_skill_port_names_list(matrix_info: Dict[str, Any]) -> str:
    """
    Build a SKILL list of central port names aligned with all_pixels,
    in row-first order.  Pixels of type 0 and 1 get an empty string.

    Example output: list("" "" "IND_0" "SW_0" "" "IND_1" ...)
    """
    names: List[str] = []
    for row, col, ptype in matrix_info["all_pixels"]:
        if ptype == 2:
            # find inductor index
            idx = next(i for r, c, i in matrix_info["ind_pixels"] if r == row and c == col)
            names.append(f'"IND_{idx}"')
        elif ptype == 3:
            idx = next(i for r, c, i in matrix_info["sw_pixels"] if r == row and c == col)
            names.append(f'"SW_{idx}"')
        else:
            names.append('""')
    return "list(" + " ".join(names) + ")"
