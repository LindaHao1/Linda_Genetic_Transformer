#!/usr/bin/env python3
"""
pixel_gui_designer/cli_designer.py
Simple command-line interface for editing pixelated RF filter designs (config.yaml).

Pixel types:
  0  GND  — ground
  1  MTL  — metal
  2  IND  — inductor port
  3  SW   — switch port
  4  VIA  — via
  5  SCAP — shunt-capacitor port (permanent MIM cap to ground)

Usage:
  python pixel_gui_designer/cli_designer.py
  python pixel_gui_designer/cli_designer.py path/to/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reuse the DesignModel from the GUI designer
sys.path.insert(0, str(Path(__file__).parent.parent))
from pixel_gui_designer.designer import DesignModel

PIXEL_NAMES = {0: "GND", 1: "MTL", 2: "IND", 3: "SW", 4: "VIA", 5: "SCAP"}
PIXEL_CHARS = {0: ".", 1: "#", 2: "I", 3: "S", 4: "V", 5: "C"}

HELP_TEXT = """
Commands:
  print / p          Print the current matrix
  info               Show pixel type legend and grid dimensions
  set R C T          Set pixel at row R, col C to type T (0-5)
  fill T             Fill entire grid with type T
  addrow [IDX]       Add a blank row before IDX (default: append at bottom)
  delrow IDX         Delete row IDX
  addcol [IDX]       Add a blank column before IDX (default: append at right)
  delcol IDX         Delete column IDX
  save [PATH]        Save to PATH (default: auto-generated filename)
  load PATH          Load a different config file
  help / h / ?       Show this help
  quit / q / exit    Exit without saving
"""


def print_matrix(model: DesignModel) -> None:
    nr, nc = model.num_rows(), model.num_cols()
    col_header = "     " + "  ".join(f"{c:2d}" for c in range(nc))
    print(col_header)
    print("     " + "---" * nc)
    for r in range(nr):
        row_str = "  ".join(PIXEL_CHARS[model.get_pixel(r, c)] for c in range(nc))
        print(f" {r:2d} | {row_str}")
    print()
    print("Legend:", "  ".join(f"{ch}={name}" for (t, ch), name in
          zip([(k, PIXEL_CHARS[k]) for k in sorted(PIXEL_CHARS)], PIXEL_NAMES.values())))
    print(f"Grid: {nr} rows × {nc} cols")


def auto_save_path(base_path: str | None) -> str:
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_path:
        p = Path(base_path)
        return str(p.parent / f"{p.stem}_cli_{ts}{p.suffix}")
    return f"cli_design_{ts}.yaml"


def run(initial_path: str | None = None) -> None:
    model = DesignModel()
    current_path: str | None = initial_path

    if current_path:
        try:
            model.load(current_path)
            print(f"Loaded: {current_path}")
        except Exception as e:
            print(f"Failed to load '{current_path}': {e}")
            print("Starting with a default 3×3 grid.")
            model = DesignModel.default()
            current_path = None
    else:
        model = DesignModel.default()
        print("No file specified — starting with a default 3×3 grid.")

    print_matrix(model)
    print(HELP_TEXT)

    while True:
        try:
            raw = input("cli-designer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        # ── print ──────────────────────────────────────────────────────
        if cmd in ("print", "p"):
            print_matrix(model)

        # ── info ───────────────────────────────────────────────────────
        elif cmd == "info":
            print(f"Grid: {model.num_rows()} rows × {model.num_cols()} cols")
            print("Pixel types:")
            for t, name in PIXEL_NAMES.items():
                print(f"  {t} = {name} ({PIXEL_CHARS[t]})")
            sw = model.sw_pixels()
            ind = model.ind_pixels()
            print(f"Switch pixels : {[(r, c) for r, c, _ in sw]}")
            print(f"Inductor pixels: {[(r, c) for r, c, _ in ind]}")

        # ── set R C T ──────────────────────────────────────────────────
        elif cmd == "set":
            if len(parts) != 4:
                print("Usage: set R C T")
                continue
            try:
                r, c, t = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                print("R, C, T must be integers.")
                continue
            if not (0 <= r < model.num_rows() and 0 <= c < model.num_cols()):
                print(f"Row must be 0-{model.num_rows()-1}, col 0-{model.num_cols()-1}.")
                continue
            if t not in PIXEL_NAMES:
                print(f"Type must be one of {list(PIXEL_NAMES.keys())}.")
                continue
            model.set_pixel(r, c, t)
            print(f"Set ({r},{c}) → {PIXEL_NAMES[t]}")
            print_matrix(model)

        # ── fill T ─────────────────────────────────────────────────────
        elif cmd == "fill":
            if len(parts) != 2:
                print("Usage: fill T")
                continue
            try:
                t = int(parts[1])
            except ValueError:
                print("T must be an integer.")
                continue
            if t not in PIXEL_NAMES:
                print(f"Type must be one of {list(PIXEL_NAMES.keys())}.")
                continue
            for r in range(model.num_rows()):
                for c in range(model.num_cols()):
                    model.set_pixel(r, c, t)
            print(f"Filled entire grid with {PIXEL_NAMES[t]}.")
            print_matrix(model)

        # ── addrow [IDX] ───────────────────────────────────────────────
        elif cmd == "addrow":
            idx = model.num_rows() if len(parts) < 2 else int(parts[1])
            model.add_row(idx)
            print(f"Added row at index {idx}.")
            print_matrix(model)

        # ── delrow IDX ─────────────────────────────────────────────────
        elif cmd == "delrow":
            if len(parts) != 2:
                print("Usage: delrow IDX")
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                print("IDX must be an integer.")
                continue
            model.remove_row(idx)
            print(f"Deleted row {idx}.")
            print_matrix(model)

        # ── addcol [IDX] ───────────────────────────────────────────────
        elif cmd == "addcol":
            idx = model.num_cols() if len(parts) < 2 else int(parts[1])
            model.add_col(idx)
            print(f"Added column at index {idx}.")
            print_matrix(model)

        # ── delcol IDX ─────────────────────────────────────────────────
        elif cmd == "delcol":
            if len(parts) != 2:
                print("Usage: delcol IDX")
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                print("IDX must be an integer.")
                continue
            model.remove_col(idx)
            print(f"Deleted column {idx}.")
            print_matrix(model)

        # ── save [PATH] ────────────────────────────────────────────────
        elif cmd == "save":
            if len(parts) >= 2:
                save_path = " ".join(parts[1:])
            else:
                save_path = auto_save_path(current_path)
            try:
                model.save(save_path)
                print(f"Saved to: {save_path}")
                current_path = save_path
            except Exception as e:
                print(f"Save failed: {e}")

        # ── load PATH ──────────────────────────────────────────────────
        elif cmd == "load":
            if len(parts) < 2:
                print("Usage: load PATH")
                continue
            load_path = " ".join(parts[1:])
            try:
                model.load(load_path)
                current_path = load_path
                print(f"Loaded: {load_path}")
                print_matrix(model)
            except Exception as e:
                print(f"Load failed: {e}")

        # ── help ───────────────────────────────────────────────────────
        elif cmd in ("help", "h", "?"):
            print(HELP_TEXT)

        # ── quit ───────────────────────────────────────────────────────
        elif cmd in ("quit", "q", "exit"):
            print("Exiting.")
            break

        else:
            print(f"Unknown command '{cmd}'. Type 'help' for usage.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run(path)
