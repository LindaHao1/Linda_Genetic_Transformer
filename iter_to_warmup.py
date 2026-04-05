#!/usr/bin/env python3
"""
Convert iter_sims into warmup_sims format.

iter_sims structure:   iterN/candNNNN/worker_X/sim_name/...
warmup_sims structure: warmup_NNNN/worker_X/sim_name/...

Each (iterN, candNNNN) pair becomes a new warmup_NNNN folder,
continuing from the highest existing warmup index.
"""

import argparse
import os
import re
import shutil
import sys


def get_next_warmup_index(warmup_dir: str) -> int:
    """Return the next available warmup index (max existing + 1, or 0)."""
    if not os.path.isdir(warmup_dir):
        return 0
    indices = []
    for name in os.listdir(warmup_dir):
        m = re.fullmatch(r"warmup_(\d+)", name)
        if m:
            indices.append(int(m.group(1)))
    return max(indices) + 1 if indices else 0


def collect_iter_candidates(iter_dir: str) -> list[tuple[str, str]]:
    """
    Return sorted list of (iter_name, cand_name) pairs found in iter_dir.
    Sorted by iter number first, then cand number.
    """
    candidates = []
    for iter_name in os.listdir(iter_dir):
        if not re.fullmatch(r"iter\d+", iter_name):
            continue
        iter_path = os.path.join(iter_dir, iter_name)
        if not os.path.isdir(iter_path):
            continue
        for cand_name in os.listdir(iter_path):
            if not re.fullmatch(r"cand\d+", cand_name):
                continue
            cand_path = os.path.join(iter_path, cand_name)
            if os.path.isdir(cand_path):
                candidates.append((iter_name, cand_name))

    candidates.sort(key=lambda x: (int(x[0][4:]), int(x[1][4:])))
    return candidates


def convert(iter_dir: str, warmup_dir: str, dry_run: bool = False, move: bool = False) -> None:
    """Copy (or move) each iter candidate into warmup format."""
    os.makedirs(warmup_dir, exist_ok=True)

    candidates = collect_iter_candidates(iter_dir)
    if not candidates:
        print("No iter candidates found.")
        return

    next_idx = get_next_warmup_index(warmup_dir)
    print(f"Found {len(candidates)} iter candidates.")
    print(f"Starting warmup index: {next_idx}")
    if dry_run:
        print("[DRY RUN] No files will be copied/moved.\n")

    op = "move" if move else "copy"

    for i, (iter_name, cand_name) in enumerate(candidates):
        src = os.path.join(iter_dir, iter_name, cand_name)
        dst_name = f"warmup_{next_idx + i:04d}"
        dst = os.path.join(warmup_dir, dst_name)

        print(f"  {iter_name}/{cand_name}  ->  {dst_name}")

        if dry_run:
            continue

        if os.path.exists(dst):
            print(f"    WARNING: {dst} already exists, skipping.")
            continue

        if move:
            shutil.move(src, dst)
        else:
            shutil.copytree(src, dst)

    print(f"\nDone. {len(candidates)} candidates {'moved' if move else 'copied'} as warmup_{next_idx:04d} .. warmup_{next_idx + len(candidates) - 1:04d}.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert iter_sims candidates into warmup_sims format."
    )
    parser.add_argument("iter_dir", help="Path to iter_sims directory")
    parser.add_argument("warmup_dir", help="Path to warmup_sims directory")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without copying anything"
    )
    parser.add_argument(
        "--move", action="store_true",
        help="Move instead of copy (default: copy)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.iter_dir):
        print(f"ERROR: iter_dir not found: {args.iter_dir}", file=sys.stderr)
        sys.exit(1)

    convert(args.iter_dir, args.warmup_dir, dry_run=args.dry_run, move=args.move)


if __name__ == "__main__":
    main()
