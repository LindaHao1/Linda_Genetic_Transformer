"""
Inspect one tokenized design to sanity-check the tokenizer.

  python scripts/inspect_tokens.py --config configs/default.yaml          # synthetic
  python scripts/inspect_tokens.py --config configs/default.yaml --source sims

Prints: the raw token stream grouped by segment, the grid rendered as a
rows x cols character map, the decoded values/cap-pairs/states, and a
decode round-trip check (structure exact, values within one bin).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import make_random_design  # noqa: E402
from src.factory import freq_points, load_config  # noqa: E402
from src.tokenizer import Tokenizer, TokenizerConfig  # noqa: E402

_PIX = {0: "G", 1: "M", 2: "I", 3: "S", 4: "V"}   # ground/metal/ind/switch/via


def _tok_from_cfg(cfg) -> Tokenizer:
    g = cfg["grid"]
    return Tokenizer(TokenizerConfig(
        rows=g["rows"], cols=g["cols"], n_states=g["n_states"],
        n_bins=cfg["tokenizer"]["n_bins"],
        n_inductor_values=g.get("n_inductor_values", 1),
        cap_min=float(g["cap_min"]), cap_max=float(g["cap_max"]),
        inductor_cap_min=float(g["inductor_cap_min"]),
        inductor_cap_max=float(g["inductor_cap_max"]),
        cap_pair_min=float(g["cap_pair_min"]),
        cap_pair_max=float(g["cap_pair_max"]),
        cap_pair_min_dist=g["cap_pair_min_dist"],
        cap_pair_max_dist=g["cap_pair_max_dist"],
        max_cap_pairs=g["max_cap_pairs"]))


def _print_stream(tok: Tokenizer, ids):
    """Token stream, newline at each SEP/structural boundary."""
    print("\n=== TOKEN STREAM (idx:NAME) ===")
    line = []
    for i, t in enumerate(ids):
        n = tok.id2tok[t]
        line.append(f"{i}:{n}")
        if n.startswith("SEP_") or n in ("BOS", "CAPPAIR_STOP", "EOS"):
            print("  " + "  ".join(line))
            line = []
    if line:
        print("  " + "  ".join(line))


def _print_grid(tok: Tokenizer, rec):
    print("\n=== GRID (G=gnd M=metal I=ind S=switch V=via | "
          "ports decode to metal) ===")
    for row in rec["matrix"]:
        print("  " + " ".join(_PIX.get(v, "?") for v in row))


def _print_components(tok: Tokenizer, rec):
    cfg = tok.config
    print("\n=== VALUES (raster order) ===")
    sw = ind = 0
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            v = rec["matrix"][r][c]
            if v == 3:
                fF = rec["cap_values"][sw] * 1e15
                print(f"  switch[{sw}] @({r},{c})  cap={fF:7.1f} fF")
                sw += 1
            elif v == 2:
                mfF = rec["inductor_cap_values"][ind] * 1e15
                print(f"  ind[{ind}]    @({r},{c})  model="
                      f"{rec['inductor_values'][ind]}  MIMcap={mfF:7.1f} fF")
                ind += 1
    print("\n=== CAP PAIRS ===")
    for k, ((r1, c1), (r2, c2)) in enumerate(rec["cap_pixel_pairs"]):
        orient = "H" if r1 == r2 else "V"
        L = max(abs(r2 - r1), abs(c2 - c1))
        print(f"  pair[{k}] head=({r1},{c1}) -> ({r2},{c2})  {orient} len={L}"
              f"  val={rec['cap_pair_values'][k]*1e15:7.1f} fF")
    if not rec["cap_pixel_pairs"]:
        print("  (none)")
    print("\n=== STATE CODES (per state: switches then inductors) ===")
    for s, (sw_row, ind_row) in enumerate(zip(rec["switch_states"],
                                              rec["inductor_switch_states"])):
        print(f"  state {s}: sw={sw_row}  ind={ind_row}")


def _roundtrip(tok: Tokenizer, rec):
    ids = tok.encode(rec)
    dec = tok.decode(ids)
    ok = (dec["matrix"] == rec["matrix"]
          and dec["switch_states"] == rec["switch_states"]
          and dec["inductor_switch_states"] == rec["inductor_switch_states"]
          and dec["cap_pixel_pairs"] == rec["cap_pixel_pairs"])
    stable = tok.encode(dec) == ids        # quantization fixed-point
    print("\n=== ROUND-TRIP ===")
    print(f"  structure exact : {'PASS' if ok else 'FAIL'}")
    print(f"  re-encode stable: {'PASS' if stable else 'FAIL'}")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", choices=["synthetic", "sims"],
                    default="synthetic")
    ap.add_argument("--index", type=int, default=0,
                    help="which design (sims source)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = _tok_from_cfg(cfg)
    print(f"vocab_size={tok.vocab_size}  grid={tok.config.rows}x"
          f"{tok.config.cols}  K={tok.config.n_states}  "
          f"n_bins={tok.config.n_bins}")

    if args.source == "sims":
        from src.dataset import build_from_sims
        ds = build_from_sims(cfg["data"]["sims_dirs"], freq_points(cfg),
                             limit=args.index + 1,
                             n_states=cfg["grid"]["n_states"],
                             seed=cfg["data"].get("state_select_seed", 0))
        design = ds[args.index].design
    else:
        design = make_random_design(tok.config, np.random.default_rng(0))

    rec = design.to_record()
    ids = _roundtrip(tok, rec)
    print(f"\n  sequence length: {len(ids)} tokens")
    # per-family token counts
    from collections import Counter
    cnt = Counter(tok.get_token_type(t) for t in ids)
    print("  family counts:", dict(cnt))

    _print_stream(tok, ids)
    _print_grid(tok, rec)
    _print_components(tok, rec)


if __name__ == "__main__":
    main()
