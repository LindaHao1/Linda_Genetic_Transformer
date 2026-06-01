"""
Constrained decoding mask.

`get_invalid_token_mask(partial_sequence, tokenizer, config) -> BoolTensor`
returns a [vocab_size] mask, True where the token is INVALID as the next token.
`generate()` applies it via `logits[mask] = -inf` at every step.

Grammar (single source of truth — matches `tokenizer.py`):

    BOS -> SEP_spec
    grid cells, raster row-major; per cell exactly one pixel-family token:
        GROUND / METAL / VIA / PORT                (fresh cell)
        SWITCH        VAL_*                        (fresh cell, inline VAL)
        INDUCTOR      INDVAL_* VAL_*               (fresh cell, inline INDVAL+VAL)
        CAP_PAIR_{H|V}{L}  VAL_*                   (fresh cell; head only)
    interior/dest cells of an already-started pair are forced to GROUND /
    METAL by the per-cell forced-code map below.
    SEP_grid -> K*(n_sw+n_ind) STATE_* -> SEP_states -> EOS

Cap-pair spatial validity (user's rule):
    head cell: METAL/PORT (we emit a CAP_PAIR_* token there);
    interior cells: all GROUND; dest cell: METAL/PORT.
    Path + budget enforced by the forced-code bookkeeping during the walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch

from src.tokenizer import (
    P_GROUND, P_INDUCTOR, P_METAL, P_SWITCH, P_VIA, Tokenizer,
)


@dataclass
class ConstraintConfig:
    cap_pair_interior_all_ground: bool = True


# ---------------------------------------------------------------------------
# Grid-walk simulation (recovers per-cell forced map + pending state)
# ---------------------------------------------------------------------------

def _is_pixel_name(n: str, cap_pair_names: Set[str]) -> bool:
    return n in ("GROUND", "METAL", "INDUCTOR", "SWITCH", "VIA", "PORT") \
        or n in cap_pair_names


def _name_to_pixel_code(n: str) -> int:
    if n == "GROUND":
        return P_GROUND
    if n in ("METAL", "PORT"):
        return P_METAL
    if n == "INDUCTOR":
        return P_INDUCTOR
    if n == "SWITCH":
        return P_SWITCH
    if n == "VIA":
        return P_VIA
    if n.startswith("CAP_PAIR_"):
        return P_METAL
    raise ValueError(f"not a pixel name: {n!r}")


def _parse_cap_pair(name: str) -> Tuple[str, int]:
    """'CAP_PAIR_H2' -> ('H', 2)."""
    body = name[len("CAP_PAIR_"):]
    return body[0], int(body[1:])


def _pair_path(hr: int, hc: int, orient: str, length: int
               ) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    if orient == "H":
        dest = (hr, hc + length)
        interior = [(hr, hc + k) for k in range(1, length)]
    else:
        dest = (hr + length, hc)
        interior = [(hr + k, hc) for k in range(1, length)]
    return dest, interior


def _simulate_grid_walk(grid_tokens: List[str], rows: int, cols: int,
                        cap_pair_names: Set[str]) -> Dict[str, object]:
    """Replay the grid stream so far and return current parser state.

    Fields:
        cursor: next raster cell index to be emitted.
        pending: None | "VAL" | "INDVAL" | "VAL_AFTER_INDVAL" | "VAL_PAIR"
        forced: { (r, c) -> "GROUND" | "METAL" }
        emitted_cells: cells whose pixel token has been consumed (cell index)
        n_pairs: cap pairs emitted so far
        n_ports_left / n_ports_right: how many PORT tokens have been emitted
            on the left/right feed columns (col 0 / col cols-1).  The
            constraint mask enforces exactly one PORT per feed column.
        matrix: per-cell code so far (0..4); cells past the cursor are GROUND
            placeholders (overwritten when their token arrives).
    """
    cursor = 0
    pending: str | None = None
    forced: Dict[Tuple[int, int], str] = {}
    n_pairs = 0
    n_ports_left = 0
    n_ports_right = 0
    matrix = [[P_GROUND] * cols for _ in range(rows)]
    consumed_cells: List[int] = []

    for n in grid_tokens:
        if pending == "VAL" or pending == "VAL_AFTER_INDVAL" or pending == "VAL_PAIR":
            assert n.startswith("VAL_"), n
            pending = None
            continue
        if pending == "INDVAL":
            assert n.startswith("INDVAL_"), n
            pending = "VAL_AFTER_INDVAL"
            continue
        # pending is None: this must be a pixel-family token at the cursor cell
        assert _is_pixel_name(n, cap_pair_names), \
            f"expected pixel at cursor, got {n!r}"
        r, c = divmod(cursor, cols)
        matrix[r][c] = _name_to_pixel_code(n)
        consumed_cells.append(cursor)
        if n == "PORT":
            if c == 0:
                n_ports_left += 1
            elif c == cols - 1:
                n_ports_right += 1
        if n == "SWITCH":
            pending = "VAL"
        elif n == "INDUCTOR":
            pending = "INDVAL"
        elif n in cap_pair_names:
            orient, length = _parse_cap_pair(n)
            dest, interior = _pair_path(r, c, orient, length)
            for (ir, ic) in interior:
                forced[(ir, ic)] = "GROUND"
            forced[(dest[0], dest[1])] = "METAL"
            n_pairs += 1
            pending = "VAL_PAIR"
        else:
            pending = None
        cursor += 1

    return {
        "cursor": cursor,
        "pending": pending,
        "forced": forced,
        "n_pairs": n_pairs,
        "n_ports_left": n_ports_left,
        "n_ports_right": n_ports_right,
        "matrix": matrix,
        "consumed_cells": consumed_cells,
    }


# ---------------------------------------------------------------------------
# Allowed-next set
# ---------------------------------------------------------------------------

def _allowed_next(names: List[str], tok: Tokenizer,
                  ccfg: ConstraintConfig) -> Set[str]:
    cfg = tok.config
    rows, cols = cfg.rows, cfg.cols
    n_grid = rows * cols
    cap_pair_names = set(tok._cap_pair_names)

    val_names = {f"VAL_{i}" for i in range(cfg.n_bins)}
    indval_names = {f"INDVAL_{i}" for i in range(max(cfg.n_inductor_values, 1))}
    pixel_basic = {"GROUND", "METAL", "INDUCTOR", "SWITCH", "VIA"}

    # 0) sequence start
    if len(names) == 0:
        return {"BOS"}
    if len(names) == 1:
        return {"SEP_spec"}

    # body = tokens after BOS, SEP_spec
    body = names[2:]

    # find SEP_grid (if reached)
    try:
        sep_grid_at = body.index("SEP_grid")
    except ValueError:
        sep_grid_at = None

    # ------------------------------------------------------------------
    # Phase: grid
    # ------------------------------------------------------------------
    if sep_grid_at is None:
        grid_tokens = body
        st = _simulate_grid_walk(grid_tokens, rows, cols, cap_pair_names)
        cursor: int = st["cursor"]        # type: ignore
        pending: str | None = st["pending"]  # type: ignore
        forced: Dict[Tuple[int, int], str] = st["forced"]  # type: ignore
        n_pairs: int = st["n_pairs"]      # type: ignore
        n_ports_left: int = st["n_ports_left"]    # type: ignore
        n_ports_right: int = st["n_ports_right"]  # type: ignore

        if pending == "VAL" or pending == "VAL_PAIR" \
                or pending == "VAL_AFTER_INDVAL":
            return val_names
        if pending == "INDVAL":
            return indval_names

        # pending is None -> we're at the cursor cell (or at SEP_grid).
        if cursor >= n_grid:
            return {"SEP_grid"}

        r, c = divmod(cursor, cols)
        is_feed_col = (c == 0 or c == cols - 1)
        # `port_needed` = the feed column still needs its single PORT and this
        # is the last chance (last row of that column).
        if c == 0:
            port_needed = (n_ports_left == 0 and r == rows - 1)
            port_already = (n_ports_left > 0)
        elif c == cols - 1:
            port_needed = (n_ports_right == 0 and r == rows - 1)
            port_already = (n_ports_right > 0)
        else:
            port_needed = False
            port_already = False

        # If this cell is forced by a prior cap-pair, that code wins -- but a
        # cap-pair can never land in a feed column (the head/dest constraints
        # below filter that), so the forced branch never collides with feed
        # rules in practice.
        if (r, c) in forced:
            code = forced[(r, c)]
            if code == "GROUND":
                return {"GROUND"}
            allowed: Set[str] = {"METAL"}
            return allowed

        # Feed column: only GROUND or PORT, with the single-PORT rule.
        if is_feed_col:
            if port_already:
                return {"GROUND"}
            if port_needed:
                return {"PORT"}
            return {"GROUND", "PORT"}

        # Interior cell: regular pixel set.
        allowed_free: Set[str] = set(pixel_basic)

        # cap pairs only legal in the interior, and must avoid feed columns
        if n_pairs < cfg.max_cap_pairs:
            consumed_cells = set(st["consumed_cells"])  # type: ignore
            for orient in ("H", "V"):
                for length in cfg.lengths:
                    dest, interior = _pair_path(r, c, orient, length)
                    if not (0 <= dest[0] < rows and 0 <= dest[1] < cols):
                        continue
                    # feed columns are off-limits to pair endpoints + interior
                    bad_feed = False
                    if dest[1] == 0 or dest[1] == cols - 1:
                        bad_feed = True
                    for (ir, ic) in interior:
                        if ic == 0 or ic == cols - 1:
                            bad_feed = True; break
                    if bad_feed:
                        continue
                    # No cell in path may have already been emitted (raster
                    # cursor is past it) or conflict-forced.
                    conflict = False
                    for (ir, ic) in interior:
                        cell_idx = ir * cols + ic
                        if cell_idx in consumed_cells:
                            conflict = True; break
                        if (ir, ic) in forced and forced[(ir, ic)] != "GROUND":
                            conflict = True; break
                    if conflict:
                        continue
                    dest_idx = dest[0] * cols + dest[1]
                    if dest_idx in consumed_cells:
                        continue
                    if (dest[0], dest[1]) in forced \
                            and forced[(dest[0], dest[1])] != "METAL":
                        continue
                    allowed_free.add(f"CAP_PAIR_{orient}{length}")
        return allowed_free

    # ------------------------------------------------------------------
    # Phase: after SEP_grid (states + EOS)
    # ------------------------------------------------------------------
    grid_tokens = body[:sep_grid_at]
    st = _simulate_grid_walk(grid_tokens, rows, cols, cap_pair_names)
    matrix = st["matrix"]  # type: ignore
    n_sw = sum(v == P_SWITCH for row in matrix for v in row)
    n_ind = sum(v == P_INDUCTOR for row in matrix for v in row)

    after = body[sep_grid_at + 1:]      # tokens after SEP_grid
    n_state_tokens = cfg.n_states * (n_sw + n_ind)
    if len(after) < n_state_tokens:
        return {"STATE_OFF", "STATE_ON"}
    if len(after) == n_state_tokens:
        return {"SEP_states"}
    if after[n_state_tokens] != "SEP_states":
        return {"SEP_states"}
    if len(after) == n_state_tokens + 1:
        return {"EOS"}
    return set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_invalid_token_mask(
    partial_sequence: List[int],
    tokenizer: Tokenizer,
    config: ConstraintConfig | None = None,
) -> torch.Tensor:
    """Return BoolTensor[vocab_size]; True where the token is invalid next."""
    ccfg = config or ConstraintConfig()
    names = [tokenizer.id2tok[t] for t in partial_sequence]

    allowed_names = _allowed_next(names, tokenizer, ccfg)
    mask = torch.ones(tokenizer.vocab_size, dtype=torch.bool)
    for nm in allowed_names:
        mask[tokenizer.tok2id[nm]] = False
    return mask
