"""Constrained-decoding mask: grammar + cap-pair spatial validity."""

from src.constraints import get_invalid_token_mask
from src.tokenizer import Tokenizer, TokenizerConfig


def _tiny_cfg() -> TokenizerConfig:
    return TokenizerConfig(rows=6, cols=8, n_states=2, n_bins=16,
                           n_inductor_values=1,
                           cap_pair_min_dist=2, cap_pair_max_dist=3,
                           max_cap_pairs=4)


def _hand_record():
    rows, cols = 6, 8
    m = [[0] * cols for _ in range(rows)]
    m[2][0] = 1
    m[3][cols - 1] = 1
    m[1][3] = 3
    m[3][5] = 2
    m[4][1] = 1
    m[4][3] = 1
    return {
        "matrix": m,
        "cap_values": [300e-15],
        "inductor_values": [0],
        "inductor_cap_values": [500e-15],
        "cap_pixel_pairs": [[[4, 1], [4, 3]]],
        "cap_pair_values": [200e-15],
        "switch_states": [[1], [0]],
        "inductor_switch_states": [[1], [0]],
    }


def _allowed(seq, tok):
    mask = get_invalid_token_mask(seq, tok)
    return {tok.id2tok[i] for i in range(tok.vocab_size) if not bool(mask[i])}


def test_start_and_spec():
    tok = Tokenizer(_tiny_cfg())
    assert _allowed([], tok) == {"BOS"}
    assert _allowed([tok.t("BOS")], tok) == {"SEP_spec"}


def test_grid_phase_starts_at_feed_col():
    """Cursor at (0,0) is in the LEFT feed column: only GROUND or PORT."""
    tok = Tokenizer(_tiny_cfg())
    seq = [tok.t("BOS"), tok.t("SEP_spec")]
    a = _allowed(seq, tok)
    assert a == {"GROUND", "PORT"}, a


def test_grid_phase_interior_cell_allows_full_pixel_set():
    """An interior column allows the regular pixel set + cap-pair heads."""
    tok = Tokenizer(_tiny_cfg())
    cols = tok.config.cols
    # walk to cell (0, 1) -- first interior column
    seq = [tok.t("BOS"), tok.t("SEP_spec"), tok.t("GROUND")]
    a = _allowed(seq, tok)
    for n in ("GROUND", "METAL", "INDUCTOR", "SWITCH", "VIA"):
        assert n in a
    # PORT only on feed columns -> NOT here
    assert "PORT" not in a
    # cap pairs available
    assert "CAP_PAIR_H2" in a
    assert "CAP_PAIR_V2" in a
    # no state tokens leaking
    assert "STATE_ON" not in a


def test_feed_column_at_most_one_port():
    """After emitting PORT at (0,0), the rest of col 0 must be GROUND."""
    tok = Tokenizer(_tiny_cfg())
    cols = tok.config.cols
    seq = [tok.t("BOS"), tok.t("SEP_spec"), tok.t("PORT")]
    # walk to (1, 0) -- skip across row 0 (cols-1 cells of interior + the wrap)
    for _ in range(cols - 1):  # complete row 0
        seq.append(tok.t("GROUND"))
    # now at (1, 0), feed col with port_already=True
    a = _allowed(seq, tok)
    assert a == {"GROUND"}, a


def test_feed_column_last_cell_forces_port_when_no_port_emitted():
    """If we reach the last cell of a feed column with no PORT yet, force PORT."""
    tok = Tokenizer(_tiny_cfg())
    rows, cols = tok.config.rows, tok.config.cols
    seq = [tok.t("BOS"), tok.t("SEP_spec")]
    # walk through all earlier rows of column 0 as GROUND.  For each row r in
    # [0, rows-2]: emit GROUND at (r, 0), then GROUND for cols-1 interior cells.
    for r in range(rows - 1):
        seq.append(tok.t("GROUND"))                # (r, 0)
        for _ in range(cols - 1):                  # (r, 1) .. (r, cols-1)
            seq.append(tok.t("GROUND") if r != 0 or _ != cols - 2 else tok.t("GROUND"))
    # cursor is now at (rows-1, 0) — last row of left feed column, still 0 ports
    a = _allowed(seq, tok)
    assert a == {"PORT"}, a


def test_full_sequence_is_accepted_at_every_prefix():
    """The strongest test: a real encoded design must satisfy the grammar."""
    tok = Tokenizer(_tiny_cfg())
    toks = tok.encode(_hand_record())
    for k in range(len(toks)):
        mask = get_invalid_token_mask(toks[:k], tok)
        nxt = toks[k]
        assert not bool(mask[nxt]), (
            f"step {k}: next token {tok.id2tok[nxt]} was masked invalid")


def _walk_to_cursor(tok: Tokenizer, target_cell: int) -> list:
    """Build a prefix that walks the raster to `target_cell` with all GROUNDs."""
    seq = [tok.t("BOS"), tok.t("SEP_spec")]
    g = tok.t("GROUND")
    for _ in range(target_cell):
        seq.append(g)
    return seq


def test_horizontal_pair_forces_interior_ground_then_dest_metal():
    """After CAP_PAIR_H2 at (r,c), cell (r,c+1) must be GROUND, (r,c+2) METAL."""
    tok = Tokenizer(_tiny_cfg())
    cols = tok.config.cols
    # head at (r=2, c=1) -> raster cell 17
    head_cell = 2 * cols + 1
    seq = _walk_to_cursor(tok, head_cell)
    # emit CAP_PAIR_H2 + VAL_0
    seq.append(tok.t("CAP_PAIR_H2"))
    seq.append(tok.t("VAL_0"))
    # now at cell (2,2) -> must be GROUND
    a = _allowed(seq, tok)
    assert a == {"GROUND"}, a
    seq.append(tok.t("GROUND"))
    # now at cell (2,3) -> must be METAL (col 3 is not a feed column)
    a = _allowed(seq, tok)
    assert a == {"METAL"}, a


def test_vertical_pair_forces_next_row_ground_then_metal():
    """CAP_PAIR_V2 at (r=1,c=3): forces (2,3) GROUND, (3,3) METAL across rows."""
    tok = Tokenizer(_tiny_cfg())
    cols = tok.config.cols
    head_cell = 1 * cols + 3
    seq = _walk_to_cursor(tok, head_cell)
    seq.append(tok.t("CAP_PAIR_V2"))
    seq.append(tok.t("VAL_0"))
    # walk the rest of row 1: cells (1,4)..(1,7) -> 4 cells of GROUND
    for _ in range(cols - 4):
        seq.append(tok.t("GROUND"))
    # cursor is now at (2,0). walk to (2,3): (2,0), (2,1), (2,2) -> 3 GROUNDs.
    for _ in range(3):
        seq.append(tok.t("GROUND"))
    # cell (2,3) is forced GROUND by the pair interior
    a = _allowed(seq, tok)
    assert a == {"GROUND"}, a
    seq.append(tok.t("GROUND"))
    # walk to (3,3): (2,4)..(2,7) then (3,0), (3,1), (3,2) -> 4 + 3 = 7 GROUNDs
    for _ in range(cols - 4):
        seq.append(tok.t("GROUND"))
    for _ in range(3):
        seq.append(tok.t("GROUND"))
    # cell (3,3) is forced METAL by the pair dest
    a = _allowed(seq, tok)
    assert a == {"METAL"}, a


def test_pair_budget_exhausted_removes_cap_pair_tokens():
    cfg = _tiny_cfg()
    cfg.max_cap_pairs = 0
    tok = Tokenizer(cfg)
    seq = [tok.t("BOS"), tok.t("SEP_spec")]
    a = _allowed(seq, tok)
    assert "CAP_PAIR_H2" not in a
    assert "CAP_PAIR_V2" not in a
    assert "GROUND" in a


def test_sep_grid_then_states():
    """After SEP_grid (with no switches/inductors), expect SEP_states then EOS."""
    tok = Tokenizer(_tiny_cfg())
    cfg = tok.config
    # walk all GROUND cells, then SEP_grid -> no switches/inductors -> 0 state tokens.
    seq = _walk_to_cursor(tok, cfg.rows * cfg.cols)
    assert _allowed(seq, tok) == {"SEP_grid"}
    seq.append(tok.t("SEP_grid"))
    assert _allowed(seq, tok) == {"SEP_states"}
    seq.append(tok.t("SEP_states"))
    assert _allowed(seq, tok) == {"EOS"}
