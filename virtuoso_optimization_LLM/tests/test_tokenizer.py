"""Tokenizer round-trip + surrogate-encoder compatibility."""

import os
import sys

import numpy as np
import pytest

from src.tokenizer import Tokenizer, TokenizerConfig

_WM = "/disk/mcao53/RFIC_RL/world_model_test"


def _tiny_cfg() -> TokenizerConfig:
    return TokenizerConfig(rows=6, cols=8, n_states=2, n_bins=16,
                           n_inductor_values=1,
                           cap_pair_min_dist=2, cap_pair_max_dist=3,
                           max_cap_pairs=4)


def _hand_record():
    rows, cols = 6, 8
    m = [[0] * cols for _ in range(rows)]
    # feed-column ports (metal in col 0 / col cols-1 -> PORT token)
    m[2][0] = 1
    m[3][cols - 1] = 1
    # one switch, one inductor
    m[1][3] = 3
    m[3][5] = 2
    # a horizontal cap pair: head (4,1) metal, interior (4,2) ground, dest (4,3) metal
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


def test_roundtrip_structure_exact():
    tok = Tokenizer(_tiny_cfg())
    rec = _hand_record()
    dec = tok.decode(tok.encode(rec))

    assert dec["matrix"] == rec["matrix"]
    assert dec["switch_states"] == rec["switch_states"]
    assert dec["inductor_switch_states"] == rec["inductor_switch_states"]
    assert dec["inductor_values"] == rec["inductor_values"]
    # cap pair endpoints reconstructed exactly
    assert dec["cap_pixel_pairs"] == rec["cap_pixel_pairs"]


def test_roundtrip_values_within_one_bin():
    cfg = _tiny_cfg()
    tok = Tokenizer(cfg)
    rec = _hand_record()
    dec = tok.decode(tok.encode(rec))

    def tol(lo, hi):
        return (hi - lo) / cfg.n_bins

    assert abs(dec["cap_values"][0] - rec["cap_values"][0]) <= tol(cfg.cap_min, cfg.cap_max)
    assert abs(dec["inductor_cap_values"][0] - rec["inductor_cap_values"][0]) <= \
        tol(cfg.inductor_cap_min, cfg.inductor_cap_max)
    assert abs(dec["cap_pair_values"][0] - rec["cap_pair_values"][0]) <= \
        tol(cfg.cap_pair_min, cfg.cap_pair_max)


def test_encode_decode_encode_is_fixed_point():
    """Quantization is idempotent: tokens are stable after one round-trip."""
    tok = Tokenizer(_tiny_cfg())
    rec = _hand_record()
    toks = tok.encode(rec)
    assert tok.encode(tok.decode(toks)) == toks


def test_vertical_cap_pair():
    tok = Tokenizer(_tiny_cfg())
    rec = _hand_record()
    # replace with a vertical pair: (1,6)-(3,6), interior (2,6) ground
    rec["matrix"][4][1] = 0
    rec["matrix"][4][3] = 0
    rec["matrix"][1][6] = 1
    rec["matrix"][3][6] = 1
    rec["cap_pixel_pairs"] = [[[1, 6], [3, 6]]]
    dec = tok.decode(tok.encode(rec))
    assert dec["cap_pixel_pairs"] == [[[1, 6], [3, 6]]]


def test_diagonal_pair_rejected():
    tok = Tokenizer(_tiny_cfg())
    rec = _hand_record()
    rec["cap_pixel_pairs"] = [[[1, 1], [3, 3]]]
    with pytest.raises(ValueError):
        tok.encode(rec)


def test_save_load(tmp_path):
    tok = Tokenizer(_tiny_cfg())
    p = tmp_path / "vocab.json"
    tok.save(str(p))
    tok2 = Tokenizer.load(str(p))
    assert tok2.vocab_size == tok.vocab_size
    rec = _hand_record()
    assert tok2.encode(rec) == tok.encode(rec)


def test_decoded_record_feeds_world_model_encoder():
    """The decoded dict must round-trip through encode.py:encode_sample."""
    if _WM not in sys.path:
        sys.path.insert(0, _WM)
    import encode as wm_encode  # noqa: E402

    tok = Tokenizer(_tiny_cfg())
    rec = tok.decode(tok.encode(_hand_record()))

    cfg = wm_encode.build_encode_config({
        "grid": {
            "rows": 6, "cols": 8,
            "cap_min": 100e-15, "cap_max": 1000e-15,
            "n_inductor_values": 1,
            "inductor_cap_min": 200e-15, "inductor_cap_max": 1000e-15,
            "cap_pair_min": 50e-15, "cap_pair_max": 1000e-15,
        }
    })
    img = wm_encode.encode_sample(rec, 0, cfg)
    assert img.shape == (3, 6, 8)
    assert img.dtype == np.float32
    # state channel reflects switch_states[0] = ON at the switch pixel (1,3)
    assert img[1, 1, 3] == 1.0
