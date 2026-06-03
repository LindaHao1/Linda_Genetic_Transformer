"""Per-collate K-state resampling: see docs/running.md."""

import numpy as np

from src.training import _slice_example_for_k
from src.types import Design, SParams, TrainingExample


def _ex(K_avail: int, n_sw: int = 4, n_freq: int = 5) -> TrainingExample:
    # tiny 4x4 grid with `n_sw` switches in row 0; switch_states rows are
    # binary-encoded state indices so every row is unique (K_avail < 2**n_sw).
    rows, cols = 4, 4
    m = [[0] * cols for _ in range(rows)]
    for i in range(n_sw):
        m[0][i] = 3
    sw_states = [
        [(s >> k) & 1 for k in range(n_sw)] for s in range(K_avail)
    ]
    design = Design(
        pixel_grid=m,
        switch_caps=[1e-13] * n_sw,
        inductor_values=[],
        inductor_cap_values=[],
        cap_pixel_pairs=[],
        cap_pair_values=[],
        switch_states=sw_states,
        inductor_switch_states=[],
    )
    s21 = np.arange(K_avail * n_freq, dtype=np.float32).reshape(K_avail, n_freq)
    return TrainingExample(design=design,
                            sparams=SParams(freq_hz=np.arange(n_freq),
                                            s21_db=s21))


def test_slice_when_k_avail_gt_k():
    ex = _ex(K_avail=10)        # default n_sw=4 -> 16 unique state encodings
    rng = np.random.default_rng(0)
    out = _slice_example_for_k(ex, K=3, rng=rng)
    assert out.design.n_states == 3
    assert out.sparams.s21_db.shape == (3, ex.sparams.n_freq)
    # switch_states rows in the output must be EXACTLY rows from the input
    for row in out.design.switch_states:
        assert row in ex.design.switch_states
    # s21 rows correspond to the same state slice (state-major alignment)
    for i, row in enumerate(out.design.switch_states):
        j = ex.design.switch_states.index(row)
        assert np.array_equal(out.sparams.s21_db[i], ex.sparams.s21_db[j])


def test_slice_noop_when_k_avail_le_k():
    ex = _ex(K_avail=3)
    out = _slice_example_for_k(ex, K=3, rng=np.random.default_rng(0))
    assert out is ex                          # no-op short-circuit


def test_slice_deterministic_when_rng_none():
    ex = _ex(K_avail=10)
    out = _slice_example_for_k(ex, K=3, rng=None)
    # falls back to first K state indices in order
    assert out.design.switch_states == ex.design.switch_states[:3]
    assert np.array_equal(out.sparams.s21_db, ex.sparams.s21_db[:3])


def test_slice_diversity_across_calls():
    ex = _ex(K_avail=10)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(20):
        out = _slice_example_for_k(ex, K=3, rng=rng)
        seen.add(tuple(tuple(r) for r in out.design.switch_states))
    # 20 draws should yield more than one distinct triple (with overwhelming
    # probability — collision prob of all-same is < 1e-15)
    assert len(seen) > 1


def test_slice_respects_inductor_states_when_present():
    ex = _ex(K_avail=10)
    ex.design.pixel_grid[1][1] = 2                    # add 1 inductor
    ex.design.inductor_values = [0]
    ex.design.inductor_cap_values = [5e-13]
    ex.design.inductor_switch_states = [[k % 2] for k in range(10)]
    rng = np.random.default_rng(42)
    out = _slice_example_for_k(ex, K=3, rng=rng)
    assert len(out.design.inductor_switch_states) == 3
    # each kept inductor row must correspond to the same state index as
    # the matching switch row (lock-step slice)
    for i, (sw_row, ind_row) in enumerate(
            zip(out.design.switch_states, out.design.inductor_switch_states)):
        j_sw = ex.design.switch_states.index(sw_row)
        assert ex.design.inductor_switch_states[j_sw] == ind_row
