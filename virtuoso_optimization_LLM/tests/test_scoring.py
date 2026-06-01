"""GA PV-contrast scoring reuse + per-state best ratchet."""

import numpy as np

from src.scoring import (
    PerStateBest, ScoreNorm, design_cost, design_score, design_score_matrix,
    load_specs,
)

_FINETUNE = ("/disk/mcao53/RFIC_RL/virtuoso_optimization_GA/"
             "configs/base/finetune_3_states.yaml")


def _specs():
    return load_specs(_FINETUNE)


def _synthetic_s21(freqs, specs):
    """0 dB inside each state's passband, -40 dB elsewhere."""
    K = len(specs)
    s = np.full((K, len(freqs)), -40.0, dtype=np.float32)
    for spec in specs:
        lo, hi = spec.passband
        band = (freqs >= lo) & (freqs <= hi)
        s[spec.state_index, band] = 0.0
    return s


def test_specs_load():
    specs = _specs()
    assert len(specs) == 3
    assert all(sp.enable_pv_contrast for sp in specs)


def test_score_vector_semantics():
    freqs = np.linspace(0, 70e9, 71)
    specs = _specs()
    s21 = _synthetic_s21(freqs, specs)
    M = design_score_matrix(freqs, s21, specs)         # (3, 3)
    assert M.shape == (3, 3)
    pb_level, contrast, pb_std = M[:, 0], M[:, 1], M[:, 2]
    np.testing.assert_allclose(pb_level, 0.0, atol=1e-5)   # 0 dB passband
    assert np.all(contrast > 35.0)                          # ~0 - (-40)
    np.testing.assert_allclose(pb_std, 0.0, atol=1e-5)      # flat


def test_design_score_is_negative_softmax_cost():
    freqs = np.linspace(0, 70e9, 71)
    specs = _specs()
    s21 = _synthetic_s21(freqs, specs)
    c = design_cost(freqs, s21, specs, agg="soft_max", temp=2.0)
    sc = design_score(freqs, s21, specs, agg="soft_max", temp=2.0)
    assert sc == -c
    # a good (passband-met, high-contrast) design beats an all -40 dB design
    bad = np.full_like(s21, -40.0)
    assert design_score(freqs, s21, specs) > design_score(freqs, bad, specs)


def test_per_state_best_ratchet_independent():
    psb = PerStateBest(K=3)
    a = np.array([[-3.0, 20.0, 2.0],
                  [-5.0, 15.0, 3.0],
                  [-2.0, 30.0, 1.0]], dtype=np.float32)
    b = np.array([[-1.0, 18.0, 4.0],     # state0: better level, worse contrast/std
                  [-9.0, 25.0, 1.0],     # state1: worse level, better contrast/std
                  [-4.0, 35.0, 0.5]], dtype=np.float32)
    psb.update(a)
    psb.update(b)
    best = psb.best_matrix()
    # pv_pb_level / pv_contrast -> max ; pv_pb_std -> min, per state independently
    np.testing.assert_allclose(best[:, 0], [-1.0, -5.0, -2.0])   # level: max
    np.testing.assert_allclose(best[:, 1], [20.0, 25.0, 35.0])   # contrast: max
    np.testing.assert_allclose(best[:, 2], [2.0, 1.0, 0.5])      # std: min


def test_score_norm_roundtrip():
    norm = ScoreNorm.from_config({
        "pv_pb_level": {"center": -5.0, "scale": 5.0},
        "pv_contrast": {"center": 20.0, "scale": 20.0},
        "pv_pb_std": {"center": 2.0, "scale": 2.0},
    })
    v = np.array([-1.0, 40.0, 0.5], dtype=np.float32)
    np.testing.assert_allclose(norm.denormalize(norm.normalize(v)), v, rtol=1e-6)
