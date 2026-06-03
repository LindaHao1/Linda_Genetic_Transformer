"""GA PV-contrast scoring reuse + per-state best ratchet."""

import numpy as np

from src.scoring import (
    PerStateBest, ScoreNorm, TrainScoreTracker, design_cost, design_score,
    design_score_matrix, load_specs,
)
from src.types import Design, SParams, TrainingExample

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


def _dummy_example(s21: np.ndarray, freqs: np.ndarray) -> TrainingExample:
    return TrainingExample(
        design=Design(pixel_grid=[[0]]),
        sparams=SParams(freq_hz=freqs, s21_db=s21),
    )


def test_train_score_tracker_tracks_per_state_and_overall():
    freqs = np.linspace(0, 70e9, 71)
    specs = _specs()
    good = _synthetic_s21(freqs, specs)              # high-quality reference
    bad = np.full_like(good, -40.0)                  # uniform-bad

    trk = TrainScoreTracker(K=3, specs=specs, freqs=freqs)
    trk.update_from_s21(bad, step=0)
    trk.update_from_s21(good, step=5)
    trk.update_from_s21(bad, step=10)

    # best_overall must come from the good example, at step 5
    assert trk.best_overall_step == 5
    assert trk.best_overall == design_score(freqs, good, specs)
    assert trk.n_examples == 3

    # per-state best ratchet: good beats bad on contrast and matches level
    psb = trk.per_state_best.best_matrix()
    assert psb.shape == (3, 3)
    # contrast (col 1) should reflect the 'good' design's high contrast on
    # every state -- the bad design contributes 0 contrast
    assert (psb[:, 1] > 30.0).all()

    snap = trk.snapshot()
    assert snap["n_examples"] == 3
    assert snap["best_overall_step"] == 5
    assert snap["per_state_best"] is not None


def test_train_score_tracker_format_short_no_examples():
    freqs = np.linspace(0, 70e9, 71)
    trk = TrainScoreTracker(K=3, specs=_specs(), freqs=freqs)
    assert "no examples" in trk.format_short()


def test_train_score_tracker_batch_update():
    freqs = np.linspace(0, 70e9, 71)
    specs = _specs()
    good = _synthetic_s21(freqs, specs)
    bad = np.full_like(good, -40.0)
    trk = TrainScoreTracker(K=3, specs=specs, freqs=freqs)
    trk.update_batch_from_examples(
        [_dummy_example(bad, freqs), _dummy_example(good, freqs)], step=7)
    assert trk.best_overall_step == 7
    assert trk.n_examples == 2


def test_score_norm_roundtrip():
    norm = ScoreNorm.from_config({
        "pv_pb_level": {"center": -5.0, "scale": 5.0},
        "pv_contrast": {"center": 20.0, "scale": 20.0},
        "pv_pb_std": {"center": 2.0, "scale": 2.0},
    })
    v = np.array([-1.0, 40.0, 0.5], dtype=np.float32)
    np.testing.assert_allclose(norm.denormalize(norm.normalize(v)), v, rtol=1e-6)
