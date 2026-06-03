"""
Smoke tests for Phase-2 generation.  The full-iteration end-to-end test now
requires the real parallel Virtuoso engine (the dev surrogate-sim path was
removed), so it is skipped in environments without Virtuoso.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _tiny_cfg_yaml(tmp_path) -> str:
    p = tmp_path / "tiny.yaml"
    p.write_text(textwrap.dedent("""
      grid: {rows: 6, cols: 8, n_states: 2,
             cap_min: 1.0e-13, cap_max: 1.0e-12, n_inductor_values: 1,
             inductor_cap_min: 2.0e-13, inductor_cap_max: 1.0e-12,
             cap_pair_min: 5.0e-14, cap_pair_max: 1.0e-12,
             cap_pair_min_dist: 2, cap_pair_max_dist: 3, max_cap_pairs: 2}
      tokenizer: {n_bins: 8}
      constraints: {cap_pair_interior_all_ground: true}
      scoring: {filter_specs_config: %s,
                aggregation_mode: soft_max, aggregation_temperature: 2.0}
      spec:
        score_scalars: [pv_pb_level, pv_contrast, pv_pb_std]
        normalization:
          pv_pb_level: {center: -10.0, scale: 10.0}
          pv_contrast: {center: 15.0, scale: 20.0}
          pv_pb_std: {center: 4.0, scale: 4.0}
        target_margins:
          pv_pb_level: {add: 0.5}
          pv_contrast: {add: 3.0}
          pv_pb_std: {mult: 0.8}
        bounds: {pv_pb_level: {max: 0.0}, pv_pb_std: {min: 0.0}}
        cold_start_target: [-3.0, 25.0, 2.0]
        stretch_frac: 0.5
      model: {model_dim: 32, n_layers: 2, n_heads: 2, ffn_mult: 2,
              dropout: 0.0, max_seq_len: 256, spec_dropout: 0.15}
      freq: {min: 0.0, max: 7.0e10, n: 9}
      surrogate: {checkpoint_path: null, ensemble_size: 2, device: cpu}
      data: {sims_dirs: [], use_synthetic: true, synthetic_n: 8,
             engine_config_path: %s}
      sft: {steps: 3, batch_size: 4, lr: 3.0e-4,
            type_weights: {pixel: 1.0, value: 3.0, state: 3.0}}
      iterated: {POOL_MULTIPLIER: 50, STEPS_PER_ITER: 500, N_ITERATIONS: 30,
                 K_RECAL: 3, n_parallel: 2, label_mode: soft_weighted,
                 soft_label_temperature: 0.1, cfg_scale: 4.0,
                 explore_frac: 0.2, n_targets: 8, temperature: 1.0,
                 n_per_target: 4, ground_steps_per_recal: 2, ground_replay: true}
      logging: {backend: none, dir: %s, project: rfic_lm}
      viz: {enabled: true}
    """ % ("/disk/mcao53/RFIC_RL/virtuoso_optimization_GA/configs/base/"
           "finetune_3_states.yaml",
           "/disk/mcao53/RFIC_RL/virtuoso_engine/config_ps31_gs25.yaml",
           str(tmp_path / "runs"))))
    return str(p)


def test_generate_decode_validity(tmp_path):
    """Constrained generation must yield grammatically decodable designs."""
    sys.path.insert(0, ROOT)
    from src.factory import build_model, build_surrogate, build_tokenizer, load_config

    cfg_path = _tiny_cfg_yaml(tmp_path)
    cfg = load_config(cfg_path)
    tok = build_tokenizer(cfg)
    model = build_model(cfg, tok)
    torch.manual_seed(0)

    spec = torch.zeros(3, cfg["grid"]["n_states"],
                       len(cfg["spec"]["score_scalars"]))
    seqs = model.generate_batch(spec, max_len=256, temperature=1.0,
                                cfg_scale=1.0, apply_constraints=True)
    from src.training import decode_to_design
    decoded = [decode_to_design(tok, s) for s in seqs]
    assert any(d is not None for d in decoded), "no grammatical design generated"
    # decoded designs must round-trip through the surrogate encoder
    surr = build_surrogate(cfg)
    for d in decoded:
        if d is not None:
            mean, var = surr.predict_multistate(d)
            assert mean.shape == (cfg["grid"]["n_states"], int(cfg["freq"]["n"]))
            break


@pytest.mark.skip(reason="Phase-2 recalibration now requires the real parallel "
                         "Virtuoso engine; no dev surrogate-sim path remains.")
def test_full_iteration_smoke(tmp_path):
    """Run scripts/train_iterated.py --smoke end to end (requires Virtuoso)."""
    cfg_path = _tiny_cfg_yaml(tmp_path)
    r = subprocess.run(
        [PY, os.path.join(ROOT, "scripts", "train_iterated.py"),
         "--config", cfg_path, "--smoke"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "done." in r.stdout
    assert "[iter 0]" in r.stdout
    # visualization artifacts written under <logging.dir>/plots
    plots = tmp_path / "runs" / "plots"
    pngs = list(plots.glob("iter_00000_*.png")) if plots.exists() else []
    assert pngs, f"no viz PNGs written; stdout:\n{r.stdout}"
