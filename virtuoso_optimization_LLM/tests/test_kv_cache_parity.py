"""
KV-cache parity: `FilterLM.generate_batch` (incremental, cached) must produce
exactly the same tokens as the reference full-sequence re-forward decode, under
greedy sampling (temperature<=0) -- with and without classifier-free guidance.

This guards the optimization in src/model.py: the cached decode is only a speed
change, not a behavior change.  Run on a node where `torch` imports cleanly.
"""

import os
import sys
import textwrap
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _tiny_cfg(tmp_path) -> str:
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
        target_margins: {pv_pb_level: {add: 0.5}, pv_contrast: {add: 3.0},
                         pv_pb_std: {mult: 0.8}}
        bounds: {pv_pb_level: {max: 0.0}, pv_pb_std: {min: 0.0}}
        cold_start_target: [-3.0, 25.0, 2.0]
        stretch_frac: 0.5
      model: {model_dim: 32, n_layers: 2, n_heads: 2, ffn_mult: 2,
              dropout: 0.0, max_seq_len: 256, spec_dropout: 0.15}
      freq: {min: 0.0, max: 7.0e10, n: 9}
      surrogate: {checkpoint_path: null, ensemble_size: 2, device: cpu}
      data: {sims_dirs: [], use_synthetic: true, synthetic_n: 8}
      sft: {steps: 3, batch_size: 4, lr: 3.0e-4,
            type_weights: {pixel: 1.0, value: 3.0, state: 3.0}}
      logging: {backend: none, dir: %s, project: rfic_lm}
      viz: {enabled: false}
    """ % ("/disk/mcao53/RFIC_RL/virtuoso_optimization_GA/configs/base/"
           "finetune_3_states.yaml", str(tmp_path / "runs"))))
    return str(p)


def _ref_generate(model, spec, max_len, cfg_scale, apply_constraints):
    """Reference: the pre-KV-cache decode -- re-forward the whole sequence
    every step, greedy argmax."""
    from src.constraints import get_invalid_token_mask
    model.eval()
    B = spec.shape[0]
    bos, eos, pad = model.tok.t("BOS"), model.tok.t("EOS"), model.tok.t("PAD")
    seqs = [[bos] for _ in range(B)]
    done = [False] * B
    with torch.no_grad():
        for _ in range(max_len - 1):
            if all(done):
                break
            T = max(len(s) for s in seqs)
            batch = torch.tensor([s + [pad] * (T - len(s)) for s in seqs],
                                 device=model.device)
            cond = model.forward(batch, spec,
                                 spec_drop=torch.zeros(B, dtype=torch.bool))
            if cfg_scale != 1.0:
                uncond = model.forward(batch, spec,
                                       spec_drop=torch.ones(B, dtype=torch.bool))
            for b in range(B):
                if done[b]:
                    continue
                last = len(seqs[b]) - 1
                lg = cond[b, last]
                if cfg_scale != 1.0:
                    lg = uncond[b, last] + cfg_scale * (lg - uncond[b, last])
                if apply_constraints:
                    m = get_invalid_token_mask(seqs[b], model.tok, None)
                    lg = lg.masked_fill(m.to(lg.device), float("-inf"))
                nxt = int(lg.argmax())
                seqs[b].append(nxt)
                if nxt == eos:
                    done[b] = True
    return seqs


def _build(tmp_path):
    from src.factory import build_model, build_tokenizer, load_config
    cfg = load_config(_tiny_cfg(tmp_path))
    torch.manual_seed(0)
    tok = build_tokenizer(cfg)
    model = build_model(cfg, tok)
    return cfg, model


def _spec(cfg, model, B):
    K = cfg["grid"]["n_states"]
    S = len(cfg["spec"]["score_scalars"])
    torch.manual_seed(123)
    return torch.randn(B, K, S, device=model.device)


def test_kv_cache_parity_no_cfg(tmp_path):
    cfg, model = _build(tmp_path)
    spec = _spec(cfg, model, B=3)
    got = model.generate_batch(spec, max_len=120, temperature=0.0,
                               cfg_scale=1.0, apply_constraints=True)
    ref = _ref_generate(model, spec, 120, cfg_scale=1.0, apply_constraints=True)
    assert got == ref, "cached decode diverged from full-forward (no CFG)"


def test_kv_cache_parity_with_cfg(tmp_path):
    cfg, model = _build(tmp_path)
    spec = _spec(cfg, model, B=3)
    got = model.generate_batch(spec, max_len=120, temperature=0.0,
                               cfg_scale=4.0, apply_constraints=True)
    ref = _ref_generate(model, spec, 120, cfg_scale=4.0, apply_constraints=True)
    assert got == ref, "cached decode diverged from full-forward (CFG=4.0)"


def test_kv_cache_speedup(tmp_path):
    """Cached decode should be at least as fast as the re-forward reference;
    informational (asserts only that it is not pathologically slower)."""
    cfg, model = _build(tmp_path)
    spec = _spec(cfg, model, B=4)
    t0 = time.perf_counter()
    model.generate_batch(spec, max_len=200, temperature=0.0, cfg_scale=4.0)
    t_cached = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ref_generate(model, spec, 200, cfg_scale=4.0, apply_constraints=True)
    t_ref = time.perf_counter() - t0
    print(f"cached={t_cached:.3f}s  ref={t_ref:.3f}s  speedup={t_ref / t_cached:.1f}x")
    assert t_cached <= t_ref * 1.5, "cached decode unexpectedly slower than reference"
