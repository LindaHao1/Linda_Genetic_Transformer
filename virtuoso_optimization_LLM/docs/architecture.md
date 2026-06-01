# Architecture

Generative decoder-only LM that produces switchable RF filter layouts
conditioned on multi-state S-parameter targets, trained by iterated
surrogate-guided SFT (no RL). Read `world_model_notes.md` first.
**To run the pipeline, see [`running.md`](running.md). Token format /
vocabulary: [`tokenizer.md`](tokenizer.md).**

## Clarifying questions — RESOLVED

The prompt's work order asks to confirm four things before coding. Answered:

| Question | Decision |
|---|---|
| Grid dimensions | **Configurable, default 21×27** (matches world model + dataset). |
| Number of states K | **Configurable, default 3** (matches GA/SA FilterStateSpec). |
| Spec frequency bins | **71** (0–70 GHz, 1 GHz), from the world model. |
| Ensemble surrogates | **Yes — 5-member ensemble.** Checkpoint path from config (`TODO(user)`); untrained fallback for the smoke test. |

Additional resolved ambiguities (the prompt left these open):

- **Cap-pair tokenization** — *in-grid combined orient+length token at the
  head cell* (`CAP_PAIR_H{L}`, `CAP_PAIR_V{L}`), followed inline by one
  `VAL_b`. Interior cells along the path are forced to GROUND and the dest
  to METAL by the constraint mask as the raster walk continues; vertical
  pairs reach into a later row. See `tokenizer.md`.
- **Valid-token mask for cap pairs** — masking enforces, at decode time, that
  the chosen (head, orientation, length) yields a path whose **interior cells
  are all ground** and whose **destination cell is metal**, in bounds, budget
  not exhausted. (User's rule; stricter than `feasibility.py`'s "≥1 ground".)
- **Spec = compact GA PV-contrast score vector**, NOT a raw S21 curve
  (superseded — see "PV-score conditioning" below). Per state: `[pv_pb_level,
  pv_contrast, pv_pb_std]` (3) → K×3 via a small MLP, prefixed. Hindsight
  relabeling uses each design's *achieved* score vector.
- Switch cap-bank size = 1 (matches flat `cap_values` in `encode.py`).

## PV-score conditioning (supersedes raw-S21 spec)

The LM mirrors the GA: it is conditioned on, and ranked by, the GA's
**PV-contrast** scoring (`virtuoso_optimization_SA/cost.py`, PV mode enabled
in `virtuoso_optimization_GA/configs/base/finetune_3_states.yaml`), reused via
`src/scoring.py` (never reimplemented).

- **Conditioning vector** (`spec.score_scalars`, default
  `[pv_pb_level, pv_contrast, pv_pb_std]`): per state, normalized
  `(x−center)/scale` (`spec.normalization`), encoded by the spec MLP →
  K prefix embeddings. `pv_pb_level` ↑ (worst-k passband S21 dB),
  `pv_contrast` ↑ (passband − worst-k stopband dB), `pv_pb_std` ↓ (ripple).
- **Ranking score** (`src/scoring.py:design_score`):
  `−aggregate(per-state compute_state_cost)` with **`soft_max`** aggregation
  (`cost.py:_aggregate_state_costs`, worst state dominates). Fixed state↔spec
  assignment (i↔i); dynamic Hungarian = `TODO(user)`.
- **Targets** (`src/training.py:sample_score_targets`): per-state, per-scalar
  **running best from simulated designs only** (`PerStateBest`, decoupled per
  state) + a margin on "stretch" targets, with physical clamps —
  `pv_pb_level ≤ 0 dB` (passive), `pv_pb_std ≥ 0`. Cold start uses
  `spec.cold_start_target` until any design is simulated.
- Reused: `cost.py:specs_from_config` (:961), `compute_state_cost` (:227),
  `compute_state_cost_breakdown` (:441), `_aggregate_state_costs` (:782).

## Cap pairs as in-grid pixel-family tokens

`encode.py`'s `matrix` only holds codes 0–4 (ground/metal/inductor/switch/via).
A capacitor pair is an *overlay*: two metal-pixel coordinates + a value,
written into `record["cap_pixel_pairs"]` / `["cap_pair_values"]` and rendered
by `make_value_channel` (`encode.py:146`). Validity rules live in
`virtuoso_optimization_SA/feasibility.py:306` `check_cap_pair_valid`.

The LM emits a pair as a single pixel-family token at its head cell during the
raster walk (`CAP_PAIR_{H|V}{L}`), immediately followed inline by a `VAL_b`.
The matrix code at the head is METAL; the geometry (orient + length → dest +
interior) is recovered at decode time and recorded in `cap_pixel_pairs`. The
constraint mask forces interior cells to GROUND and the dest cell to METAL as
the raster cursor reaches them — preserving the encode-time validity rules
while keeping the design fully autoregressive.

## Token vocabulary (`src/tokenizer.py`)

JSON-persisted, single source of truth.

- **Structural**: `BOS EOS PAD SEP_spec SEP_grid SEP_states`. The grid block
  is the only payload between `SEP_spec` and `SEP_grid`; values are inline
  with their pixel.
- **Pixel-type** (grid): `GROUND METAL INDUCTOR SWITCH VIA PORT`. `PORT` is the
  position-locked feed-column pixel (col 0 / col W−1), decodes to matrix 1.
- **Cap-pair pixel-family**: `CAP_PAIR_H{L}` and `CAP_PAIR_V{L}` for each `L`
  in `[cap_pair_min_dist, cap_pair_max_dist]`. Emitted at the head cell only.
- **Value bins**: `VAL_0..VAL_{N_BINS-1}` (default 64), shared; the family
  (switch-cap / inductor-MIM-cap / cap-pair) is positional and de-quantized with
  that family's `[min,max]` from `EncodeConfig`.
- **Inductor categorical**: `INDVAL_0..INDVAL_{n_inductor_values-1}`.
- **State codes**: `STATE_OFF STATE_ON`.

`Tokenizer`: `encode(design_dict)->List[int]`, `decode(tokens)->design_dict`,
`vocab_size`, `get_token_type(id)`, `save`/`load`.

## Sequence layout

```
BOS
<spec prefix: K×n_scalars PV-scores → MLP → K embeddings>   SEP_spec
<grid: H*W cells, raster; per cell one pixel-family token
       + inline VAL (after SWITCH / CAP_PAIR_*) or
       INDVAL + VAL (after INDUCTOR)>                       SEP_grid
<states: K × (n_sw + n_ind) STATE_*; state-major>           SEP_states
EOS
```

`n_sw`/`n_ind` are counted from the decoded grid, fixing the states-segment length.

### In-grid cap pairs

One pair = `CAP_PAIR_{H|V}{L}  VAL_b` at its head cell. From head
`(r,c)`, orient `O`, and length `L`: destination = `(r, c+L)` if `H` else
`(r+L, c)`; interior = the `L−1` cells between. Decode rebuilds
`cap_pixel_pairs = [[ [r,c], [dest_r,dest_c] ]]` and `cap_pair_values` from
the de-quantized bin. The interior cells along the path are forced to GROUND
and the dest cell to METAL by the constraint mask as the raster cursor reaches
them; for vertical pairs the dest reaches into a later row.

## Constrained decoding (`src/constraints.py`)

Signature per prompt:
`get_invalid_token_mask(partial_sequence, tokenizer, config) -> BoolTensor[vocab]`
(True = invalid). Applied in `generate()` every step:
`logits[mask] = -inf` before sampling.

Three layers:

1. **Structural grammar (implemented, always on)** — only the legal next token
   family for the current segment/position is allowed.
2. **Cap-pair spatial validity (implemented — user's design)** — at a fresh
   raster cell `(r,c)`: in addition to the regular pixel set, allow each
   `CAP_PAIR_{O}{L}` whose path `(r,c) → dest` fits in bounds, whose interior
   cells have not been emitted yet and are not conflict-forced by an earlier
   pair, whose dest is similarly free, and whose budget is not exhausted. Once
   a head is consumed, the interior cells along its path are forced to GROUND
   and the dest cell to METAL when the raster cursor reaches them. Config flag
   `cap_pair_interior_all_ground` (default True; matches the user's rule).
3. **`TODO(user)` placeholders** — switch/inductor Chebyshev separation
   (`feasibility.py:125`), component-count budgets, port-pixel locking.

## Model (`src/model.py`)

Decoder-only transformer, 8–12 layers, `model_dim` 384–512 (~30M). 2D learned
positional embeddings for the grid segment (sized to max grid), 1D elsewhere;
additive segment embeddings (spec/grid/values/cappairs/states). Spec is a causal
prefix (prefix-LM). CFG: 15% spec-dropout in training (learned null embedding);
`cfg_scale` default 4.0 at inference. `generate()` applies the mask each step.

## Surrogate (`src/surrogate.py`)

Wraps `world_model_test` (imported via `sys.path`, not duplicated):
`build_encode_config`, `encode_sample`, `WorldModel`, `EnsembleModel`.
`predict`, `predict_multistate` (K calls, stack `(K,71)` mean+var),
`predict_multistate_batch`, `score` (default neg-MSE dB, pluggable). Untrained
fallback logs a warning; single member ⇒ variance 0 + warning.

## Data / simulator / training

- `src/types.py`: `Design`, `SParams`, `TrainingExample`; `Design.to_record()`
  emits exactly the `encode.py` dict.
- `src/dataset.py`: parse real sims (`config.yaml` `pixelated:` +
  `*_state*.s2p`, per `process_dataset.py:119`); hindsight relabel; optional
  surrogate-labeled random feasible designs via `feasibility.py`.
- `src/simulator.py`: `run_simulator` raises `NotImplementedError`; dev flag
  returns surrogate ± noise for the smoke test. Real wiring
  (`ParallelVirtuosoEngine`, `ga_optimizer.py:1104`) is `TODO(user)`.
- `scripts/train_sft.py` (Phase 1) and `scripts/train_iterated.py` (Phase 2):
  fully specified in **Training loop (detailed)** below.
- `scripts/eval.py` + logging: reward-hacking warning when surrogate score
  rises while true score plateaus.

## Training loop (detailed)

### Phase 1 — initial SFT (`scripts/train_sft.py`)

- **Objective**: next-token cross-entropy over the full design token sequence
  (`BOS … EOS`). `logits[i]` predicts `token[i+1]`.
- **Conditioning = hindsight**: the spec is the example's *own achieved*
  **PV-score vector** (`K×len(score_scalars)`), computed from its achieved S21
  via `src/scoring.py:design_score_matrix` then normalized — never an intended
  target. Synthetic bootstrap "achieved" = surrogate prediction.
- **Token-type loss weighting** (`src/training.py:token_type_weights`):
  per-token CE is multiplied by a vocab weight then averaged by the sum of
  weights. Defaults: `pixel 1.0`, `value 3.0`, `state 3.0`; cap-pair record
  tokens (`pos/orient/len`) and `indval` also `3.0`; `structural`/reserved
  `1.0`. Overridable via `sft.type_weights`.
- **CFG spec-dropout**: in training mode each sample's spec is replaced by a
  learned null embedding with prob `model.spec_dropout` (0.15).
- **Optimizer**: AdamW, `lr = sft.lr` (3e-4), weight_decay 1e-4, grad-norm
  clip 1.0.
- **Crucial params**: `sft.steps`, `sft.batch_size`, `sft.lr`,
  `model.spec_dropout`, `model.max_seq_len`; data from `data.sims_dirs` or
  `data.synthetic_n` synthetic designs. Output: `{model, tokenizer_config}`.

### Phase 2 — iterated surrogate-guided SFT (`scripts/train_iterated.py`)

Per iteration `it` in `range(N_ITERATIONS)`:

1. **Sample targets** — `n_targets` normalized PV-score targets
   (`K×len(score_scalars)`) via `sample_score_targets`: per-state running best
   (`PerStateBest`, from simulated designs only) + margin on stretch targets,
   physically clamped (`pv_pb_level ≤ 0`, `pv_pb_std ≥ 0`); cold start uses
   `spec.cold_start_target`.
2. **Generate pool** — for each target group, generate `POOL_MULTIPLIER`
   candidates conditioned on that target, with `apply_constraints=True`,
   sampling `temperature` and CFG `cfg_scale`. Pool ≈
   `n_targets × POOL_MULTIPLIER` (minus ungrammatical decodes).
3. **Score** — each candidate → `Design`; `surrogate.predict_multistate`
   gives mean S21 `(K×71)`; `score = scoring.design_score(freqs, mean, specs,
   soft_max, T)` (see **Score definition**).
4. **Soft labels per target group** (`compute_soft_weights`,
   `mode = label_mode`):
   - `soft_weighted`: `w_i = exp((s_i − max s)/T)`, renormalized so the
     group's mean weight is 1, with `T = soft_label_temperature`.
   - `rejection_topk` / `rejection_topp`: weight 1 for the top
     `(1 − explore_frac)` fraction by score, else 0.
5. **Soft-label SFT** — `STEPS_PER_ITER` mini-steps; each draws a
   `sft.batch_size` batch from the pool, **conditioned on the TARGET score
   vector** (`collate(..., spec_vectors=target)`, not hindsight here),
   next-token CE × token-type weight × per-sample soft weight.
6. **Visualize best generated** (if `viz.enabled`): the top-`design_score`
   pool design → grid + WM-predicted S21 + achieved-vs-target scores →
   `<logging.dir>/plots/iter_{N:05d}_best_gen.png`.
7. **Recalibration** every `K_RECAL` iters (`it % K_RECAL == 0`):
   - epistemic uncertainty = mean ensemble variance per candidate;
   - `select_for_simulation`: `M_RECAL` designs = top `(1 − explore_frac)·M`
     by score **+** `explore_frac·M` by highest uncertainty;
   - simulate: real `run_simulator()` (TODO(user)) or `run_simulator_dev`
     (surrogate ± Gaussian noise `noise_db=0.5`) when `use_surrogate_sim`;
   - `dataset.add_relabeled(...)` → cumulative **real** set; **`PerStateBest`
     updated per state from simulated S21 scores**;
     `surrogate.retrain(real_only, epochs=1)` (per-member MSE, AdamW 3e-4);
   - **calibration logged**: `calib_mse`, `calib_r2` (surrogate vs true S21),
     `mean_true_score`/`best_true_score` (GA score on simulated S21);
   - visualize best simulated → `iter_{N:05d}_best_sim.png`.
8. **Held-out eval** (`run_eval`) on a fixed cold-start target set:
   `best_of_n_mean_score` (max GA score per target), `valid_frac`,
   `pool_diversity` (mean pairwise pixel-grid Hamming),
   `n_per_target = max(2, POOL_MULTIPLIER//2)`.
9. **Reward-hacking warning**: after ≥3 recal points, emit
   `REWARD_HACK_WARNING` if surrogate score rose vs 3 recals ago while
   `mean_true_score` did not improve.

`--smoke` overrides: `POOL_MULTIPLIER=4, STEPS_PER_ITER=3, N_ITERATIONS=1,
K_RECAL=1, M_RECAL=4, n_targets=2`.

### Crucial parameters (`configs/default.yaml`)

| Key | Default | Role |
|---|---|---|
| `iterated.N_ITERATIONS` | 30 | outer iterations |
| `iterated.n_targets` | 8 | target specs sampled per iteration |
| `iterated.POOL_MULTIPLIER` | 50 | candidates generated per target |
| `iterated.STEPS_PER_ITER` | 500 | SFT mini-steps per iteration |
| `iterated.label_mode` | soft_weighted | soft_weighted \| rejection_topk \| rejection_topp |
| `iterated.soft_label_temperature` | 0.1 | softmax temp for soft weights |
| `iterated.cfg_scale` | 4.0 | classifier-free guidance strength |
| `iterated.temperature` | 1.0 | sampling temperature |
| `iterated.K_RECAL` | 3 | resimulate every K iterations |
| `iterated.M_RECAL` | 64 | designs resimulated per recalibration |
| `iterated.explore_frac` | 0.2 | recal budget for high-uncertainty designs |
| `iterated.use_surrogate_sim` | true | use dev simulator stand-in |
| `sft.batch_size` / `sft.lr` | 16 / 3e-4 | SFT batch / learning rate |
| `model.spec_dropout` | 0.15 | CFG spec-dropout fraction |
| `model.max_seq_len` | 1024 | max generated token length |
| `grid.n_states` (K) | 3 | states per filter |
| `freq.n` | 71 | S21 bins for the **surrogate** (not the spec) |
| `scoring.filter_specs_config` | finetune_3_states.yaml | GA `filter_specs` source |
| `scoring.aggregation_mode` | soft_max | per-state cost aggregation |
| `scoring.aggregation_temperature` | 2.0 | soft_max temperature |
| `spec.score_scalars` | pb_level,contrast,pb_std | per-state conditioning scalars |
| `spec.target_margins` | +0.5/+3.0/×0.8 | stretch-target step per scalar |
| `spec.bounds` | pb_level≤0, pb_std≥0 | physical target clamps |
| `spec.stretch_frac` | 0.5 | fraction of targets that get the margin |
| `viz.enabled` | true | write best-design plots to `<dir>/plots` |

## Score definition

The ranking score is the GA's own objective, reused via
`src/scoring.py:design_score` on world-model-predicted S21
`(K×71, dB)` + the per-state `FilterStateSpec` (PV-contrast mode):

```
state_cost_k = compute_state_cost(freqs, s21_k, spec_k)        # cost.py:227
score        = − _aggregate_state_costs(state_costs, "soft_max", T)
```

`soft_max` aggregation `= max_k c_k + (1/T)·log Σ_k exp(T·(c_k−max))` is
dominated by the **worst state**, so the model is pushed to fix its weakest
state first. Higher score = better (lower GA cost). This single scalar drives
soft-label weights (step 4), recalibration selection (step 7), and held-out
eval (step 8).

The **conditioning vector** is a separate, smaller signal: per state
`[pv_pb_level, pv_contrast, pv_pb_std]` from
`compute_state_cost_breakdown` (cost.py:441) — semantic dB scalars
(`pv_pb_level` ↑, `pv_contrast` ↑, `pv_pb_std` ↓), normalized for the MLP.

`Surrogate.score`/`neg_mse_db` is legacy curve-MSE, kept only for ad-hoc
S21-curve comparisons. Dynamic state↔spec assignment (Hungarian) is
`TODO(user)` — cost.py's dynamic variant is `.s2p`-path based; we use fixed
`state i ↔ spec i`.

## Verification

`pytest` (20 tests, rfic2 env): `test_tokenizer.py` (round-trip +
`encode_sample` compatibility + exact `cap_pixel_pairs`), `test_constraints.py`
(cap-pair mask; placeholders all-False), `test_scoring.py` (PV-contrast reuse:
`pv_pb_level/contrast/std` semantics, `design_score == −soft_max cost`,
`PerStateBest` per-state ratchet + `pv_pb_level≤0` clamp),
`test_pipeline_dryrun.py` (one full Phase-2 iteration, tiny model, untrained
ensemble, surrogate-sim, **asserts viz PNGs written** — must pass). Plus a
manual `train_sft.py` tiny run (loss decreases with PV-score hindsight spec).
