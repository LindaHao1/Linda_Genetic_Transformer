# How to run

All commands assume the project directory and the `rfic2` conda Python:

```bash
cd /disk/mcao53/RFIC_RL/virtuoso_optimization_LLM
PY=/disk/mcao53/conda_env/rfic2/bin/python
```

Dependencies (torch, numpy, pyyaml, einops, skrf, matplotlib, pytest) are
already in `rfic2`. The scripts add the project root to `sys.path`, so run
them from this directory.

## 0. Sanity check (tests)

```bash
$PY -m pytest tests/ -q
```

20 tests: tokenizer, constraints, scoring (GA PV-contrast reuse), and a full
Phase-2 dry run (`test_pipeline_dryrun.py`, also asserts plots are written).

## 1. Phase 1 — initial SFT

Next-token SFT, conditioned (hindsight) on each design's own achieved
PV-score vector.

```bash
$PY scripts/train_sft.py --config configs/default.yaml \
    --steps 2000 --out ./runs/sft.pt
```

- No `data.sims_dirs` set → bootstraps from `data.synthetic_n` synthetic
  designs labeled by the surrogate. Point `data.sims_dirs` at real sims dirs
  (`design_*/worker_*/*/`) for real hindsight data.
- Output checkpoint: `./runs/sft.pt`.

### Per-state best + best-overall tracker

`train_sft.py` keeps a `TrainScoreTracker` (`src/scoring.py`) that ingests
each batch's just-sliced K-state S21 and tracks two things across the whole
training run:

- **`per_state_best`** — best value of each PV scalar in each state,
  *independent* across states (`pv_pb_level` ↑, `pv_contrast` ↑,
  `pv_pb_std` ↓). Captures the envelope of what the model has been
  conditioned on.
- **`best_overall`** — highest GA `design_score` (= −soft-max cost) across
  all examples seen, with the step at which it landed and that design's
  per-state score matrix.

The current snapshot is printed alongside the loss line every print interval:

```
step   100 | loss 1.3753 | loss_pixel=0.96 loss_value=6.71 ...
           best_overall=-0.182@step83 | per_state_best: S0(-2.40,+24.10,+1.05) | S1(-3.10,+19.40,+1.30) | S2(-1.80,+27.60,+0.84)
```

The final snapshot is also saved inside the checkpoint under the `"tracker"`
key — useful for setting `spec.cold_start_target` in Phase 2:

```python
ckpt = torch.load("runs/sft.pt", weights_only=False)
ckpt["tracker"]   # {n_examples, scalars, per_state_best, best_overall,
                  #  best_overall_step, best_overall_state_matrix}
```

Implementation: `src/scoring.py:TrainScoreTracker`; computed from the
**same** K-state slices the model is conditioned on (slicing done explicitly
in `train_sft.py`, `state_rng=None` passed to `collate`).

### Dataset cache (skip the slow per-design parse on re-runs)

Parsing 20k+ designs + their `.s2p` files takes several minutes. Set
`data.cache_path` to pickle the parsed `LMDataset` on the first run; later
runs (any script, any config that matches) reload it in seconds.

```yaml
data:
  cache_path: ./runs/dataset_cache.pkl   # project-local, relative to CWD
  cache_refresh: false                   # set true to force a rebuild
```

The cache (format_version=2) stores **all `K_avail` states** per design.
Per-`K` state resampling happens at minibatch collate time, not at
build/cache time -- see "State diversity (10 → 3 resampled per minibatch)"
below.

The cache header (`sims_dirs`, `freq_points` hash via SHA-1, `limit`,
`min_states`, `format_version`) is checked at load -- any change invalidates
the cache automatically. Implementation: `src/dataset.py:build_from_sims_cached`.

### State diversity (10 → 3 resampled per minibatch)

The raw world-model sims have 10 switch states per design; the LM uses
`grid.n_states = K = 3`. To expose more of the 10-state combinatorics during
SFT, every minibatch draws **fresh K state indices per example** at collate
time, sliced consistently across `switch_states`,
`inductor_switch_states`, and the per-state S21 curve.

- The dataset (and cache) retain ALL `K_avail` states per design (no
  build-time slice). Designs with `K_avail < K` are skipped.
- `train_sft.py` creates one `state_rng = np.random.default_rng(
  data.state_select_seed)` and passes it to every `collate(...)` call.
- Implementation: `src/training.py:_slice_example_for_k` +
  `collate(..., state_rng=...)`.
- Same design + same minibatch index = different state-triple every step
  → effectively `C(10, 3) = 120` distinct (state-triple, S21-triple,
  hindsight-score) views of each layout.
- Phase-2 (`train_iterated.py`) generated pool examples already have K=3
  states, so the slice helper is a no-op for them.

## 2. Phase 2 — iterated surrogate-guided SFT

```bash
$PY scripts/train_iterated.py --config configs/default.yaml \
    --sft-ckpt ./runs/sft.pt --iterations 30
```

- `--sft-ckpt` is optional (omit to start from a fresh model).
- `--smoke` runs one tiny iteration end-to-end (pipeline check):
  ```bash
  $PY scripts/train_iterated.py --config configs/default.yaml --smoke
  ```
- Per-iteration metrics print to stdout (GA scores, calibration, eval,
  `REWARD_HACK_WARNING`).
- **Plots** (`viz.enabled: true`) are written to
  `<logging.dir>/plots/iter_NNNNN_best_gen.png` (best generated design +
  world-model S21 + scores) and `_best_sim.png` (best simulated design at
  recalibration). Default `logging.dir: ./runs`.

## 3. Standalone evaluation

```bash
$PY scripts/eval.py --config configs/default.yaml --ckpt ./runs/sft.pt \
    --n-targets 8
```

Prints best-of-N GA score, valid fraction, and pool diversity on cold-start
target score vectors.

## Config knobs (`configs/default.yaml`)

- `surrogate.checkpoint_path` — the trained **3-channel** ensemble
  (`/disk/mcao53/Sim_engine_results/wm_train_combined_dataset/ensemble_model.pt`).
  `null` ⇒ untrained fallback (smoke only). Do **not** use
  `wm_trained/ensemble_model.pt` (old 2-channel, incompatible).
- `grid.rows/cols` = 21×27 (must match the trained surrogate's dataset).
- `grid.n_states` (K), `spec.score_scalars`, `spec.target_margins`,
  `spec.bounds` — conditioning / target ratchet.
- `scoring.filter_specs_config` — GA `filter_specs` source
  (`finetune_3_states.yaml`); `scoring.aggregation_mode: soft_max`.
- `iterated.*` — pool size, steps, recal cadence, CFG scale (see
  `architecture.md` → Crucial parameters).
- `viz.enabled` — toggle plot output.

## 4. End-to-end test: scores → design → predicted + simulated S21

[`scripts/test_generation.py`](../scripts/test_generation.py) is a single
entry point that:

1. Takes a **raw** PV-score vector you type on the command line.
2. Normalizes it (see "Score normalization" below) and feeds it to the LM as
   a continuous prefix.
3. Generates one design with constrained decoding.
4. Dumps the **raw generated token stream** (the human-readable "prompt"
   record) to `prompt.txt`.
5. Writes the engine-ready `design.yaml` (with `_base_:` pointing at the same
   engine config the training data was simulated under) and prints a 40-line
   preview to stdout.
6. Predicts S21 with the world model.
7. Optionally calls the **real** EM simulator (`ParallelVirtuosoEngine`)
   and reads back the per-state S21.
8. Saves a side-by-side PNG: layout + world-model curve (solid) overlayed on
   the real-sim curve (dashed) + per-state requested / predicted / simulated
   PV scores.

### How to run

```bash
$PY scripts/test_generation.py \
    --config configs/default.yaml \
    --checkpoint runs/sft.pt \
    --scores=-5,20,2,-7,18,3,-6,22,2 \
    --output runs/test_generation/run_001
```

> **Note on `--scores=...`.** Use `=` (no space) so that argparse doesn't
> interpret the leading negative as a flag. The value is `K*n_scalars`
> comma-separated raw dB floats in `(state, scalar)` row-major order. For
> `K=3` and `spec.score_scalars: [pv_pb_level, pv_contrast, pv_pb_std]`
> that's 9 numbers: `pb_level_S0, contrast_S0, std_S0, pb_level_S1, …`.

Flags:

| Flag | Default | Effect |
|---|---|---|
| `--scores=<csv>` | required | raw dB targets, `K*n_scalars` values |
| `--output <dir>` | `runs/test_generation/run` | output directory |
| `--checkpoint <path>` | `runs/sft.pt` | LM checkpoint; random init if missing |
| `--data-collection-config <path>` | `world_model_test/wm_data_collection.yaml` | resolved → engine config used as the design YAML's `_base_:` |
| `--base-sim-config <path>` | — | explicit engine config (overrides the data-collection resolution) |
| `--no-simulate` | off | skip the real EM sim (surrogate only) |
| `--temperature <float>` | from `iterated.temperature` | sampling temperature |
| `--cfg-scale <float>` | from `iterated.cfg_scale` | classifier-free guidance |
| `--seed <int>` | `0` | torch + numpy seeds |

### Score normalization

The LM is conditioned on a **continuous** per-state PV-score vector — there
are no score *tokens*. The raw dB numbers you pass on the CLI go through
`ScoreNorm` ([src/scoring.py:107-127](../src/scoring.py#L107-L127)):

```
normalized[k, j] = (raw[k, j] - center[j]) / scale[j]
```

with `(center, scale)` per scalar set in `spec.normalization` of
[configs/default.yaml](../configs/default.yaml):

| scalar | center | scale | rationale |
|---|---|---|---|
| `pv_pb_level` | −10 dB | 10 dB | typical lossy-passband range |
| `pv_contrast` | 15 dB | 20 dB | typical pb − sb gap |
| `pv_pb_std` | 4 dB | 4 dB | typical ripple |

The normalized `(K, n_scalars)` matrix is run through the model's
`spec_mlp` (`Linear(n_scalars → d) → GELU → Linear(d → d)`) into K
`d`-dimensional embeddings, prepended as a causal prefix
([src/model.py:124-157](../src/model.py#L124-L157)). At training time
([src/training.py:44-54](../src/training.py#L44-L54)) the same `ScoreNorm`
is applied to each example's *achieved* PV scores (hindsight) before the
prefix, so the model learns the inverse map "normalized PV vector → design".

This means you should always pass scores in the same *raw dB* units the
config's `(center, scale)` were tuned for. Out-of-range values are not
clipped — they pass through the affine transform and through `spec_mlp`,
but the model's training distribution gets thinner the further from the
chosen centers you go.

### Outputs (under `--output <dir>/`)

| File | Contents |
|---|---|
| `prompt.txt` | raw scores → normalized scores → full generated token stream (`idx:NAME`, segment-grouped) |
| `design.json` | `Design.to_record()` — directly consumable by `world_model_test/encode.py` |
| `design.yaml` | `_base_: <engine_config>` + `mode: pixelated_design_simple` + `pixelated:` overlay; drop-in for `ParallelVirtuosoEngine` |
| `scores.json` | `{scalars, requested, predicted[, simulated]}` matrices |
| `test_generation.png` | 3 panels when sim available: [grid \| world-model S21 \| real-sim S21] (matched y-limits); 2 panels [grid \| world-model S21] otherwise. Footer = req / pred / sim PV scores per state |
| `wm_prediction.png` | Standalone world-model S21 plot (one axes, per-state lines, spec shading) — always written |
| `engine.log` | engine stdout/stderr (only when not `--no-simulate`) |
| `sim_workspace/` | engine per-state output directory (only when not `--no-simulate`) |

### Engine-config linkage

The default flow reads
`/disk/mcao53/RFIC_RL/world_model_test/wm_data_collection.yaml`, walks its
`_base_:` chain to find the GA config's `virtuoso_config:` field, and uses
that engine YAML (e.g. `virtuoso_engine/config_ps31_gs25.yaml`) as the
`_base_:` of the generated `design.yaml`. That guarantees the LM design
inherits the same `layout`, `pixelated.feedline`, `inductor_translation_table`,
`inductor_s1p`, EMX/Spectre frequency sweep, etc. as the sims that produced
the training data — no parameter drift. Override with `--base-sim-config`
to point at a different engine config.

## Real EM simulator (status)

`scripts/test_generation.py` wires `ParallelVirtuosoEngine` directly (same
pattern as `ga_optimizer.py:1104`). For Phase-2 batch recalibration
(`src/simulator.py:run_simulator`) the path still raises `NotImplementedError`
— keep `iterated.use_surrogate_sim: true` (surrogate + noise stand-in) until
that batch path is wired.
