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

### Dataset cache (skip the slow per-design parse on re-runs)

Parsing 20k+ designs + their `.s2p` files takes several minutes. Set
`data.cache_path` to pickle the parsed `LMDataset` on the first run; later
runs (any script, any config that matches) reload it in seconds.

```yaml
data:
  cache_path: /disk/mcao53/Sim_engine_results/lm_dataset_cache_21x27.pkl
  cache_refresh: false   # set true to force a rebuild
```

The cache header (`sims_dirs`, `freq_points` hash, `limit`, `n_states`,
`seed`) is checked at load — any change invalidates the cache automatically.
Implementation: `src/dataset.py:build_from_sims_cached`.

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

`scripts/test_generation.py` takes target PV scores, generates a design with
the LM, writes it to (a) `design.json` (world-model record format) and (b)
`design.yaml` (simulation-engine override; `_base_:` inherits from the engine
config), runs the world model, optionally runs the **real** EM simulator, and
saves a comparison PNG.

```bash
$PY scripts/test_generation.py \
    --config configs/default.yaml \
    --checkpoint runs/sft.pt \
    --scores=-5,20,2,-7,18,3,-6,22,2 \
    --output runs/test_generation/run_001
```

Use `--scores=...` (with `=`) so argparse doesn't interpret the leading
negative as a flag. Add `--no-simulate` to skip the EM step.

**Engine config linkage.** By default the script reads the data-collection
config (`world_model_test/wm_data_collection.yaml`), follows its `_base_:`
chain + `virtuoso_config:` field to find the engine config (e.g.
`virtuoso_engine/config_ps31_gs25.yaml`), and uses that as the LM design
YAML's `_base_:`. This guarantees the generated YAML inherits **exactly the
same engine parameters** (pixel size, inductor table, EMX/Spectre settings,
inductor_translation_table, etc.) as the sims that produced the training data.

Override the resolution with `--data-collection-config <path>` or skip it
entirely with `--base-sim-config <engine_config_path>`.

Outputs (`<output_dir>/`):
- `design.json` — `Design.to_record()` (drop-in for the world-model encoder)
- `design.yaml` — `_base_: <engine_config>` + `mode: pixelated_design_simple`
  + `pixelated:` override (drop-in for the engine, GA fine-tune, or any
  pipeline consuming the engine YAML format)
- `scores.json` — `{requested, predicted, simulated}` PV-score matrices
- `test_generation.png` — grid + per-state S21 (predicted solid, simulated
  dashed) + per-state requested/predicted/simulated PV-score readouts
- `engine.log` — simulator stdout/stderr (only when not `--no-simulate`)

## Real EM simulator (status)

`scripts/test_generation.py` wires `ParallelVirtuosoEngine` directly (same
pattern as `ga_optimizer.py:1104`). For Phase-2 batch recalibration
(`src/simulator.py:run_simulator`) the path still raises `NotImplementedError`
— keep `iterated.use_surrogate_sim: true` (surrogate + noise stand-in) until
that batch path is wired.
