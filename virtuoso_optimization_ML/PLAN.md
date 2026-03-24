# ML-Based Optimization for Pixelated RFIC Filter Design

## Context

The project already has SA and GA optimizers for a pixelated switchable RF filter design space. This plan implements the third optimizer: an online ML framework described in `virtuoso_optimization_ML/online_training.md`. The goal is to train a conditional flow matching generative model (conditioned on target S-parameters) alongside an ensemble world model (surrogate simulator), so that after a warm-up phase, new high-quality designs can be generated at low simulation cost via active learning.

---

## File Structure

```
virtuoso_optimization_ML/
├── __init__.py
├── ml_config.yaml
├── run_ml.py
├── encoding/
│   ├── __init__.py
│   └── design_encoder.py        # DesignState ↔ image tensor
├── dataset/
│   ├── __init__.py
│   └── replay_buffer.py         # Priority replay buffer
├── models/
│   ├── __init__.py
│   ├── s_param_encoder.py       # 1D CNN encoder for S-param conditioning
│   ├── flow_matching.py         # Conditional flow matching generative model
│   └── world_model.py           # Ensemble CNN surrogate
├── training/
│   ├── __init__.py
│   ├── warmup.py                # Step 0: random designs + bulk simulation
│   └── online_trainer.py        # Main 5-step training loop
└── utils/
    ├── __init__.py
    └── s_param_utils.py         # Touchstone loading + interpolation helpers
```

---

## Implementation Order (dependency-safe)

1. `utils/s_param_utils.py`
2. `encoding/design_encoder.py`
3. `dataset/replay_buffer.py`
4. `models/s_param_encoder.py`
5. `models/world_model.py`
6. `models/flow_matching.py`
7. `training/warmup.py`
8. `training/online_trainer.py`
9. `ml_config.yaml` + `run_ml.py`

---

## Module Details

### 1. `utils/s_param_utils.py`

- `load_and_interpolate_s21(s2p_path, freq_points) → Tensor(n_freq)` — load skrf Network, interpolate S21 dB to canonical grid
- `result_to_s_param_tensor(result: RunResult, freq_points, n_states) → Tensor(n_states, n_freq) | None` — parse RunResult.s_param_files list
- `build_target_s_param_tensor(filter_specs, freq_points) → Tensor(n_states, n_freq)` — construct ideal target curve from FilterStateSpec objects (passband / stopband floors/ceilings)

### 2. `encoding/design_encoder.py`

Single-channel image encoding. Image shape: `(1, rows, cols + n_states)`.

**Region map (no overlap, gaps prevent smooth transitions):**

| Pixel type | Float range | Embedded value |
|---|---|---|
| 0 (ground) | [0.00, 0.15] | 0.05 (constant) |
| 1 (metal)  | [0.20, 0.35] | 0.25 (constant) |
| 2 (inductor)| [0.45, 0.60] | 0.45 + idx/(n_ind-1)*0.15 |
| 3 (switch) | [0.70, 0.85] | 0.70 + (cap-cap_min)/(cap_max-cap_min)*0.15 |

Last `n_states` columns: binary {0.0, 1.0} switch states per state (decoder knows these columns by position, not by value range).

**Class `DesignEncoder`:**
- `encode(design: DesignState) → Tensor`
- `decode(image: Tensor) → DesignState` — snap each pixel to nearest region; parse last n_states columns as switch_states; may produce infeasible designs (caller validates)

### 3. `dataset/replay_buffer.py`

**`BufferEntry` dataclass:** `image, s_params, is_real, is_feasible, iteration, confidence, entry_id`

- `is_feasible=False` entries use an **infeasibility sentinel**: `s_params` is set to `torch.ones(n_states, n_freq)` (all +1 in linear scale, impossible for a passive device). This teaches the generative model which image regions are infeasible without any separate loss function.
- Infeasible entries bypass the world model and simulation entirely — they are created directly from the decoded image.

**`ReplayBuffer`:**
- `add(image, s_params, is_real, is_feasible, iteration, confidence) → entry_id`
- `sample_for_generative(batch_size, current_iter)` — priority = `alpha_recency * exp(-decay*(iter-entry.iter)) + alpha_real * (real_bonus if is_real else 0)`. Infeasible entries get a fractional weight `infeasible_weight` (e.g., 0.1×) so they don't dominate training but still provide gradient signal.
- `sample_for_world_model(batch_size)` — uniform over real-only AND feasible entries only
- `sample_high_confidence_real(n, threshold)` — for step 5 correction (feasible + real only)
- `update_entry(entry_id, s_params, is_real, confidence)` — overwrite label after real simulation
- `save/load` — pickle for checkpointing

### 4. `models/s_param_encoder.py`

**`SParamEncoder(n_states, n_freq, cond_dim)`:**
- Input: `(B, n_states, n_freq)` → reshape to `(B*n_states, 1, n_freq)`
- 3× Conv1d layers (1→32→64→128, kernels 7/5/3), AdaptiveAvgPool1d(32)
- Reshape to `(B, n_states*128*32)`, MLP → `(B, cond_dim)`

### 5. `models/world_model.py`

**`WorldModelMember(img_h, img_w_total, n_states, n_freq)`:**
- Conv2d(1,64,3) → GroupNorm → GELU
- 3× ResBlock with stride=2 (64→128→256→256)
- AdaptiveAvgPool2d(4,4) → flatten → Linear(256*16, 512) → GELU → Linear(512, n_states*n_freq)

**`EnsembleWorldModel(n_members, ...)`:**
- `predict(x) → (mean_pred, std_pred)` both `(B, n_states, n_freq)`
- `confidence(x) → (mean_pred, conf_score (B,))` — `conf = exp(-mean(std)/std_scale)`
- `train_step(x, s_params_true) → loss` — independent MSE per member; optional bagging (each member trains on random 80% subset)

### 6. `models/flow_matching.py`

**Formulation (linear interpolation path):**
- `x_t = (1-t)*x_0 + t*x_1`, target `u_t = x_1 - x_0`
- Loss: `MSE(v_θ(t, x_t, c), u_t)`
- Classifier-free guidance (CFG): 10% dropout to learned null embedding during training; at inference `v_guided = v_uncond + scale*(v_cond - v_uncond)`
- Inference: Euler integration, `n_steps=50` by default

**`ConditionalVectorField(img_h, img_w, cond_dim, t_dim, base_channels)`:**
- U-Net with 3 down/up stages (channels 64→128→256), self-attention at bottleneck
- Time: SinusoidalPosEmb → Linear → SiLU → Linear → `(B, t_dim)`
- Conditioning injected via AdaGN (Adaptive Group Norm: scale+shift from `Linear(t_dim+cond_dim, 2*C)`) at each ResBlock

**`FlowMatchingModel`:**
- `train_step(x_1, target_s, null_emb) → loss`
- `sample(target_s, n_steps, guidance_scale) → (B, 1, H, W+S)`

### 7. `training/warmup.py`

`run_warmup(cfg, encoder, buffer, freq_points, rng)`:
- Calls `generate_random_design()` (from SA feasibility) ~100 times
- Builds config overrides via `design.to_pixelated_config()` (same pattern as SA `_build_override`)
- Runs `ParallelVirtuosoEngine.run_batch()` in batches of `n_parallel_sim`
- Parses results via `result_to_s_param_tensor()`, adds to buffer as `is_real=True`

### 8. `training/online_trainer.py`

**`OnlineMLTrainer(config_path)`** — main orchestrator:

```
for iteration in range(n_iterations):
    Step 1: sample n_gen_samples designs from FlowMatchingModel
            → decode → feasibility check
            → try up to max_resample_attempts to get feasible designs
            → any still-infeasible designs are labeled with sentinel
              s_params (all-ones) and added to buffer with is_feasible=False
            → infeasible designs skip steps 2a and simulation
    Step 2: EnsembleWorldModel.confidence() on each
            → high-conf → add to buffer with world model label
            → low-conf (up to n_sim_max) → simulate → add as real
    Step 3: train FlowMatchingModel (gen_train_steps Adam steps)
            using priority-sampled buffer entries
    Step 4: train EnsembleWorldModel (wm_train_steps Adam steps)
            using real-only buffer entries
    if iteration % wm_correction_interval == 0:
        Step 5: sample_high_confidence_real → simulate → update_entry
                → fine-tune world model on corrected entries
    if iteration % checkpoint_interval == 0:
        save models + buffer
```

Logging via TensorBoard: gen_loss, wm_loss, n_real_sims, mean_confidence, best_cost_so_far.

### 9. `ml_config.yaml`

Key parameters (mirrors SA/GA config structure):
```yaml
grid: {rows: 25, cols: 27, n_switches: 4, max_inductors: 10, n_states: 2, ...}
filter_specs: [...]          # same format as SA/GA
s_param: {freq_min: 1e9, freq_max: 50e9, n_freq: 201}
generative_model: {cond_dim: 256, t_dim: 128, base_channels: 64, n_steps: 50, guidance_scale: 2.0, ...}
world_model: {n_members: 5, std_scale: 3.0, confidence_threshold: 0.7, correction_interval: 10, ...}
replay_buffer: {max_size: 10000, alpha_recency: 0.6, alpha_real: 0.4, real_bonus: 2.0, ...}
training: {warmup_n_designs: 100, n_iterations: 500, n_gen_samples: 20, n_sim_max: 8,
           max_resample_attempts: 5, infeasible_weight: 0.1, ...}
```

---

## SA/GA Code Reused (imports only, not copied)

| What | From |
|---|---|
| `DesignState` | `virtuoso_optimization_SA.design` |
| `generate_random_design`, `is_feasible` | `virtuoso_optimization_SA.feasibility` |
| `FilterStateSpec`, `specs_from_config`, `compute_total_cost` | `virtuoso_optimization_SA.cost` |
| `random_action` | `virtuoso_optimization_SA.actions` (optional: fixup infeasible decoded designs) |
| `ParallelVirtuosoEngine` | `virtuoso_engine.parallel_engine` |
| `RunResult` | `virtuoso_engine.engine` |

The `_build_override` method follows the exact same pattern as `sa_optimizer.py`: `design.to_pixelated_config()` wrapped in the standard config override dict.

---

## Verification

1. **Encoding round-trip:** `encode(decode(encode(design)))` should recover the same image within floating-point tolerance on a valid DesignState.
2. **Warmup:** After warmup, buffer contains ~100 real entries with valid `(1, rows, cols+n_states)` images and `(n_states, n_freq)` S-param tensors.
3. **World model:** After training on warmup data for 100 steps, training loss should decrease; ensemble std should be nonzero (members diverging).
4. **Flow matching:** `train_step` loss should decrease over 200 steps on warmup data; `sample()` should return tensors of the correct shape; decoded designs should sometimes pass `is_feasible()`. After many iterations the infeasible-to-feasible ratio in generated samples should improve.
5. **Full loop:** Run `python run_ml.py ml_config.yaml` — warmup completes, iteration 0 runs, TensorBoard logs appear in `output_dir/tb_logs/`.
