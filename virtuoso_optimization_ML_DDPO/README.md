# virtuoso_optimization_ML_DDPO

DDPM + DDPO online optimizer for the pixelated switchable RF filter design problem.

This module is a variant of `virtuoso_optimization_ML` that replaces the
conditional flow-matching generative model with a **DDPM diffusion model**,
then adds a **DDPO (Denoising Diffusion Policy Optimisation) fine-tuning step**
that uses the ensemble world model as a reward signal.

---

## Algorithm overview

```
Step 0  Warmup — simulate ~200 random feasible designs; populate replay buffer.

Pre-train world model (20 steps) and diffusion model (50 steps) on warmup data.

Repeat for n_iterations:
  Step 1  Sample n_gen_samples designs via DDIM from the diffusion model.
          Feasibility-check; infeasible designs added to buffer with sentinel label.

  Step 2  World model confidence split on feasible designs:
            high confidence  → relabel with WM prediction, add to buffer (no sim)
            low  confidence  → simulate with Virtuoso, add as real data

  Step 3  Train diffusion model (gen_train_steps Adam steps, priority-sampled buffer).

  Step 4  Train world model (wm_train_steps Adam steps, real-only buffer entries).

  Step 5  (every wm_correction_interval iters) Re-simulate high-confidence WM entries;
          correct buffer labels; fine-tune world model.

  DDPO    (every ddpo_interval iters) Fine-tune diffusion model with RL:
            - Sample ddpo_batch_size trajectories via stochastic DDPM (Markovian).
            - Decode x_0 → reward:
                infeasible  → R = −penalty × n_violations (proportional excess included)
                feasible + confident WM → R = −cost
                feasible + uncertain WM → masked (no gradient)
            - If all masked: skip update.
            - Normalise rewards; REINFORCE gradient update on diffusion model.
```

---

## Diffusion model and sampling schemes

The generative model is a **DDPM (Ho et al. 2020)** with T=1000 linear beta schedule:

```
Forward:   x_t = sqrt(ā_t)·x_0 + sqrt(1−ā_t)·ε,    ε ~ N(0,I)
Training:  L = E[ ||ε_θ(x_t, t, c) − ε||² ]
```

The network (a U-Net with AdaGN conditioning) predicts the noise ε at any
t ∈ {0,...,999} given a conditioning vector c from the S-parameter encoder.

### Why you can sample in fewer than 1000 steps

Two different reverse-process schemes are used, serving different purposes:

| Scheme | Markovian? | log p computable? | DDPO valid? | Quality (few steps) |
|--------|-----------|-------------------|-------------|---------------------|
| **DDIM** (eta=0, Song 2020) | No — non-Markovian ODE | No — sigma=0, delta fn | **No** | High |
| **Stochastic DDIM** (0 < eta ≤ 1, strided) | **Yes** | Yes — Gaussian at each step | **Yes** | Medium–High |
| Full DDPM (eta=1, stride=1) | Yes | Yes | Yes | High, but needs T=1000 steps |
| Full DDPM (eta=1, strided 20 steps) | Yes | Yes | Yes | Lower — large σ per step |

**DDIM** (`sample()`) — used for Step 1 and inference:

Rewrites the reverse process as a deterministic ODE; the same trained network
can skip to any subset of timesteps without retraining:

```
x_{t'} = sqrt(ā_{t'})·x_0_pred + sqrt(1−ā_{t'})·ε_θ
```

High quality at 50 steps, but sigma=0 → log-probability is undefined
→ **cannot be used for DDPO**.

**Stochastic strided DDPM** (`ddpo_sample()`) — used for DDPO:

Uses the generalised DDIM formula with eta > 0:

```
β_eff  = 1 − ā_t / ā_{t'}
σ_t²   = eta² · (1−ā_{t'})/(1−ā_t) · β_eff
μ_t    = sqrt(ā_{t'})·x_0_pred + sqrt(1−ā_{t'}−σ_t²)·ε_θ
x_{t'} = μ_t + σ_t · z,    z ~ N(0,I)
```

This IS a valid Markov chain with Gaussian transitions → log-probability is
well-defined → DDPO is valid. The `sigma_sq` value is stored in each trajectory
step dict and consumed directly by `ddpo_loss` for the log-prob computation.

### The `eta` parameter

The consistency between DDIM samples (Step 1) and DDPO samples depends on `eta`:

| eta | Behaviour |
|-----|-----------|
| 1.0 | Full DDPM — maximum noise per step; DDPO trajectories are very different from DDIM at few steps |
| 0.7 | **Recommended default** — 0.49× variance; samples close to DDIM, log-probs well-conditioned |
| 0.5 | Very close to DDIM; use if DDPO is unstable early in training |
| 0.0 | Becomes DDIM — sigma=0 → log-prob degenerates → DDPO loss is undefined / explodes |

Set via `ml_config.yaml → ddpo → eta`.

---

## Reward function

For each DDPO trajectory sample:

**Infeasible design** (always contributes):
```
R = −infeasible_penalty × ( binary_violations + switch_excess + inductor_excess )
```
- `binary_violations`: 1 per failed rule (feed columns, connectivity,
  switch spacing, inductor spacing, distinct switch states, switch count too low)
- `switch_excess`: max(0, n_switches − n_switches_max)  — proportional
- `inductor_excess`: max(0, n_inductors − max_inductors)  — proportional

**Feasible + world model confident** (contributes):
```
R = −cost        (lower cost = higher reward)
```

**Feasible + world model uncertain** (masked — no gradient):
No update; the diffusion model is not penalised for uncertain designs.

If **all** designs in a DDPO batch are masked (uncertain), the update is skipped entirely.

Rewards are **normalised** (subtract mean, divide by std) over contributing designs
before computing the REINFORCE gradient.

---

## File structure

```
virtuoso_optimization_ML_DDPO/
├── README.md                     ← this file
├── DDPO.md                       ← original design spec
├── ml_config.yaml                ← all hyperparameters
├── run_ml.py                     ← entry point
├── encoding/
│   └── design_encoder.py         ← DesignState ↔ image tensor
├── dataset/
│   └── replay_buffer.py          ← priority replay buffer
├── models/
│   ├── s_param_encoder.py        ← 1D CNN encoder for S-param conditioning
│   ├── diffusion_model.py        ← DDPM + DDIM sampling + DDPO trajectory/loss
│   └── world_model.py            ← ensemble CNN surrogate
├── training/
│   ├── warmup.py                 ← Step 0: random designs + bulk simulation
│   └── online_trainer.py         ← main training loop (Steps 1–5 + DDPO)
└── utils/
    ├── s_param_utils.py
    ├── cost_utils.py
    └── plot_utils.py
```

---

## Key hyperparameters (`ml_config.yaml`)

```yaml
generative_model:
  n_ddim_steps:   50      # DDIM steps for Step 1 sampling
  guidance_scale: 2.5     # classifier-free guidance scale
  T:              1000    # diffusion timesteps (training)

ddpo:
  n_steps:            50      # stochastic DDPM steps per DDPO trajectory
  batch_size:         32      # designs sampled per DDPO step
  interval:           1       # run DDPO every N training iterations
  infeasible_penalty: 1.0     # base penalty multiplier per rule violation
  eta:                0.7     # stochasticity: 0.0=DDIM (invalid), 1.0=full DDPM

world_model:
  confidence_threshold: 0.75  # min confidence to use WM reward without simulation
```

---

## Usage

```bash
# First run (warmup + training)
python run_ml.py ml_config.yaml

# Resume from checkpoint
python run_ml.py ml_config.yaml --resume /path/to/checkpoint/iterXXXXX
```

TensorBoard logs are written to `output_dir/tb_logs/`. Key metrics to monitor:

| Metric | Meaning |
|--------|---------|
| `loss/gen_model_step` | Diffusion model denoising loss (should decrease) |
| `loss/world_model_step` | World model MSE loss (should decrease) |
| `loss/ddpo` | DDPO policy gradient loss |
| `ddpo/reward_mean` | Mean raw reward over active DDPO samples |
| `ddpo/n_active` | Number of designs contributing to DDPO gradient |
| `ddpo/n_infeasible` | Infeasible designs in DDPO batch |
| `ml/n_sim` | Real Virtuoso simulations per iteration |
