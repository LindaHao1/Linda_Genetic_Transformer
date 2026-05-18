# World Model Test

Standalone surrogate model pipeline for predicting RF filter S-parameters without running full EM simulation. Given a pixelated filter design and one switch state, the model predicts the S21 response across frequency.

---

## Overview

The surrogate (world) model is a CNN that learns to map a filter design + switch configuration → S21 magnitude (dB) across 0–70 GHz. It is trained on 10,000 simulated designs × 10 switch states = **100,000 (design, state) samples**.

### Files

| File | Purpose |
|------|---------|
| `encode.py` | Converts a design record into a 2-channel image tensor |
| `wm_dataset.py` | PyTorch Dataset — reads the JSONL index and loads S2P files |
| `model.py` | CNN surrogate model definition |
| `train.py` | Training + evaluation script |
| `wm_data_collection.yaml` | Config used to generate the dataset |
| `data_generation.py` | Script that produced the dataset (already run) |

---

## Design Encoding

Each (design, state) pair is encoded as a **2-channel image of shape `(2, rows, cols)`**:

**Channel 0 — pixel type map** (static, same for all states of a design):

| Pixel type | Value range | Meaning |
|------------|-------------|---------|
| Ground (0) | 0.05 | Metal layer tied to ground |
| Metal (1) | 0.20 | Routing metal |
| Inductor (2) | 0.30 – 0.50 | Inductor pixel; value encodes which inductor model |
| Switch (3) | 0.60 – 0.90 | Switch pixel; value encodes series cap value (100–1000 fF) |

**Channel 1 — state ON/OFF map** (changes per state):

Each pixel is `1.0` if it is a switch or inductor that is **ON** in this specific state, `0.0` otherwise. This preserves spatial structure — the CNN sees both *where* a component is (ch0) and *whether it is active* (ch1) at the same spatial location.

---

## Model Architecture

### Single member (deterministic)

```
Input: (B, 2, H, W)
  ↓ Stem: Conv2d(2→64) + GroupNorm + GELU
  ↓ ResBlock(64→128, stride=2)
  ↓ ResBlock(128→256, stride=2)
  ↓ ResBlock(256→256, stride=1)
  ↓ AdaptiveAvgPool2d(4×4) → Flatten → (B, 4096)
  ↓ Linear(4096→512) + GELU
  ↓ Linear(512→n_freq)
  Output: (B, n_freq)  — predicted S21 in dB
```

Each member is a plain deterministic regression network. `AdaptiveAvgPool2d` makes it work with any grid size.

### Ensemble (epistemic uncertainty)

```
N independent members, each trained with a different random seed
  ↓ stack predictions: (N, B, n_freq)
  ↓ ensemble mean:     (B, n_freq)
  ↓ ensemble variance: (B, n_freq)  ← inter-member disagreement
```

Uncertainty comes from the **disagreement between ensemble members**, not from a per-prediction variance head. This is purely **epistemic** — it reflects how well the training data covers this region of design space, not noise in the simulation (which is deterministic).

---

## Training Details

### Loss Function — MSE

Each ensemble member is trained with plain **mean squared error**:

```
L = mean( (y_pred - y_true)^2 )
```

where `y_true` is the ground-truth S21 vector from EM simulation. There is no stochastic loss term (no learned variance, no NLL). This is the correct choice because **the simulation is fully deterministic** — running the same design and state twice produces the same S21. There is no aleatoric (measurement) noise to model, so a heteroscedastic loss would only add parameters without capturing anything real.

### Epistemic Uncertainty via Deep Ensemble

Uncertainty is estimated not from the loss but from **disagreement between ensemble members**:

```
epistemic_variance(x) = Var_i [ f_i(x) ]   (variance across N member predictions)
```

Each member is initialised with a different random seed, so they find different solutions in weight space. For designs well-covered by training data, all members converge to similar predictions (low variance = high confidence). For out-of-distribution designs, members diverge (high variance = low confidence). This is a principled measure of **epistemic uncertainty** — how much the model doesn't know, which is reducible with more data.

### Optimiser and Schedule

| Setting | Value |
|---------|-------|
| Optimiser | AdamW |
| Weight decay | 1e-4 |
| Initial LR | 3e-4 (default) |
| LR schedule | Cosine annealing, decaying to 0 over all epochs |
| Gradient clipping | Global norm clipped to 1.0 per step |
| Ensemble size | 5 (default) |

Each member trains independently from its own random seed (`base_seed + i`). This is the only source of diversity between members — same architecture, same data, different initialisation and stochastic gradient noise.

### Checkpoint

A single `ensemble_model.pt` is saved after all members finish, containing all member weight dicts and their individual validation MAEs. Per-member training logs are written to `training_log_member{i}.csv`.

Each checkpoint stores the model weights alongside the `EncodeConfig` and frequency grid, so the exact encoding parameters used during training are always recoverable from the checkpoint alone.

---

## How to Run

### Prerequisites

Use the `rfic2` conda environment:

```bash
/disk/mcao53/conda_env/rfic2/bin/python world_model_test/train.py [options]
```

### Quick smoke test (1 epoch, ~2 min)

```bash
/disk/mcao53/conda_env/rfic2/bin/python world_model_test/train.py \
    --epochs 1 --batch-size 32 --num-workers 4 \
    --output-dir /tmp/wm_smoke
```

### Full training run (50 epochs, recommended)

```bash
/disk/mcao53/conda_env/rfic2/bin/python world_model_test/train.py \
    --epochs 50 --batch-size 64 --num-workers 8 \
    --output-dir /disk/mcao53/Sim_engine_results/wm_trained
```

The dataset path and grid config are auto-detected from `wm_data_collection.yaml`. Override with:

```bash
--jsonl /path/to/dataset.jsonl   # explicit JSONL path
--config /path/to/config.yaml    # use a different config
```

### All options

| Argument | Default | Description |
|----------|---------|-------------|
| `--jsonl` | auto from config | Path to `dataset.jsonl` |
| `--config` | `wm_data_collection.yaml` | YAML config (supports `_base_` inheritance) |
| `--epochs` | 50 | Number of training epochs per member |
| `--batch-size` | 64 | Batch size |
| `--lr` | 3e-4 | Initial learning rate (cosine decay to 0) |
| `--val-frac` | 0.1 | Fraction of data held out for validation |
| `--ensemble-size` | 5 | Number of independently-trained ensemble members |
| `--n-freq` | 71 | Number of frequency points (0–70 GHz in 1 GHz steps) |
| `--freq-min` | 0 | Minimum frequency in Hz |
| `--freq-max` | 70e9 | Maximum frequency in Hz |
| `--output-dir` | `/tmp/wm_run` | Where to save checkpoints and plots |
| `--device` | auto | `cuda` or `cpu` |
| `--cache-dir` | none | Cache encoded tensors to disk for faster re-runs |
| `--num-workers` | 4 | DataLoader worker processes |
| `--seed` | 42 | Base seed; member i uses `seed + i` |

---

## Outputs

After training, `--output-dir` contains:

| File | Description |
|------|-------------|
| `ensemble_model.pt` | All member weights + per-member val MAEs |
| `training_log_member{i}.csv` | Per-epoch train/val MSE and MAE for member i |
| `training_curves.png` | Validation loss and MAE for all members (see below) |
| `val_predictions.png` | Ensemble predictions on 6 validation samples (see below) |
| `eval_metrics.json` | Ensemble summary statistics in JSON format |

---

## Plots

### `training_curves.png`

Two side-by-side plots, one line per ensemble member:

- **Left — Validation MSE (dB²)**: The training objective for each member. Lines that converge to similar final values indicate the ensemble members learned consistently; large spread indicates high variance in member quality.
- **Right — Validation S21 MAE (dB)**: Most interpretable accuracy metric. After 50 epochs expect per-member val MAE around 1–2 dB.

Because each member trains independently, plotting all members together gives a visual check on ensemble diversity and stability.

### `val_predictions.png`

Six randomly selected validation samples shown as S21 vs frequency:

- **Blue solid line** — ground truth S21 from EM simulation
- **Red dashed line** — ensemble mean prediction (average of all member outputs)
- **Red shaded region** — **±1σ epistemic uncertainty** (standard deviation across ensemble members)

The shaded band is **not a noise model** — it is the inter-member disagreement at each frequency. A narrow band means all members agree and the model is confident the training data covers this region well. A wide band means members disagree — the design is likely OOD and the prediction should not be trusted without running a real simulation.

Concretely: a design from a well-explored part of the training distribution will have a thin band tightly around the ground truth. A design with an unusual topology or extreme capacitor values that appeared rarely in training will show a wide, spread-out band, correctly flagging it as uncertain.

---

## Using the Model Programmatically

```python
import sys, numpy as np, torch
sys.path.insert(0, 'world_model_test')

from model import EnsembleModel, WorldModel

ckpt = torch.load('path/to/ensemble_model.pt', map_location='cpu', weights_only=False)

members = []
for state in ckpt['member_states']:
    m = WorldModel(n_freq=ckpt['n_freq'])
    m.load_state_dict(state)
    members.append(m)

ensemble = EnsembleModel(members)
ensemble.eval()

# image: (1, 2, rows, cols) float32 tensor built with encode.encode_sample()
mean, epistemic_var = ensemble.predict(image)   # (1, n_freq) each
mean, conf_score    = ensemble.confidence(image) # conf_score in [0, 1]
```

`conf_score` is computed as `exp(-mean(epistemic_std) / std_scale)`:
- Near **1.0** → all members agree → design is well-covered by training data → prediction is reliable.
- Near **0.0** → members strongly disagree → design is likely OOD → run a real simulation instead.
