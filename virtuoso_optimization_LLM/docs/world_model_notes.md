# World Model Notes

Source: `/disk/mcao53/RFIC_RL/world_model_test/`. This is the pretrained
surrogate the LM project depends on. Everything below was verified by reading
the source, not inferred.

## TL;DR

- Input: a **3-channel image** `(3, rows, cols)` float32 (default 21×27).
- Output: **S21 magnitude in dB**, shape `(B, n_freq)`, `n_freq=71`, 0–70 GHz.
- It is **single-state**: one switch/inductor ON-OFF configuration per call.
  A K-state filter requires **K calls** (one per `state_idx`).
- Architecture: deterministic CNN (ResNet-ish) + 5-member deep **ensemble**
  giving epistemic variance.
- Reuse `encode.py` and `model.py` directly — do not re-implement them.

## Input format — `encode.py`

`encode_sample(record, state_idx, cfg) -> np.ndarray (3, rows, cols)` float32
(`world_model_test/encode.py:179`). Channels:

- **Ch 0** (`make_pixel_channel`, `:75`): pixel type + continuous value, static
  across states. Ground→0.05, Metal→0.20, Inductor→[0.30,0.50] (encodes
  `inductor_values` index), Switch→[0.60,0.90] (encodes `cap_values` normalized
  over `[cap_min,cap_max]`), Via→0.95.
- **Ch 1** (`make_state_channel`, `:114`): ON/OFF map for `state_idx`. 1.0 at a
  switch/inductor pixel that is ON in that state, else 0.0. **This is the only
  channel that changes between states.**
- **Ch 2** (`make_value_channel`, `:146`): secondary values, static. Inductor
  pixels → normalized `inductor_cap_values`; cap-pair endpoint pixels →
  normalized `cap_pair_values`; else 0.0.

### The `record` dict (the contract the LM must emit)

Keys consumed by `encode.py` (cross-checked against
`process_dataset.py:162`, which builds records from each design's
`config.yaml` `pixelated:` block):

| key | type | notes |
|---|---|---|
| `matrix` | `List[List[int]]` rows×cols | pixel codes only **0–4** (ground/metal/inductor/switch/via). No cap-pair code. |
| `cap_values` | `List[float]` | **flat**, one per switch pixel, row-major. Farads. |
| `inductor_values` | `List[int]` | one per inductor pixel, index into translation table. |
| `inductor_cap_values` | `List[float]` | one per inductor pixel. Farads. |
| `cap_pixel_pairs` | `List[[[r1,c1],[r2,c2]]]` | metal-pixel coordinate pairs. |
| `cap_pair_values` | `List[float]` | one per pair. Farads. |
| `switch_states` | `List[List[int]]` `(n_states, n_switches)` | per-state ON/OFF. |
| `inductor_switch_states` | `List[List[int]]` `(n_states, n_inductors)` | per-state ON/OFF; `[]` ⇒ all inductors ON. |

Note `cap_values` is **flat per switch** here (bank size 1). The SA
`DesignState` uses a banked `[n_switches][n_caps_in_bank]` shape — our `Design`
targets the flat `encode.py` contract; banked switches are a `TODO(user)`.

`EncodeConfig` (value ranges) is built by
`build_encode_config(config) -> EncodeConfig` (`encode.py:48`) from a merged
YAML: `rows, cols, cap_min, cap_max, n_inductor_values, inductor_cap_min/max,
cap_pair_min/max`. Reuse this; do not hardcode ranges.

## Output format

`(B, n_freq)` = predicted **S21 magnitude in dB** (not normalized, trained with
raw-dB MSE). Default frequency grid: `np.linspace(0, 70e9, 71)` → 71 points,
1 GHz spacing. Configurable via processing args; the trained checkpoint stores
`freq_min`/`freq_max`/`n_freq`.

S21 ground truth in the dataset comes from `.s2p` Touchstone files; loaded via
`skrf.Network`, `s21_db = ntwk.s_db[:, 1, 0]`, interpolated onto the uniform
grid (`world_model_test/wm_dataset.py`).

## Single-state confirmation

`make_state_channel(record, state_idx, cfg)` takes a scalar `state_idx` and
emits exactly one ON/OFF map. Ch 0 and Ch 2 are state-independent. To evaluate
a K-state filter you call `encode_sample` K times (`state_idx = 0..K-1`) and run
the model K times. The training dataset reflects this: N designs × n_states
(design, state) samples.

## Model + loading — `model.py`

- `WorldModel(n_freq)` (`model.py:55`): stem → 3 ResBlocks → AdaptiveAvgPool2d(4)
  → MLP(4096→512) → Linear(512→n_freq). `forward(x:(B,3,H,W)) -> (B,n_freq)`.
  `AdaptiveAvgPool2d` makes it input-size agnostic (configurable grid works).
- `EnsembleModel(members, std_scale=1.0)` (`model.py:124`):
  - `predict(x) -> (mean, epistemic_variance)`, each `(B, n_freq)`
    (`forward`/`predict`, `:146`/`:161`, both `@torch.no_grad`).
  - `confidence(x) -> (mean, conf_score)` where
    `conf_score = exp(-mean(epistemic_std)/std_scale) ∈ [0,1]` (`:166`).
  - `n_members` property.

### Loading a trained checkpoint

Checkpoint written by `train.py:323` to `<out_dir>/ensemble_model.pt`:

```python
{
  "member_states":   [state_dict, ...],   # one per member
  "member_val_maes": [float, ...],
  "n_freq":          int,
  "freq_min":        float,
  "freq_max":        float,
  "processed_dir":   str,
  "ensemble_size":   int,
}
```

Load (per `world_model_test/README.md`):

```python
import sys; sys.path.insert(0, ".../world_model_test")
import torch
from model import WorldModel, EnsembleModel

ckpt = torch.load(path, map_location="cpu", weights_only=False)
members = []
for sd in ckpt["member_states"]:
    m = WorldModel(n_freq=ckpt["n_freq"]); m.load_state_dict(sd); members.append(m)
ensemble = EnsembleModel(members).eval()
mean, var = ensemble.predict(x)   # x: (B,3,H,W) -> (B,71),(B,71)
```

Device: `ensemble.to(device)`; the model auto-uses CUDA in `train.py` but
inference works on CPU. Batch size is arbitrary (standard PyTorch).

**No `ensemble_model.pt` exists inside the repo** — `train.py` writes it to an
external `--output-dir` (e.g. under `/disk/mcao53/Sim_engine_results/...`). The
surrogate wrapper takes the checkpoint path from config (`TODO(user)`) and
falls back to freshly-initialized untrained members for the smoke test.

## Normalization summary

- Inputs are min/max normalized inside `encode.py` (no learned input embedding,
  no external scaler). The LM only needs to produce the `record` dict; encoding
  is the world model's job.
- Output S21 is raw dB. The surrogate `score()` operates in dB directly.
