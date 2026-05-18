# Score / Cost Function Description

Lower cost = better design. The GA minimises total cost across all filter states.

---

## Per-State Cost

Each filter state is evaluated independently against its `FilterStateSpec`. Two modes exist:

### Peak-Valley (PV) Contrast Mode (`enable_pv_contrast: true`)

Used for **lowpass** configurations. Replaces all standard S21 penalty terms with two components:

```
s21_cost = (pv_contrast_weight × (−contrast) + pv_passband_weight × pb_term) × weight
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Contrast term** | `−(pb_avg − worst_k_sb_avg)` | Maximise passband-to-stopband separation. Faster stopband decay → larger contrast → lower cost. |
| **Passband term** | `pb_term = pb_dev ^ pv_passband_order` | Enforce passband insertion loss. |

**Passband deviation `pb_dev`:**
- If `passband_target_db` is set (e.g. `−3.0 dB`):  
  `pb_dev = max(0, passband_target_db − pb_avg)` — **one-sided**: no penalty when passband is better than target; penalty grows as `pv_passband_order` power for shortfalls.
- If not set (legacy):  
  `pb_dev = |pb_avg|` — pushes passband average toward 0 dB from both sides.

**`pv_passband_order` (default 2.0):**  
Quadratic growth means a large insertion-loss shortfall dominates the total cost before the contrast term matters. This enforces passband recovery first.

**`worst_k` for contrast:** `pv_contrast_k` (default 15) — uses the average of the `k` highest (worst) stopband S21 values, so a few compliant frequencies don't hide real violations.

### Standard Mode (`enable_pv_contrast: false`)

Used for **bandpass** configurations.

```
s21_cost = (passband_weight × mean_pb + stopband_weight × mean_sb
            + cross_stopband_weight × mean_cs + spurious_peak_weight × mean_sp) × weight
```

- **Passband penalty** per frequency: `max(0, min_db − S21)^cost_order`  
  Uses `passband_target_db` if set, else `passband_min_db`.
- **Stopband penalty** per frequency: `max(0, S21 − stopband_max_db)^cost_order`
- All components aggregated via **worst-k mean** (`worst_k`) to prevent dilution.

---

## Multi-State Aggregation

After computing per-state costs, they are combined into a single scalar:

### `state_aggregation_mode: sum` (default)
```
total = Σ cost_i
```
All states contribute equally. A single bad state can be masked by good ones.

### `state_aggregation_mode: soft_max`
```
total = max_i(cost_i) + (1/T) × log( Σ exp(T × (cost_i − max_i)) )
```
**Log-sum-exp** — numerically stable. The worst state dominates the total score.  
- `state_aggregation_temperature` (T): higher T → closer to hard max; lower T → closer to mean.  
- Recommended starting point: `T = 2.0`.  
- The design cannot "hide" a poorly performing state behind two good ones.

---

## Dynamic State Assignment (`dynamic_state_assignment: true`)

The Hungarian algorithm finds the bijective assignment of simulated states to filter specs that minimises total cost. After assignment, the per-pair costs are passed to `_aggregate_state_costs`.

---

## Configuration Reference (lowpass base config)

```yaml
filter_specs:
  - enable_pv_contrast:   true
    pv_contrast_k:        15       # top-k worst stopband points for contrast
    pv_contrast_weight:   8.0      # weight on −contrast term
    pv_passband_weight:   16.0     # weight on passband penalty term
    passband_target_db:  -3.0      # one-sided target: penalise pb_avg < −3 dB
    pv_passband_order:    2.0      # quadratic: large shortfalls dominate

ga:
  state_aggregation_mode:        soft_max
  state_aggregation_temperature: 2.0      # tune: higher → harder max
  dynamic_state_assignment:      true
```
