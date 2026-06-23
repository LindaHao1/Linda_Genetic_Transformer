# RFIC_RL — Pixelated RF Filter Optimization

End-to-end flow: Virtuoso layout → EMX EM extraction → Spectre S-parameter → Genetic Algorithm.

---

## Repository layout

```
virtuoso_engine/          Virtuoso/EMX/Spectre simulation engine
  config_base.yaml          shared base config (inherited by all mode configs)
  config_impedance_transfer.yaml   impedance-transfer experiment settings
  config_differential.yaml         differential-filter experiment settings
  servers/
    mars.yaml               server profile: paths + singularity wrapper
    local.yaml              local (no wrapper)
  templates/
    pixelated_design_simple/
      layout.il             SKILL layout generation script
      emx.il                SKILL EMX extraction script  ← mesh & parallel here
    differential_filter/
      layout.il

virtuoso_optimization_GA/ Genetic algorithm
  run_ga.py               CLI entry point
  ga_optimizer.py         SwitchableFilterGA class
  configs/
    differential/
      base_differential.yaml   shared GA parameters for differential mode
      scratch_21x27.yaml       experiment: metal-only scratch
      scratch_shunt_cap_21x27.yaml
      ...
    impedance_transfer/
      base_impedance_transfer.yaml
      scratch_pixel_cap_shunt_cap_21x27.yaml
      finetune_gen290_cap_shunt_21x27.yaml
      ...

simulation_engine/        Standalone EMX runner (not used by GA)
```

---

## Configuration hierarchy

Every config uses `_base_:` inheritance and is deep-merged top-to-bottom:

```
config_base.yaml          (shared defaults: lib name, layout params, sim params)
  └─ config_differential.yaml / config_impedance_transfer.yaml
       └─ servers/mars.yaml   (merged automatically when RFIC_SERVER=mars)
```

The GA experiment config adds a third layer on top, overriding only what changes:

```
ga experiment config      (output_dir, grid size, GA weights, finetune path …)
  └─ virtuoso_config:     points to config_differential.yaml or config_impedance_transfer.yaml
```

**Key parameters and where to set them:**

| Parameter | File | Key |
|---|---|---|
| EMX mesh edge length | `templates/pixelated_design_simple/emx.il` | `-e <val>` |
| EMX mesh thickness | `templates/pixelated_design_simple/emx.il` | `-t <val>` |
| EMX 3D metals | `templates/pixelated_design_simple/emx.il` | `--3d=*` (all) or `--3d=-*` (none) |
| EMX threads per job | `config_base.yaml` → `simulation.emx_parallel_threads` | integer |
| Concurrent GA workers | GA experiment config or `base_differential.yaml` → `ga.n_parallel` | integer |
| Output directory | GA experiment config → `output_dir` | path string |
| Freq sweep | `config_base.yaml` → `simulation.emx_freq_start/stop/step` | Hz |

### Example: change EMX mesh and parallel workers

In `templates/pixelated_design_simple/emx.il` (line ~74):
```
... --quasistatic -e 1.5 -t 1.5 -v 0.5 --3d=* ...
                   ^^^   ^^^
                   edge  thickness (um)
```

In `configs/differential/base_differential.yaml`:
```yaml
ga:
  n_parallel: 6     # 6 concurrent Virtuoso+EMX workers
```

In `config_base.yaml`:
```yaml
simulation:
  emx_parallel_threads: 8    # threads inside each EMX job
```

---

## Server profiles

The mars server profile (`servers/mars.yaml`) provides:
- Correct `virtuoso_dir` path
- Correct `proc_file`, `tsmc_models`, `emx_binary` paths
- Singularity container command wrapping Virtuoso and EMX

**Always set `RFIC_SERVER=mars` when running on this machine:**

```bash
export RFIC_SERVER=mars
# or prefix every command:
RFIC_SERVER=mars conda run -n rfic python3 ...
```

Without it, the engine falls back to `server.active: local` in `config_base.yaml`
and uses stale `/disk/<username>/` paths that do not exist.

---

## Parallel engine architecture

```
GeneticAlgorithm
  └─ ParallelVirtuosoEngine (n_parallel workers)
       ├─ Worker 0  →  VirtuosoPixelEngine  →  PersistentVirtuosoRunner (Virtuoso process)
       │                                              └─ EMX subprocess
       ├─ Worker 1  →  VirtuosoPixelEngine  →  PersistentVirtuosoRunner
       │                                              └─ EMX subprocess
       └─ Worker N  ...
```

Each worker has its own:
- Virtuoso library: `pySim_w<id>`
- Output subfolder: `<output_dir>/sims/gen<N>_ind<M>/worker_<id>/`
- Virtuoso session (persistent across generations — starts once, reused)
- EMX child process (spawned per simulation)

---

## Step 1 — Test a single simulation

Before running the GA, verify the full Virtuoso → EMX pipeline with one design:

```python
# test_single_sim.py
import os, sys
os.environ["RFIC_SERVER"] = "mars"
sys.path.insert(0, "/volumes/research-localdata/<username>/RFIC_RL")

from virtuoso_engine.engine import VirtuosoPixelEngine

engine = VirtuosoPixelEngine(
    "virtuoso_engine/config_impedance_transfer.yaml",
    worker_id=0,
)

result = engine.run({
    "mode": "pixelated_design_simple",
    "project": {
        "parent_output_dir": "/volumes/research-localdata/<username>/Sim_engine_results/test_single",
    },
    "pixelated": {
        "pixel_state_matrix": [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        "state_list": [[]],
        "cap_values": [],
        "feedline": {
            "feedline_row": 1,
            "feedline_width": 10.2,
            "feedline_padding": 10.0,
            "feedline_width_in": 20,
            "feedline_width_out": 10.2,
        },
    },
    "layout": {"pixel_size": 12.5, "grid_size": 10.0},
})

engine.close()
print("Layout OK:", result.layout_result.success)
print("EMX OK:   ", result.sim_result.success)
print("S-params: ", result.s_param_files)
```

Run it:
```bash
cd /volumes/research-localdata/<username>/RFIC_RL
RFIC_SERVER=mars conda run -n rfic python3 test_single_sim.py
```

For the **differential** mode, swap the config and mode:
```python
engine = VirtuosoPixelEngine("virtuoso_engine/config_differential.yaml", worker_id=0)
result = engine.run({
    "mode": "differential_filter",
    "project": {"parent_output_dir": "..."},
    "pixelated": { ... },   # half design (engine mirrors it)
    "layout": {"pixel_size": 12.5, "grid_size": 10.0},
})
```

---

## Step 2 — Run the GA

```bash
cd /volumes/research-localdata/<username>/RFIC_RL
RFIC_SERVER=mars conda run -n rfic python3 \
    virtuoso_optimization_GA/run_ga.py \
    virtuoso_optimization_GA/configs/differential/scratch_21x27.yaml
```

Log goes to stdout. Redirect to a file and background it for long runs:
```bash
RFIC_SERVER=mars conda run -n rfic python3 \
    virtuoso_optimization_GA/run_ga.py \
    virtuoso_optimization_GA/configs/differential/scratch_shunt_cap_21x27.yaml \
    > /tmp/exp_shunt_cap.log 2>&1 &

echo "GA PID: $!"   # save this
```

**Output structure** (written to `output_dir` in the experiment config):
```
ga_<experiment>/
  ga_config_snapshot.yaml       copy of the GA config used
  virtuoso_config_snapshot.yaml copy of the engine config used
  ga_progress.json              per-generation stats
  ga_progress.png               cost/diversity plot
  best_design.json              best design found so far
  population/gen<N>.json        full population snapshots
  sims/gen<N>_ind<M>/worker_<id>/   per-simulation output
    config.yaml                 full merged config (use as finetune seed)
    emx_results/*.s<N>p         S-parameter files
```

### Fine-tuning from an existing design

Set `ga.finetune_design_path` in your experiment config to a `config.yaml`
produced by a previous run, then zero out the actions you want to freeze:

```yaml
# configs/impedance_transfer/finetune_gen290_cap_shunt_21x27.yaml
_base_: ./base_impedance_transfer_via.yaml

virtuoso_config: "/volumes/research-localdata/<username>/RFIC_RL/virtuoso_engine/config_impedance_transfer.yaml"
output_dir: "/volumes/research-localdata/<username>/Sim_engine_results/ga_my_finetune"

grid:
  rows: 21
  cols: 27
  enable_cap_pairs: true
  enable_shunt_caps: true

ga:
  n_immigrants: 0               # keep metal frozen — no random immigrants
  action_weights:
    flip_01:          0.0       # metal frozen
    add_cap_pair:     4.0
    remove_cap_pair:  3.0
    perturb_pair_caps: 6.0
    modify_shunt_cap: 4.0
    perturb_shunt_caps: 6.0
  finetune_design_path: "/volumes/research-localdata/<username>/Sim_engine_results/ga_prev_run/sims/gen290_ind1/worker_7/.../config.yaml"
  finetune_expand:
    enabled: true
    p_completely_random: 0.0
    init_max_mutations: 6
    zero_pad_first: true
    n_immigrants_expanded: 0
```

---

## Killing a GA run cleanly

The GA spawns a tree: **Python GA → Virtuoso runner → EMX child processes**.
Killing the Python PID alone leaves EMX jobs running as orphans.

```bash
# 1. Kill the GA Python process
kill -9 <GA_PID>

# 2. Kill orphaned EMX jobs for that experiment
ps aux | grep "Sim_engine_results/ga_<experiment_name>" | grep -v grep \
  | awk '{print $2}' | xargs -r kill -9

# 3. Delete the partial output directory before relaunching
rm -rf /volumes/research-localdata/<username>/Sim_engine_results/ga_<experiment_name>/
```

To find all running GA processes:
```bash
ps aux | grep run_ga.py | grep -v grep
```

To find all running EMX processes:
```bash
ps aux | grep "/emx/bin" | grep -v grep
```

> **Important:** The commands above only kill processes owned by the current user (`ps aux` filters by the user running the command). They will not affect EMX jobs belonging to other users on the same machine. Do not use `sudo` or `pkill` without a user filter — other researchers may be running their own EMX jobs on shared compute nodes.
