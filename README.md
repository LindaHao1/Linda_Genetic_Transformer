# Linda Genetic Transformer

Target-driven octagonal transformer synthesis using Cadence Virtuoso, EMX,
and a genetic algorithm.

The user supplies desired `Lp`, `Ls`, coupling `k`, `Qp`, `Qs`, an initial
primary/secondary geometry, and legal geometry ranges. The program generates
candidate layouts, extracts each candidate with EMX, calculates the lumped
parameters from its differential four-port result, and searches for a geometry
closer to the target.

## Download

Clone the repository:

```bash
git clone https://github.com/LindaHao1/Linda_Genetic_Transformer.git
cd Linda_Genetic_Transformer
```

Update an existing clone:

```bash
git pull origin main
```

Without Git, open
<https://github.com/LindaHao1/Linda_Genetic_Transformer>, select
**Code → Download ZIP**, and extract it. Direct ZIP:

<https://github.com/LindaHao1/Linda_Genetic_Transformer/archive/refs/heads/main.zip>

## Requirements

- Python 3 with NumPy, PyYAML, and the dependencies imported by RFIC_RL.
- Licensed Cadence Virtuoso and Cadence EMX installations.
- A writable Virtuoso library attached to the intended PDK.
- The licensed EMX `.proc` file for the intended metal stack.
- X11 or Xvfb if required by the local Virtuoso installation.

This repository intentionally excludes proprietary PDKs and process files.
Every user must provide their own licensed files.

## Configure Virtuoso and EMX

Copy the example:

```bash
cp \
  virtuoso_optimization_GA/configs/transformer/transformer_synthesis_example.yaml \
  my_transformer_site.yaml
```

Edit `my_transformer_site.yaml`. Replace all `/path/to/...` values and confirm
the PDK layers, geometry ranges, and target frequency:

```yaml
project:
  lib_name: transformer_synthesis
  parent_output_dir: ./transformer_runs
  virtuoso_dir: /path/to/cadence/project
  virtuoso_binary: /path/to/virtuoso-or-wrapper

transformer:
  pri_layer: AP
  sec_layer: M9
  gnd_layer: M5

geometry_range:
  pri_inner_r: {min: 35.0, max: 100.0}
  pri_width:   {min: 5.0,  max: 18.0}
  sec_inner_r: {min: 25.0, max: 100.0}
  sec_width:   {min: 5.0,  max: 18.0}

simulation:
  target_freq: 15e9

paths:
  proc_file: /path/to/licensed/foundry/process.proc
  emx_binary: /path/to/emx
```

Layer names and geometry limits must be valid for the selected PDK. The
bundled `transformer_layout.il` and `transformer_emx.il` templates are used by
default.

## Terminal usage

Check the interface:

```bash
python virtuoso_optimization_GA/transformer_synthesis.py --help
```

The example below targets `Lp=832.8 pH`, `Ls=620.4 pH`, `k=0.487`, and
`Qp=Qs=20`. Inductances use henries and geometry uses micrometres.

```bash
python virtuoso_optimization_GA/transformer_synthesis.py \
  --base-config my_transformer_site.yaml \
  --output-config transformer_job.yaml \
  --run-name example_transformer \
  --lp 8.328e-10 \
  --ls 6.204e-10 \
  --k 0.487 \
  --qp 20 \
  --qs 20 \
  --pri-inner-r 55 \
  --pri-width 10 \
  --sec-inner-r 40 \
  --sec-width 10 \
  --population 12 \
  --generations 20 \
  --workers 2 \
  --run
```

To review the generated job before running EMX, omit `--run`. Start it later:

```bash
python virtuoso_optimization_GA/run_ga.py transformer_job.yaml
```

### Recommended smoke test

Verify the entire Virtuoso-to-EMX path before starting a long search:

```bash
python virtuoso_optimization_GA/transformer_synthesis.py \
  --base-config my_transformer_site.yaml \
  --output-config transformer_smoke_test.yaml \
  --run-name transformer_smoke_test \
  --lp 8.328e-10 --ls 6.204e-10 --k 0.487 \
  --qp 20 --qs 20 \
  --pri-inner-r 55 --pri-width 10 \
  --sec-inner-r 40 --sec-width 10 \
  --population 4 --generations 1 --workers 1 \
  --run
```

Confirm that Virtuoso creates the layout and EMX produces an `.s4p` file before
increasing the population, generations, mesh density, or worker count.

## Results

```text
transformer_best_design.json
transformer_ga_progress.csv
transformer_ga_progress.json
transformer_engine_config.yaml
transformer_ga_config_snapshot.yaml
transformer_parallel_engine.log
sims/
  .../transformer_config.yaml
  .../transformer_generated_layout.il
  .../transformer_generated_emx.il
  .../*.gds
  .../*.s4p
  .../*_emx.log
```

`transformer_best_design.json` contains the best geometry, extracted lumped
parameters, target values, cost, and percentage errors.

The differential extraction uses:

```text
Zp = Z11 - Z12 - Z21 + Z22
Zs = Z33 - Z34 - Z43 + Z44
Lp = imag(Zp)/(2*pi*f)
Ls = imag(Zs)/(2*pi*f)
k  = abs(M)/sqrt(Lp*Ls)
Qp = imag(Zp)/real(Zp)
Qs = imag(Zs)/real(Zs)
```

## Use with Codex

If Codex is already installed and signed in, start it from the repository so
it can inspect the code and local simulator logs:

```bash
cd Linda_Genetic_Transformer
codex
```

Official Codex documentation: <https://developers.openai.com/codex/>.

### Prompt 1: configure the local installation

```text
Inspect this repository and configure the transformer synthesis flow. Find the
installed Cadence Virtuoso and EMX executables and my licensed EMX process
file. Copy transformer_synthesis_example.yaml to my_transformer_site.yaml and
update only paths and settings verified on this machine. Do not invent a PDK,
process file, layer name, executable, component value, or license. Explain each
change. Run only CLI help and configuration checks; do not start EMX yet.
```

### Prompt 2: create a synthesis job

```text
Use my_transformer_site.yaml to create transformer_job.yaml with these targets:
Lp=<H>, Ls=<H>, k=<value>, Qp=<value>, Qs=<value>. Use this initial geometry in
micrometres: primary inner radius=<value>, primary width=<value>, secondary
inner radius=<value>, secondary width=<value>. Do not use --run yet. Validate
that the initial geometry is inside geometry_range and show the exact command.
```

### Prompt 3: run and verify a smoke test

```text
Review transformer_job.yaml and run a smoke test with population 4,
1 generation, and 1 worker. Monitor it through completion. Verify that
Virtuoso generated a transformer layout and EMX produced a valid four-port S4P
file. Report actual log errors and do not fabricate lumped results.
```

### Prompt 4: run and summarize the full optimization

```text
The smoke test passed. Run the full optimization from transformer_job.yaml and
monitor it through completion. Summarize transformer_best_design.json and
transformer_ga_progress.csv: best geometry, Lp, Ls, k, Qp, Qs, percentage
errors, and cost. Give paths to the GDS, S4P, and EMX log. Do not change targets
or bounds without my approval.
```

Depending on local Codex settings, launching licensed tools, installing
software, or changing external systems may require user approval.

## Troubleshooting

- **Virtuoso executable not found:** load the Cadence environment or correct
  `project.virtuoso_binary`.
- **Process file not found:** correct `paths.proc_file`; use only the licensed
  process file for the selected PDK and metal stack.
- **No S4P file:** inspect the EMX log, P1–P4/GND mapping, pin connectivity,
  layer names, GDS export, and process stack.
- **Every candidate has a large penalty:** inspect
  `transformer_parallel_engine.log` and the individual files under `sims/`.
- **License or memory failures:** reduce workers, population, EMX threads, or
  mesh density.

## Validation before fabrication

An optimized geometry is not automatically tape-out ready. Perform the required
DRC, LVS, EM convergence checks, process-corner analysis, and circuit-level
validation before fabrication.

More implementation detail is available in
[`virtuoso_optimization_GA/TRANSFORMER_SYNTHESIS.md`](virtuoso_optimization_GA/TRANSFORMER_SYNTHESIS.md).
