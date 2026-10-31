# Transformer synthesis from lumped targets

This workflow searches an octagonal transformer geometry that matches desired
`Lp`, `Ls`, coupling `k`, `Qp`, and `Qs` at a chosen frequency. The user gives
an initial geometry and legal geometry ranges. Each GA candidate is drawn in
Cadence Virtuoso, extracted with EMX, converted from its differential 4-port
S-parameters to lumped metrics, and scored against the targets.

## Requirements

- Python 3 with `numpy`, `PyYAML`, and the dependencies already used by RFIC_RL.
- Cadence Virtuoso with a writable library attached to the intended PDK.
- Cadence EMX and a licensed foundry `.proc` process file.
- A display such as X11 or Xvfb if the local Virtuoso setup requires one.

Foundry PDKs and process files are intentionally not included in this
repository. Users must be licensed to supply their own.

## Configure the installation

Copy
[`configs/transformer/transformer_synthesis_example.yaml`](configs/transformer/transformer_synthesis_example.yaml)
and edit:

- `project.virtuoso_dir`, `project.virtuoso_binary`, and `project.lib_name`
- `paths.emx_binary` and `paths.proc_file`
- transformer metal layers and `geometry_range`
- `simulation.target_freq`

The bundled octagonal transformer layout template is used when
`transformer.template_file` is omitted.

## Supply targets and initial geometry

The following example requests lumped values at the frequency configured in
the base YAML and starts the search near a specified geometry. Geometry units
are micrometres; inductances are henries.

```bash
python virtuoso_optimization_GA/transformer_synthesis.py \
  --base-config my_transformer_site.yaml \
  --output-config transformer_job.yaml \
  --lp 8.328e-10 --ls 6.204e-10 --k 0.487 --qp 20 --qs 20 \
  --pri-inner-r 55 --pri-width 10 \
  --sec-inner-r 40 --sec-width 10 \
  --population 12 --generations 20 --workers 2 \
  --run
```

Without `--run`, the command only creates the resolved job YAML, which is
useful for review or reproducibility. It can later be run with:

```bash
python virtuoso_optimization_GA/run_ga.py transformer_job.yaml
```

## Outputs

The run folder contains:

- `best_design.json`: best geometry, extracted lumped parameters, and errors
- `ga_progress.csv` and `ga_progress.json`: all generations and candidates
- `engine_config.yaml`: resolved simulator configuration
- `sims/`: generated SKILL, EMX logs, GDS, and Touchstone results

The extracted differential quantities are:

```text
Zp = Z11 - Z12 - Z21 + Z22
Zs = Z33 - Z34 - Z43 + Z44
Lp = imag(Zp)/(2*pi*f)
Ls = imag(Zs)/(2*pi*f)
k  = abs(M)/sqrt(Lp*Ls)
Qp = imag(Zp)/real(Zp)
Qs = imag(Zs)/real(Zs)
```

The optimizer reports failures instead of fabricating a metric when Virtuoso,
EMX, the process file, or the generated `.s4p` result is unavailable.
