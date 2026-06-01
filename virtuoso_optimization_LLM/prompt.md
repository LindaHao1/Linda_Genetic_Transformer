# Project: Generative LM for Switchable RF Filter Design with Surrogate-Guided Training

## Context to read first (do this before writing any code)

1. Read the pretrained world model in <PATH_TO_WORLD_MODEL_FOLDER>. Understand:
   - Input format (what design + state representation it expects)
   - Output format (what S-parameters or scores it predicts)
   - How to load weights and run inference (batch size, device, preprocessing)
   - Confirm it is single-state only — we will need to call it N times per
     multi-state filter (once per switch/inductor state configuration)
   - Any normalization, tokenization, or embedding it does on inputs

   Summarize what you found in a short markdown doc at docs/world_model_notes.md
   before proceeding. Include exact function signatures for loading and calling it.

2. Read any existing dataset format docs or example data files. Identify:
   - Pixel grid dimensions
   - Component types and how they're currently represented
   - How multi-state configurations are stored (state codes per design)
   - How S-parameter targets/labels are stored


## What we're building

An autoregressive transformer (decoder-only, GPT-style) that generates RF filter
layouts conditioned on multi-state S-parameter targets. The model outputs:
  (a) pixel-type grid
  (b) component tunable values
  (c) state codes for each of K states

Training uses iterated surrogate-guided fine-tuning with periodic simulator
recalibration. No RL in this version — pure rejection-sampling / soft-label SFT.


## Architecture requirements

### Token vocabulary and dictionary

Build a token vocabulary module at src/tokenizer.py that defines:
  - Pixel-type tokens: ground, metal, switch, inductor, cap_pair_H_head,
    cap_pair_V_head, cap_pair_tail, port, pad
  - Value tokens: discretize each tunable parameter (switch cap, inductor cap,
    cap-pair value) into N_BINS bins. Make N_BINS configurable (default 64).
  - State code tokens: {0, 1} per (state_index, component_index)
  - Structural tokens: BOS, EOS, SEP_spec, SEP_grid, SEP_values, SEP_states
  - Spec tokens: handle S-parameters as either binned tokens OR continuous
    embeddings projected to model_dim — make this a config flag, default to
    continuous embeddings via a small MLP encoder

Provide a Tokenizer class with:
  - encode(design_dict) -> List[int] and decode(tokens) -> design_dict
  - vocab_size property, get_token_type(token_id) helper
  - Save/load the vocab as JSON so it's reproducible

### Sequence layout

  [BOS]
  [spec_embedding or spec_tokens] [SEP_spec]
  [pixel tokens in raster or Hilbert order, H*W of them] [SEP_grid]
  [value tokens, one per component, in deterministic order] [SEP_values]
  [state code tokens, K * num_components of them]
  [EOS]

Make generation order configurable (raster vs Hilbert), default raster.

### Model

Standard decoder-only transformer at src/model.py:
  - 8–12 layers, model_dim 384–512, ~30M params for initial dev
  - 2D RoPE or 2D learned positional embeddings for the pixel segment
  - Segment embeddings added to indicate which section a token belongs to
    (spec / grid / values / states)
  - Spec is prefixed and attended to via causal attention (prefix-LM style)
  - Support classifier-free guidance: 15% spec-dropout during training

### Constrained decoding — PLACEHOLDER

Create src/constraints.py with this signature:

    def get_invalid_token_mask(
        partial_sequence,    # List[int]
        tokenizer,           # Tokenizer
        config,              # ConstraintConfig
    ) -> torch.Tensor:       # shape [vocab_size], True where token is invalid
        """
        PLACEHOLDER — user will fill in rules.
        Expected rules to support (leave clearly-marked TODOs):
          - Chebyshev separation between switches/inductors (radius r)
          - Capacitor pair grammar (tail only valid after H_head or below V_head)
          - Component count budgets
          - Port pixel locking
        For now, return an all-False mask (no constraints).
        """
        return torch.zeros(tokenizer.vocab_size, dtype=torch.bool)

Make sure generate() in the model applies this mask via logits[mask] = -inf
before sampling, at every decoding step.


## Surrogate interface

Create src/surrogate.py as a wrapper around the pretrained world model:
  - class Surrogate with predict(design, state_code) -> s_params
  - predict_multistate(design, state_codes_K) -> List[s_params] that calls the
    world model K times and stacks results
  - score(predicted_sparams, target_sparams) -> float — a scalar closeness
    metric (default: negative MSE in dB, but make it pluggable)
  - Support ensemble: load multiple pretrained instances if available, return
    mean and variance. If only one instance exists, variance is 0 — log a
    warning, since uncertainty-based selection won't work without it.


## Training pipeline

### Phase 1: Initial SFT

scripts/train_sft.py:
  - Standard next-token-prediction cross-entropy
  - Hindsight relabeling: condition each example on its TRUE achieved S-params,
    not on a target it was meant to hit
  - Token-type loss weighting: pixel=1.0, value=3.0, state=3.0 (configurable)
  - 15% spec dropout for CFG support
  - Save checkpoints, log loss broken down by token type

### Phase 2: Iterated surrogate-guided fine-tuning

scripts/train_iterated.py. Main loop:

    for iteration in range(N_ITERATIONS):

        # 1. Generate large synthetic pool
        target_specs = sample_diverse_targets(...)
        candidates = lm.generate_batch(
            target_specs,
            n_per_target=POOL_MULTIPLIER,
            apply_constraints=True,
            cfg_scale=W,
        )

        # 2. Score all candidates with surrogate (multi-state aware)
        pred_sparams, pred_uncertainty = surrogate.predict_multistate_batch(candidates)
        scores = surrogate.score(pred_sparams, target_specs)

        # 3. Soft-label SFT: train on full pool weighted by score
        #    weight = softmax(score / temperature) per target_spec group
        #    OR rejection-sampling SFT on top-K% — configurable
        soft_labels = compute_soft_weights(scores, mode=CONFIG.label_mode)
        finetune_lm(candidates, soft_labels, n_steps=STEPS_PER_ITER)

        # 4. Every K iterations: simulator recalibration
        if iteration % K_RECAL == 0:
            # Pick top-M by surrogate score with small exploration mix
            to_simulate = select_for_simulation(
                candidates, scores, pred_uncertainty,
                m=M_RECAL, explore_frac=0.2,
            )
            true_sparams = run_simulator(to_simulate)  # PLACEHOLDER

            # Relabel and add to dataset
            dataset.add_relabeled(to_simulate, true_sparams)

            # Retrain surrogate on cumulative real data
            surrogate.retrain(dataset.real_only())

            # Log calibration: surrogate predictions vs true on this batch
            log_calibration(pred_sparams_for_simulated, true_sparams)

        # 5. Eval on held-out target specs
        eval_metrics = run_eval(lm, holdout_targets, surrogate, simulator_subset)
        log(eval_metrics)

Configurable hyperparameters in configs/default.yaml:
  - POOL_MULTIPLIER (default 50)
  - STEPS_PER_ITER (default 500)
  - N_ITERATIONS (default 30)
  - K_RECAL (resimulate every K iterations, default 3)
  - M_RECAL (top-M designs to resimulate, default 64)
  - label_mode: soft_weighted | rejection_topk | rejection_topp
    (default soft_weighted)
  - soft_label_temperature (default 0.1)
  - cfg_scale (default 4.0)
  - explore_frac (default 0.2 — fraction of recal budget for high-uncertainty)

### Simulator interface — PLACEHOLDER

Create src/simulator.py:

    def run_simulator(designs):
        """
        PLACEHOLDER — user will wire to actual EM simulator.
        For development, return surrogate predictions with small added noise
        so the pipeline runs end-to-end.
        """
        raise NotImplementedError("Wire to real simulator")


## Data structures

Define dataclasses in src/types.py:
  - Design: pixel_grid (HxW int), values (dict of component_id -> float),
    state_codes (KxN_components int)
  - SParams: frequency_bins, s11/s21/etc (complex arrays), per-state
  - TrainingExample: design, sparams (true if available else predicted),
    source ('simulated' | 'surrogate_labeled'), iteration_added


## Logging and evaluation

Use Weights & Biases or TensorBoard (configurable). Log per iteration:
  - Loss by token type
  - Mean surrogate score of generated pool
  - Mean true score on resimulated designs
  - Surrogate calibration: R² and MSE of (surrogate_score, true_score) on resims
  - Diversity metric on generated pool (mean pairwise embedding distance)
  - Held-out eval: best-of-N true score on a fixed set of target specs

Flag when surrogate score keeps rising but true score plateaus — this is the
reward-hacking warning sign.


## Project structure

  project/
    docs/
      world_model_notes.md     (write after reading world model folder)
      architecture.md          (write after planning, before code)
    src/
      tokenizer.py
      model.py
      constraints.py           (placeholder, clearly marked)
      surrogate.py
      simulator.py             (placeholder, clearly marked)
      dataset.py
      types.py
    scripts/
      train_sft.py
      train_iterated.py
      eval.py
    configs/
      default.yaml
    tests/
      test_tokenizer.py
      test_constraints.py      (mostly placeholders for now)
      test_pipeline_dryrun.py  (end-to-end smoke test with tiny model)


## Work order

  1. Read the world model folder. Write docs/world_model_notes.md.
  2. Write docs/architecture.md summarizing the plan and asking me any
     clarifying questions BEFORE writing code. Specifically confirm:
       - Grid dimensions
       - Number of states K
       - Number of frequency bins in the spec
       - Whether to use ensemble surrogates
  3. Implement tokenizer + tests.
  4. Implement model + constrained decoding hook.
  5. Implement surrogate wrapper (call the existing world model K times).
  6. Implement dataset, simulator placeholder, types.
  7. Implement Phase 1 SFT script.
  8. Implement Phase 2 iterated training loop.
  9. Write a tiny end-to-end smoke test that runs one full iteration with a
     minimal config (small model, small pool, K_RECAL=1) using the placeholder
     simulator. This must pass before Phase 2 is considered done.
 10. Document how to plug in the real simulator and the real constraints.


## Style and behavior

  - Don't write code until you've read the world model folder and written
    docs/world_model_notes.md and docs/architecture.md.
  - Ask clarifying questions before implementing anything ambiguous.
  - Mark all placeholders with `# TODO(user):` comments and a brief
    description of what's expected.
  - Prefer clear, readable code over clever code. Type hints throughout.
  - Don't add libraries beyond torch, numpy, einops, pyyaml, wandb (optional),
    pytest. Ask before adding anything else.