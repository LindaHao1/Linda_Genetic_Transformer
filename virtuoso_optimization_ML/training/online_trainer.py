"""
online_trainer.py — Main 5-step online training loop for the ML optimizer.

Algorithm
---------
Step 0 (warmup)  : simulate ~100 random feasible designs; populate buffer.

Repeat for n_iterations:
  Collect: Gather simulation results submitted in the previous iteration.
           - Step-2 results → add real data to buffer, update best design.
           - Correction results → update buffer labels, fine-tune world model.
  Step 1: Sample n_gen_samples designs from the flow matching model.
          - Try up to max_resample_attempts to get feasible decoded designs.
          - Remaining infeasible designs → add to buffer with sentinel label.
  Step 2: World model predicts S-params + confidence for feasible designs.
          - High confidence → relabel + add to buffer (no simulation).
          - Low confidence  → SUBMIT to Virtuoso (non-blocking); continue.
  Step 3: Train generative model with priority-sampled buffer data.
          (Simulations run in background during Steps 3–5.)
  Step 4: Train world model on real + feasible buffer data.
  Step 5: (every wm_correction_interval iters)
          SUBMIT re-simulation of high-confidence WM-labeled entries (non-blocking).
          Results are collected at the start of the next iteration.

Logging via TensorBoard; checkpoints saved every checkpoint_interval iters.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml

from ..cost import specs_from_config, compute_total_cost
from ..feasibility import (
    generate_random_design,
    check_design_feasibility,
    FeasibilityCheckConfig,
)

from ..dataset.replay_buffer import ReplayBuffer
from ..encoding.design_encoder import DesignEncoder, EncoderConfig
from ..models.flow_matching import FlowMatchingModel
from ..models.world_model import WorldModel, RNDNetwork
from ..models.feasibility_classifier import FeasibilityClassifier
from ..training.warmup import run_warmup, reload_warmup_from_sim_folder
from ..utils.s_param_utils import (
    build_target_s_param_tensor,
    result_to_s_param_tensor,
)
from ..utils.cost_utils import compute_cost_from_tensor, compute_score_tensor
from ..utils.plot_utils import (
    plot_iter_best,
    plot_iter_worst,
    plot_iter_best_sim,
    plot_wm_vs_sim,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLConfig:
    # Grid / design space
    rows:              int
    cols:              int
    n_switches:        int      # nominal switch count (used by SA/GA; ML uses the range below)
    n_switches_min:    int      # ML feasibility: minimum allowed switch count
    n_switches_max:    int      # ML feasibility: maximum allowed switch count
    max_inductors:     int
    n_states:          int
    cap_min:           float
    cap_max:           float
    n_inductor_values: int

    # S-parameter frequency grid
    s_param_freq_min:  float
    s_param_freq_max:  float
    s_param_n_freq:    int

    # Encoding
    encoding_region_gap: float

    # Generative model
    gen_cond_dim:       int
    gen_t_dim:          int
    gen_base_channels:  int
    gen_n_steps:        int
    gen_guidance_scale: float
    gen_p_uncond:       float
    gen_lr:             float
    gen_batch_size:     int

    # World model
    wm_lr:               float
    wm_batch_size:       int
    wm_std_scale:        float
    wm_var_threshold:    float   # variance trigger for simulation
    wm_correction_interval: int
    # RND
    wm_use_rnd:          bool
    wm_rnd_threshold:    float   # RND score trigger for simulation
    wm_rnd_lr:           float

    # Feasibility classifier
    clf_enabled:                  bool
    clf_guidance_weight:          float   # weight in flow-matching loss
    clf_sample_guidance_weight:   float   # weight during ODE sampling
    clf_train_every_n_iters:      int
    clf_n_train_steps:            int
    clf_lr:                       float
    clf_min_store_size:           int     # min infeasible_image_store size before training
    clf_max_store_size:           int     # max infeasible images kept in rolling store

    # Replay buffer
    buf_max_size:     int
    buf_alpha_perf:   float
    buf_alpha_real:   float
    buf_real_bonus:   float
    buf_alpha_random: float

    # Training loop
    warmup_n_designs:       int
    n_iterations:           int
    n_gen_samples:          int
    n_sim_max:              int
    gen_train_steps:        int
    wm_train_steps:         int
    wm_correction_n:        int
    max_resample_attempts:  int
    max_infeasible_per_iter: int  # max infeasible samples added to buffer per iteration

    # Paths
    virtuoso_config: str
    output_dir:      str

    # Parallel simulation
    parallel_num_workers: int

    # Misc
    checkpoint_interval: int

    # Fields with defaults must come last
    seed:             Optional[int] = None
    debug:            bool          = False   # print pixel-type counts for each generated design

    # Feasibility check flags (which rules to enforce during ML sampling)
    feas_cfg: FeasibilityCheckConfig = field(default_factory=FeasibilityCheckConfig)

    # Guidance mode
    guidance_mode:                     str   = "s_param"   # "s_param" or "score_based"
    score_guidance_improvement_factor: float = 0.9         # target = best_scores × factor
    n_score_components:                int   = 4

    # Layout-only warmup pretraining (feasible designs, no simulation, no guidance)
    warmup_n_random_feasible:           int  = 1000  # random feasible layouts, no simulation
    warmup_random_feasible_train_steps: int  = 50    # gen model train steps on these layouts

    # World model rollout training for the generative model
    # The gen model generates a pool; WM labels each sample (no real simulation).
    # Infeasible samples receive a large penalty label.
    # Runs n_rollout_steps gradient steps per iteration (designed to be > gen_train_steps).
    gen_wm_rollout_enabled:            bool  = True
    gen_wm_rollout_n_steps:            int   = 32     # gradient steps per iteration
    gen_wm_rollout_pool_size:          int   = 128    # samples generated once per iteration
    gen_wm_rollout_infeasible_penalty: float = 100.0  # penalty for infeasible designs
    gen_wm_rollout_grad_clip:          float = 1.0

    # Two-phase generative model gradient clipping (separate budgets per loss component)
    gen_fm_grad_clip:  float = 1.0  # clip budget for flow-matching loss
    gen_clf_grad_clip: float = 0.1  # clip budget for classifier guidance loss (tighter)

    # Cross-attention spec conditioning (alternative to AdaGN)
    gen_use_cross_attn: bool = False  # enable cross-attention conditioning
    gen_seq_d_model:    int  = 128    # token embedding dimension
    gen_ca_n_heads:     int  = 4      # attention heads in cross-attention blocks
    # Classifier guidance warmup: ramp weight from 0 → clf_guidance_weight over N iters
    clf_warmup_iters:  int   = 0

    # Filter specs stored as raw dicts (parsed to FilterStateSpec on demand)
    filter_specs_raw: List[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "MLConfig":
        g   = d["grid"]
        sp  = d.get("s_param", {})
        enc = d.get("encoding", {})
        gm  = d.get("generative_model", {})
        wm  = d.get("world_model", {})
        rb  = d.get("replay_buffer", {})
        tr  = d.get("training", {})
        par = d.get("parallel", {})
        return cls(
            rows              = int(g["rows"]),
            cols              = int(g["cols"]),
            n_switches        = int(g["n_switches"]),
            n_switches_min    = int(g.get("n_switches_min", g["n_switches"])),
            n_switches_max    = int(g.get("n_switches_max", g["n_switches"])),
            max_inductors     = int(g["max_inductors"]),
            n_states          = int(g["n_states"]),
            cap_min           = float(g["cap_min"]),
            cap_max           = float(g["cap_max"]),
            n_inductor_values = int(g.get("n_inductor_values", 4)),

            s_param_freq_min  = float(sp.get("freq_min", 1e9)),
            s_param_freq_max  = float(sp.get("freq_max", 50e9)),
            s_param_n_freq    = int(sp.get("n_freq", 201)),

            encoding_region_gap = float(enc.get("region_gap", 0.05)),

            gen_cond_dim       = int(gm.get("cond_dim",       256)),
            gen_t_dim          = int(gm.get("t_dim",          128)),
            gen_base_channels  = int(gm.get("base_channels",   64)),
            gen_n_steps        = int(gm.get("n_steps",         50)),
            gen_guidance_scale = float(gm.get("guidance_scale", 2.0)),
            gen_p_uncond       = float(gm.get("p_uncond",       0.1)),
            gen_lr             = float(gm.get("lr",             1e-4)),
            gen_batch_size     = int(gm.get("batch_size",       32)),
            gen_fm_grad_clip   = float(gm.get("fm_grad_clip",   1.0)),
            gen_clf_grad_clip  = float(gm.get("clf_grad_clip",  0.1)),

            gen_use_cross_attn = bool(gm.get("use_cross_attn", False)),
            gen_seq_d_model    = int(gm.get("seq_d_model",    128)),
            gen_ca_n_heads     = int(gm.get("ca_n_heads",       4)),

            wm_lr                  = float(wm.get("lr",               1e-4)),
            wm_batch_size          = int(wm.get("batch_size",          32)),
            wm_std_scale           = float(wm.get("std_scale",         3.0)),
            wm_var_threshold       = float(wm.get("var_threshold",     0.01)),
            wm_correction_interval = int(wm.get("correction_interval", 10)),
            wm_use_rnd             = bool(wm.get("use_rnd",            True)),
            wm_rnd_threshold       = float(wm.get("rnd_threshold",     0.1)),
            wm_rnd_lr              = float(wm.get("rnd_lr",            1e-4)),

            clf_enabled                = bool(d.get("feasibility_classifier", {}).get("enabled",                True)),
            clf_guidance_weight        = float(d.get("feasibility_classifier", {}).get("guidance_weight",        0.5)),
            clf_sample_guidance_weight = float(d.get("feasibility_classifier", {}).get("sample_guidance_weight", 0.1)),
            clf_train_every_n_iters    = int(d.get("feasibility_classifier", {}).get("train_every_n_iters",      5)),
            clf_n_train_steps          = int(d.get("feasibility_classifier", {}).get("n_train_steps",            10)),
            clf_lr                     = float(d.get("feasibility_classifier", {}).get("lr",                     1e-4)),
            clf_min_store_size         = int(d.get("feasibility_classifier", {}).get("min_infeasible_store_size", 50)),
            clf_max_store_size         = int(d.get("feasibility_classifier", {}).get("max_infeasible_store_size", 500)),
            clf_warmup_iters           = int(d.get("feasibility_classifier", {}).get("warmup_iters",             0)),

            buf_max_size     = int(rb.get("max_size",      10000)),
            buf_alpha_perf   = float(rb.get("alpha_perf",   0.6)),
            buf_alpha_real   = float(rb.get("alpha_real",   0.4)),
            buf_real_bonus   = float(rb.get("real_bonus",   2.0)),
            buf_alpha_random = float(rb.get("alpha_random", 0.2)),

            warmup_n_designs      = int(tr.get("warmup_n_designs",        100)),
            n_iterations          = int(tr.get("n_iterations",            500)),
            n_gen_samples         = int(tr.get("n_gen_samples",            20)),
            n_sim_max             = int(tr.get("n_sim_max",                 8)),
            gen_train_steps       = int(tr.get("gen_train_steps",          10)),
            wm_train_steps        = int(tr.get("wm_train_steps",           10)),
            wm_correction_n       = int(tr.get("wm_correction_n",           5)),
            max_resample_attempts    = int(tr.get("max_resample_attempts",      5)),
            max_infeasible_per_iter  = int(tr.get("max_infeasible_per_iter",   5)),

            virtuoso_config      = str(d["virtuoso_config"]),
            output_dir           = str(d["output_dir"]),

            parallel_num_workers = int(par.get("num_workers", 4)),
            checkpoint_interval  = int(d.get("checkpoint_interval", 50)),
            seed                 = d.get("seed"),
            filter_specs_raw     = list(d.get("filter_specs", [])),
            debug                = bool(d.get("debug", False)),
            guidance_mode                     = str(d.get("guidance", {}).get("mode",               "s_param")),
            score_guidance_improvement_factor = float(d.get("guidance", {}).get("improvement_factor", 0.9)),
            n_score_components                = int(d.get("guidance", {}).get("n_score_components",    4)),
            warmup_n_random_feasible          = int(tr.get("warmup_n_random_feasible",           1000)),
            warmup_random_feasible_train_steps = int(tr.get("warmup_random_feasible_train_steps",  50)),
            feas_cfg = FeasibilityCheckConfig.from_dict(d.get("feasibility_checks", {})),

            gen_wm_rollout_enabled            = bool(d.get("gen_wm_rollout", {}).get("enabled",             True)),
            gen_wm_rollout_n_steps            = int(d.get("gen_wm_rollout", {}).get("n_rollout_steps",       32)),
            gen_wm_rollout_pool_size          = int(d.get("gen_wm_rollout", {}).get("pool_size",            128)),
            gen_wm_rollout_infeasible_penalty = float(d.get("gen_wm_rollout", {}).get("infeasible_penalty", 100.0)),
            gen_wm_rollout_grad_clip          = float(d.get("gen_wm_rollout", {}).get("grad_clip",            1.0)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MLConfig":
        with Path(path).open() as f:
            return cls.from_dict(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class OnlineMLTrainer:
    """
    Orchestrates the 5-step online ML training loop.

    Usage::

        trainer = OnlineMLTrainer("ml_config.yaml")
        trainer.run()
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.cfg         = MLConfig.from_yaml(config_path)
        cfg              = self.cfg

        self.rng     = random.Random(cfg.seed)
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Frequency grid
        self.freq_points = np.linspace(
            cfg.s_param_freq_min,
            cfg.s_param_freq_max,
            cfg.s_param_n_freq,
        )

        # Filter specs and target S-param conditioning tensor
        self.specs = specs_from_config(cfg.filter_specs_raw)
        self.target_s = build_target_s_param_tensor(
            self.specs, self.freq_points
        ).to(self.device)   # (n_states, n_freq)

        # Design encoder
        enc_cfg = EncoderConfig(
            rows              = cfg.rows,
            cols              = cfg.cols,
            n_states          = cfg.n_states,
            cap_min           = cfg.cap_min,
            cap_max           = cfg.cap_max,
            n_inductor_values = cfg.n_inductor_values,
            n_switches        = cfg.n_switches,
            max_inductors     = cfg.max_inductors,
        )
        self.encoder = DesignEncoder(enc_cfg)
        img_h = self.encoder.img_h
        img_w = self.encoder.img_w

        # Models
        self.gen_model = FlowMatchingModel(
            n_states           = cfg.n_states,
            n_freq             = cfg.s_param_n_freq,
            img_h              = img_h,
            img_w              = img_w,
            cond_dim           = cfg.gen_cond_dim,
            t_dim              = cfg.gen_t_dim,
            base_channels      = cfg.gen_base_channels,
            p_uncond           = cfg.gen_p_uncond,
            guidance_mode      = cfg.guidance_mode,
            n_score_components = cfg.n_score_components,
            use_cross_attn     = cfg.gen_use_cross_attn,
            seq_d_model        = cfg.gen_seq_d_model,
            ca_n_heads         = cfg.gen_ca_n_heads,
        ).to(self.device)

        self.world_model = WorldModel(
            n_states  = cfg.n_states,
            n_freq    = cfg.s_param_n_freq,
            std_scale = cfg.wm_std_scale,
        ).to(self.device)

        self.rnd_network = RNDNetwork().to(self.device) if cfg.wm_use_rnd else None

        self.clf_model = (
            FeasibilityClassifier().to(self.device)
            if cfg.clf_enabled else None
        )

        # Optimisers
        self.gen_opt = optim.Adam(self.gen_model.parameters(), lr=cfg.gen_lr)
        self.wm_opt  = optim.Adam(self.world_model.parameters(), lr=cfg.wm_lr)
        self.rnd_opt = (
            optim.Adam(
                [p for p in self.rnd_network.parameters() if p.requires_grad],
                lr=cfg.wm_rnd_lr,
            )
            if self.rnd_network is not None else None
        )
        self.clf_opt = (
            optim.Adam(self.clf_model.parameters(), lr=cfg.clf_lr)
            if self.clf_model is not None else None
        )

        # Replay buffer (feasible entries only)
        self.buffer = ReplayBuffer(
            max_size     = cfg.buf_max_size,
            alpha_perf   = cfg.buf_alpha_perf,
            alpha_real   = cfg.buf_alpha_real,
            real_bonus   = cfg.buf_real_bonus,
            alpha_random = cfg.buf_alpha_random,
            n_states     = cfg.n_states,
            n_freq       = cfg.s_param_n_freq,
            seed         = cfg.seed,
        )

        # Rolling store for infeasible images (used by FeasibilityClassifier)
        from collections import deque
        self._infeasible_store: deque = deque(maxlen=cfg.clf_max_store_size)

        # Simulation engine — started once, shared across warmup, iter sims, correction
        from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine
        self.engine = ParallelVirtuosoEngine(
            config_path = cfg.virtuoso_config,
            num_workers = cfg.parallel_num_workers,
            log_path    = self.output_dir / "engine.log",
            quiet       = True,
        )
        self.engine.start()

        # TensorBoard writer (optional)
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))
            print(f"[ML] TensorBoard logging to {self.output_dir / 'tb_logs'}")
        except ImportError:
            print("[ML] TensorBoard not available — skipping TB logging.")

        self._progress: List[dict] = []
        self._best_cost            = float("inf")
        self._best_design          = None
        self._start_iter:  int  = 0    # set by load_checkpoint() to resume mid-run
        self._skip_warmup: bool = False  # set by load_checkpoint() to avoid re-running

        # Score-based guidance state: (n_states, 4) tensor of best seen sub-scores
        # (element-wise minimum over all real simulations seen so far)
        self._best_scores: Optional[torch.Tensor] = None

        # Global step counters for per-step TensorBoard loss logging
        self._global_gen_step: int = 0
        self._global_wm_step:  int = 0

        # Pending async simulation state (submit now, collect next iteration)
        self._pending_step2: Optional[dict] = None
        self._pending_correction: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.cfg
        print(f"\n{'='*70}")
        print(f"[ML] Online ML Optimizer")
        print(f"[ML] Grid: {cfg.rows}×{cfg.cols}, switches={cfg.n_switches}, "
              f"max_ind={cfg.max_inductors}, states={cfg.n_states}")
        print(f"[ML] Device: {self.device}")
        print(f"{'='*70}\n")
        try:
            self._run_loop()
        finally:
            self.engine.close()

    def _run_loop(self) -> None:
        """Inner training loop — called by run() inside a try/finally for engine cleanup."""
        cfg = self.cfg

        # Step 0: Warmup
        # Automatically skipped if a warmup checkpoint exists in output_dir,
        # or if load_checkpoint() was called (mid-run resume).
        warmup_ckpt    = self.output_dir / "checkpoints" / "warmup"
        warmup_sim_dir = self.output_dir / "warmup_sims"

        if self._skip_warmup:
            print(f"[ML] Resuming from iteration {self._start_iter} — skipping warmup.")
        elif cfg.guidance_mode == "score_based":
            # Score-based mode: always rebuild the buffer from the sim folder so that
            # any config changes (e.g. updated filter specs) are reflected.
            print(f"[ML] Score-based guidance: reloading warmup buffer from sim folder ...")
            n_warm = reload_warmup_from_sim_folder(
                sim_dir     = warmup_sim_dir,
                encoder     = self.encoder,
                buffer      = self.buffer,
                freq_points = self.freq_points,
                specs       = self.specs,
                n_states    = cfg.n_states,
            )
            if n_warm == 0:
                # No existing sim folder — run fresh warmup
                print("[ML] No existing warmup sims found — running fresh warmup ...")
                n_warm = run_warmup(
                    cfg         = cfg,
                    encoder     = self.encoder,
                    buffer      = self.buffer,
                    freq_points = self.freq_points,
                    rng         = self.rng,
                    output_dir  = self.output_dir,
                    engine      = self.engine,
                    specs       = self.specs,
                )
                self._save_warmup_checkpoint(warmup_ckpt)
            if n_warm == 0:
                print("[ML] WARNING: Warmup produced no real data — generative model "
                      "will start from scratch.")
            # Initialise _best_scores from the best entry in the warmup buffer
            self._init_best_scores_from_buffer()
        elif warmup_ckpt.exists():
            print(f"[ML] Warmup checkpoint found — loading buffer and models, skipping warmup.")
            self._load_warmup_checkpoint(warmup_ckpt)
        elif warmup_sim_dir.exists():
            # No buffer checkpoint, but a warmup_sims folder exists — reload from it
            # so we don't re-simulate designs we already have results for.
            print(f"[ML] No warmup checkpoint but warmup_sims folder found — "
                  f"reloading buffer from {warmup_sim_dir} ...")
            n_warm = reload_warmup_from_sim_folder(
                sim_dir     = warmup_sim_dir,
                encoder     = self.encoder,
                buffer      = self.buffer,
                freq_points = self.freq_points,
                specs       = self.specs,
                n_states    = cfg.n_states,
            )
            if n_warm == 0:
                print("[ML] No usable sims found in warmup_sims — running fresh warmup ...")
                n_warm = run_warmup(
                    cfg         = cfg,
                    encoder     = self.encoder,
                    buffer      = self.buffer,
                    freq_points = self.freq_points,
                    rng         = self.rng,
                    output_dir  = self.output_dir,
                    engine      = self.engine,
                    specs       = self.specs,
                )
            else:
                print(f"[ML] Loaded {n_warm} real designs from warmup_sims folder.")
            if n_warm == 0:
                print("[ML] WARNING: Warmup produced no real data — generative model "
                      "will start from scratch.")
            self._save_warmup_checkpoint(warmup_ckpt)
        else:
            n_warm = run_warmup(
                cfg         = cfg,
                encoder     = self.encoder,
                buffer      = self.buffer,
                freq_points = self.freq_points,
                rng         = self.rng,
                output_dir  = self.output_dir,
                engine      = self.engine,
                specs       = self.specs,
            )
            if n_warm == 0:
                print("[ML] WARNING: Warmup produced no real data — generative model "
                      "will start from scratch.")

            # Save buffer immediately after data collection — before any training.
            # Future runs with the same output_dir will load this and skip simulation.
            self._save_warmup_checkpoint(warmup_ckpt)

        # Pre-train generative model on random feasible layouts (no s-params, null guidance).
        # Teaches the model the feasible design space structure before conditioning is applied.
        if not self._skip_warmup:
            self._pretrain_with_random_layouts()

        # Pre-train on whatever is in the buffer (runs on both first run and resume)
        if self.buffer.n_real() > 0:
            print("[ML] Pre-training world model on warmup data ...")
            for _ in range(20):
                self._step4_train_world_model()
            print("[ML] Pre-training generative model on warmup data ...")
            for _ in range(50):
                self._step3_train_generative(current_iter=0)

        start_time = time.time()

        for it in range(self._start_iter, cfg.n_iterations):
            iter_start = time.time()

            # Collect simulation results submitted in the previous iteration.
            # These calls block only if futures are not yet done (should be rare).
            self._collect_step2_sims()
            n_corrected = self._collect_correction_sims()

            # Step 1: Sample designs; infeasible images go to classifier store
            feasible_designs, infeasible_images = self._step1_sample_designs(
                n=cfg.n_gen_samples
            )

            # Step 2: World model variance + RND split + SUBMIT sims (non-blocking)
            n_sim, avg_variance, best_wm_cost = self._step2_evaluate(feasible_designs, it)

            # Step 3: Train generative model on real replay buffer data.
            # Simulations run in background during Steps 3–5.
            gen_loss = self._step3_train_generative(it)

            # Step 3b: Train generative model via world model rollout.
            # Generates a fresh sample pool, labels with WM (no real simulation),
            # and runs more gradient steps than Step 3 to exploit cheap WM feedback.
            gen_wm_loss = self._step3b_train_gen_wm_rollout(it)

            # Step 4: Train world model + RND predictor
            wm_loss = self._step4_train_world_model()

            # Classifier training (every N iters)
            if (
                self.clf_model is not None
                and it % cfg.clf_train_every_n_iters == 0
            ):
                self._step_train_classifier()

            # Step 5: Periodic correction — SUBMIT sims non-blockingly;
            # results are collected at the start of the next iteration.
            if it > 0 and it % cfg.wm_correction_interval == 0:
                self._submit_correction_sims(it)

            # Checkpoint
            if it % cfg.checkpoint_interval == 0:
                self._checkpoint(it)

            elapsed = time.time() - iter_start
            total   = time.time() - start_time
            best_sim_cost_val = (
                None if self._best_cost == float("inf") else self._best_cost
            )
            info = {
                "iteration":      it,
                "gen_loss":       gen_loss,
                "gen_wm_loss":    gen_wm_loss,
                "wm_loss":        wm_loss,
                "n_feasible":     len(feasible_designs),
                "n_infeasible":   len(infeasible_images),
                "avg_variance":   avg_variance,
                "best_wm_cost":   best_wm_cost,
                "best_sim_cost":  best_sim_cost_val,
                "n_sim":          n_sim,
                "n_corrected":    n_corrected,
                "buf_size":       len(self.buffer),
                "buf_real":       self.buffer.n_real(),
                "elapsed_s":      elapsed,
                "total_s":        total,
            }
            self._log_iteration(it, info)

            best_wm_str  = f"{best_wm_cost:.4f}" if best_wm_cost is not None else "n/a"
            best_sim_str = f"{best_sim_cost_val:.4f}" if best_sim_cost_val is not None else "n/a"
            print(
                f"[ML] iter {it:4d}/{cfg.n_iterations}  "
                f"gen_loss={gen_loss:.4f}  gen_wm_loss={gen_wm_loss:.4f}  wm_loss={wm_loss:.4f}  "
                f"var={avg_variance:.4f}  best_wm={best_wm_str}  best_sim={best_sim_str}  "
                f"feasible={len(feasible_designs)}  sim={n_sim}  "
                f"buf={len(self.buffer)}(real={self.buffer.n_real()})  "
                f"{elapsed:.1f}s"
            )

        # Drain any remaining pending simulations submitted in the last iteration
        print("[ML] Draining pending simulations from final iteration ...")
        self._collect_step2_sims()
        self._collect_correction_sims()

        # Final checkpoint
        self._checkpoint(cfg.n_iterations)
        print(f"\n[ML] Training complete.  Best cost: {self._best_cost:.4f}")
        if self._tb_writer:
            self._tb_writer.close()

    # ------------------------------------------------------------------
    # Step 1: Sample designs from generative model
    # ------------------------------------------------------------------

    def _step1_sample_designs(self, n: int) -> Tuple[List, List]:
        """
        Sample n designs from the flow matching model.

        Generates n * max_resample_attempts images in a single batch inference,
        then collects the first n feasible decoded designs.  Any remaining images
        (including those that could not be decoded as feasible) become the
        infeasible_images list.  If fewer than n feasible designs are found, the
        shortfall is filled with infeasible images so callers always receive n
        total entries across both lists.

        Returns
        -------
        feasible_designs : list of (DesignState, image_tensor) — up to n feasible
        infeasible_images: list of image tensors — could not be made feasible
        """
        cfg = self.cfg
        n_total    = n * cfg.max_resample_attempts
        cond_input = self._get_guidance_cond(n_total)

        # Single batch inference — no mode switching inside the loop
        self.gen_model.eval()
        raw_imgs = self.gen_model.sample(
            cond_input                 = cond_input,
            n_steps                    = cfg.gen_n_steps,
            guidance_scale             = cfg.gen_guidance_scale,
            classifier                 = self.clf_model,
            classifier_guidance_weight = cfg.clf_sample_guidance_weight if self.clf_model is not None else 0.0,
        )  # (n_total, 1, H, W)
        self.gen_model.train()

        feasible_designs: List = []
        infeasible_images: List = []

        # Diagnostic 1: per-type infeasibility counters
        _fail = {"decode_error": 0, "switch_count": 0, "inductor_count": 0,
                 "spacing": 0, "connectivity": 0, "state_distinct": 0, "other": 0}

        for i in range(n_total):
            img = raw_imgs[i]   # (1, H, W)
            if len(feasible_designs) < n:
                # Still looking for feasible designs
                design = None
                try:
                    candidate = self.encoder.decode(img)
                    feasible, reason = check_design_feasibility(
                        matrix         = candidate.matrix,
                        switch_states  = candidate.switch_states,
                        n_switches_min = cfg.n_switches_min,
                        n_switches_max = cfg.n_switches_max,
                        max_inductors  = cfg.max_inductors,
                        feas_cfg       = cfg.feas_cfg,
                    )
                    if feasible:
                        design = candidate
                    else:
                        _fail[reason if reason in _fail else "other"] += 1
                except Exception:
                    _fail["decode_error"] += 1

                if design is not None:
                    snapped_img = self.encoder.encode(design)
                    if cfg.debug:
                        self._debug_pixel_counts(i, design, feasible=True)
                    feasible_designs.append((design, snapped_img))
                else:
                    if cfg.debug:
                        self._debug_pixel_counts_raw(i, img)
                    infeasible_images.append(img.cpu())
                    self._infeasible_store.append(img.cpu())
            else:
                # We have enough feasible designs; remaining images go to infeasible store
                # infeasible_images.append(img.cpu())
                # self._infeasible_store.append(img.cpu())
                break

        # Always log feasibility breakdown
        n_fail = sum(_fail.values())
        feas_pct = 100.0 * len(feasible_designs) / n_total if n_total else 0.0
        print(
            f"  [DBG-FEAS] n_gen={n_total}  feasible={len(feasible_designs)}"
            f"({feas_pct:.0f}%)  "
            f"sw={_fail['switch_count']}  ind={_fail['inductor_count']}  "
            f"spacing={_fail['spacing']}  conn={_fail['connectivity']}  "
            f"states={_fail['state_distinct']}  decode_err={_fail['decode_error']}  "
            f"other={_fail['other']}"
        )

        return feasible_designs, infeasible_images

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _debug_pixel_counts(
        self,
        idx: int,
        design,
        feasible: bool,
        score: Optional[float] = None,
    ) -> None:
        """Print pixel-type counts, switch states, and optional score for a DesignState."""
        flat    = [v for row in design.matrix for v in row]
        n_gnd   = flat.count(0)
        n_metal = flat.count(1)
        n_ind   = flat.count(2)
        n_sw    = flat.count(3)
        total   = len(flat)
        status  = "FEASIBLE" if feasible else "INFEASIBLE"
        score_s = f"  score={score:.4f}" if score is not None else ""
        # Format each state's switch vector, e.g. "s0:[1,0,1]  s1:[0,1,0]"
        sw_s = "  ".join(
            f"s{s}:{list(design.switch_states[s])}"
            for s in range(len(design.switch_states))
        ) if design.switch_states else "none"
        print(
            f"  [DBG] design {idx:3d} {status}: "
            f"gnd={n_gnd}({n_gnd/total:.1%})  "
            f"metal={n_metal}({n_metal/total:.1%})  "
            f"inductor={n_ind}  switch={n_sw}{score_s}\n"
            f"         switch_states: {sw_s}"
        )

    def _debug_pixel_counts_raw(self, idx: int, img: torch.Tensor) -> None:
        """Print pixel-type counts for a raw (un-decodable / infeasible) image
        by doing a best-effort decode without feasibility enforcement."""
        try:
            design = self.encoder.decode(img)
            self._debug_pixel_counts(idx, design, feasible=False)
        except Exception:
            print(f"  [DBG] design {idx:3d} INFEASIBLE: decode failed")

    # ------------------------------------------------------------------
    # Step 2: World model evaluation + simulation
    # ------------------------------------------------------------------

    def _step2_evaluate(
        self,
        feasible_designs: List,
        current_iter: int,
    ) -> Tuple[int, float, Optional[float]]:
        """
        Variance + RND split.

        A design is sent to real simulation when:
          mean(variance) > var_threshold  OR  rnd_score > rnd_threshold

        All other feasible designs are relabeled with the world model prediction
        and added to the replay buffer.

        Returns
        -------
        (n_sim, avg_variance, best_wm_cost)
        """
        if not feasible_designs:
            return 0, 0.0, None

        cfg    = self.cfg
        images = torch.stack([img for _, img in feasible_designs], dim=0).to(self.device)

        self.world_model.eval()
        with torch.no_grad():
            mean_pred, variance = self.world_model.predict(images)
        self.world_model.train()

        # Per-design mean variance (scalar per sample)
        mean_var = variance.mean(dim=(-1, -2))   # (N,)

        # RND OOD scores
        rnd_scores = torch.zeros(len(feasible_designs), device=self.device)
        if self.rnd_network is not None:
            with torch.no_grad():
                rnd_scores = self.rnd_network.rnd_score(images)

        # Trigger mask: needs simulation if variance OR rnd exceeds threshold
        needs_sim = (
            (mean_var > cfg.wm_var_threshold) |
            (rnd_scores > cfg.wm_rnd_threshold)
        )

        # Proxy costs from WM predictions
        wm_costs = [
            compute_cost_from_tensor(mean_pred[i], self.freq_points, self.specs)
            for i in range(len(feasible_designs))
        ]

        if cfg.debug:
            for i in range(len(feasible_designs)):
                trigger = "SIM" if needs_sim[i].item() else "WM"
                print(
                    f"  [DBG] design {i:3d} WM_cost≈{wm_costs[i]:.4f}  "
                    f"var={mean_var[i].item():.5f}  rnd={rnd_scores[i].item():.5f}  [{trigger}]"
                )

        # S-param plots for best and worst WM-predicted designs
        plot_iter_best(
            feasible_designs = feasible_designs,
            mean_pred        = mean_pred,
            wm_costs         = wm_costs,
            current_iter     = current_iter,
            freq_points      = self.freq_points,
            target_s         = self.target_s,
            output_dir       = self.output_dir,
            tb_writer        = self._tb_writer,
            debug            = cfg.debug,
        )
        plot_iter_worst(
            feasible_designs = feasible_designs,
            mean_pred        = mean_pred,
            wm_costs         = wm_costs,
            current_iter     = current_iter,
            freq_points      = self.freq_points,
            target_s         = self.target_s,
            output_dir       = self.output_dir,
            tb_writer        = self._tb_writer,
            debug            = cfg.debug,
        )

        # Uncertainty plot (variance + RND per design)
        from ..utils.plot_utils import plot_uncertainty_summary
        plot_uncertainty_summary(
            variance_per_design = mean_var.tolist(),
            rnd_score_per_design = rnd_scores.tolist(),
            sim_mask            = needs_sim.tolist(),
            var_threshold       = cfg.wm_var_threshold,
            rnd_threshold       = cfg.wm_rnd_threshold,
            iteration           = current_iter,
            output_dir          = self.output_dir,
            writer              = self._tb_writer,
        )

        to_simulate_idx = needs_sim.nonzero(as_tuple=True)[0].tolist()

        # WM-labeled designs (low uncertainty) are no longer stored in the buffer.
        # World model feedback is instead used in _step3b_train_gen_wm_rollout, which
        # generates a fresh batch each iteration and trains the gen model directly
        # without polluting the buffer with imaginary (is_real=False) data.

        # Simulation-triggered designs (cap at n_sim_max) — submitted non-blockingly;
        # results are collected at the start of the next iteration.
        to_simulate = [feasible_designs[i] for i in to_simulate_idx[:cfg.n_sim_max]]
        n_sim = self._submit_step2_sims(to_simulate, current_iter, tag=f"iter{current_iter}")

        # TensorBoard uncertainty scalars
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("uncertainty/mean_variance",   mean_var.mean().item(),    current_iter)
            self._tb_writer.add_scalar("uncertainty/max_variance",    mean_var.max().item(),     current_iter)
            self._tb_writer.add_scalar("uncertainty/mean_rnd_score",  rnd_scores.mean().item(),  current_iter)
            self._tb_writer.add_scalar("uncertainty/max_rnd_score",   rnd_scores.max().item(),   current_iter)
            frac_sim = needs_sim.float().mean().item()
            self._tb_writer.add_scalar("uncertainty/frac_sim_triggered", frac_sim,               current_iter)

        avg_variance = mean_var.mean().item()
        best_wm_cost = min(wm_costs) if wm_costs else None
        return n_sim, avg_variance, best_wm_cost

    def _submit_step2_sims(
        self,
        design_img_pairs: List,
        current_iter: int,
        tag: str,
    ) -> int:
        """
        Submit a batch of designs to simulation non-blockingly.

        Stores futures + metadata in ``self._pending_step2`` for collection at
        the start of the next iteration via ``_collect_step2_sims()``.

        Returns
        -------
        n_submitted : number of simulations submitted.
        """
        if not design_img_pairs:
            return 0

        sim_dir   = self.output_dir / "iter_sims" / tag
        overrides = []
        for i, (design, _) in enumerate(design_img_pairs):
            overrides.append({
                "mode":      "pixelated_design_simple",
                "pixelated": design.to_pixelated_config(),
                "project":   {"parent_output_dir": str(sim_dir / f"cand{i:04d}")},
            })

        futures = [self.engine.submit(ov) for ov in overrides]
        self._pending_step2 = {
            "futures":          futures,
            "design_img_pairs": design_img_pairs,
            "current_iter":     current_iter,
            "tag":              tag,
        }
        print(f"  [ML] Submitted {len(futures)} sample(s) to simulation (non-blocking).")
        return len(futures)

    def _collect_step2_sims(
        self,
    ) -> Tuple[int, Optional[torch.Tensor], Optional[float], Optional[torch.Tensor]]:
        """
        Collect simulation results submitted by ``_submit_step2_sims()``.

        Called at the start of each iteration to gather results from the
        previous iteration's submissions.  Blocks only if futures are not yet
        done (should be rare since training ran in the meantime).

        Returns
        -------
        n_ok              : number of successful simulations.
        best_sim_s_params : (n_states, n_freq) S-param tensor for the best design,
                            or None if no simulation succeeded.
        best_sim_cost     : cost of the best design, or None if no simulation succeeded.
        best_sim_image    : (1, rows, cols+n_states) encoded image of the best design,
                            or None if no simulation succeeded.
        """
        if self._pending_step2 is None:
            return 0, None, None, None

        pending           = self._pending_step2
        futures           = pending["futures"]
        design_img_pairs  = pending["design_img_pairs"]
        current_iter      = pending["current_iter"]
        tag               = pending["tag"]

        not_done = sum(1 for f in futures if not f.done())
        if not_done:
            print(f"  [ML] Waiting for {not_done} pending simulation(s) from iter {current_iter} ...")
        results = [f.result() for f in futures]
        self._pending_step2 = None

        n_ok              = 0
        best_sim_s_params = None
        best_sim_cost     = None
        best_sim_image    = None
        n_total           = len(results)

        for (design, img), result in zip(design_img_pairs, results):
            s_params = result_to_s_param_tensor(result, self.freq_points, self.cfg.n_states)
            if s_params is None:
                continue

            cost = compute_total_cost(result.s_param_files, self.specs)
            if cost < self._best_cost:
                prev_best = self._best_cost
                self._best_cost   = cost
                self._best_design = design
                self._save_best(design, cost, current_iter)
                print(
                    f"  [ML] *** NEW BEST DESIGN ***  cost {prev_best:.4f} → {cost:.4f}"
                    f"  (iter {current_iter}, tag={tag})"
                )

            self._update_best_scores(s_params)

            if best_sim_cost is None or cost < best_sim_cost:
                best_sim_cost     = cost
                best_sim_s_params = s_params
                best_sim_image    = img

            self.buffer.add(
                image       = img,
                s_params    = s_params,
                is_real     = True,
                is_feasible = True,
                iteration   = current_iter,
                confidence  = 0.0,
                score       = cost,
            )
            n_ok += 1

        print(f"  [ML] Simulation collected (iter {current_iter}). {n_ok}/{n_total} succeeded.")

        if best_sim_s_params is not None:
            plot_iter_best_sim(
                s_params     = best_sim_s_params,
                current_iter = current_iter,
                freq_points  = self.freq_points,
                target_s     = self.target_s,
                output_dir   = self.output_dir,
                cost         = best_sim_cost,
                design_image = best_sim_image,
                tb_writer    = self._tb_writer,
                debug        = self.cfg.debug,
            )

        return n_ok, best_sim_s_params, best_sim_cost, best_sim_image

    # ------------------------------------------------------------------
    # Step 3: Train generative model
    # ------------------------------------------------------------------

    def _step3_train_generative(self, current_iter: int) -> float:
        cfg = self.cfg
        loss_accum = 0.0
        n_steps    = 0

        self.gen_model.train()
        for _ in range(cfg.gen_train_steps):
            entries = self.buffer.sample_for_generative(
                cfg.gen_batch_size,
                current_iter,
            )
            if not entries:
                break

            # Count sample types for diagnostics
            n_real_samples = sum(1 for e in entries if e.is_real)
            n_wm_samples   = sum(1 for e in entries if not e.is_real)
            if cfg.debug:
                print(f"  [DBG] gen step: real={n_real_samples}  wm={n_wm_samples}")

            images   = torch.stack([e.image    for e in entries]).to(self.device)
            s_params = torch.stack([e.s_params for e in entries]).to(self.device)

            # Re-evaluate world-model-labeled entries using the *current* world model.
            # WM predictions in the buffer may be stale (recorded when the WM was less
            # accurate).  Running a fresh inference replaces those stale labels with the
            # best available prediction without modifying the buffer.
            wm_idx = [i for i, e in enumerate(entries) if not e.is_real]
            if wm_idx and self.world_model is not None:
                wm_images = images[wm_idx]
                self.world_model.eval()
                with torch.no_grad():
                    mean_fresh, _ = self.world_model.predict(wm_images)
                self.world_model.train()
                s_params = s_params.clone()
                s_params[wm_idx] = mean_fresh

            # Conditioning: use score tensors in score_based mode, s_params otherwise.
            # In score_based mode, scores are computed from the (possibly refreshed)
            # s_params tensor so WM re-evaluations propagate to the conditioning.
            if cfg.guidance_mode == "score_based":
                cond_input = self._get_score_cond_from_s_params(
                    entries, s_params
                ).to(self.device)
                # Diagnostic 2: log training conditioning range vs. inference target
                if n_steps == 0:  # once per outer iteration (first train step only)
                    cond_mean = cond_input.mean(dim=-1)  # (B, n_states)
                    cond_min  = cond_mean.min().item()
                    cond_max  = cond_mean.max().item()
                    cond_avg  = cond_mean.mean().item()
                    print(
                        f"  [DBG-COND] training cond range: "
                        f"min={cond_min:.3f}  mean={cond_avg:.3f}  max={cond_max:.3f}"
                    )
            else:
                cond_input = s_params

            # --- Score-weighted loss ---
            # Feasible entries with a known cost: weight = 1/cost (lower cost → higher weight).
            # Feasible entries without a finite cost, and infeasible entries: weight = 1.0.
            # Weights for scored feasible entries are normalised so their mean = 1.0,
            # keeping the overall gradient magnitude comparable across steps.
            raw_w = []
            for e in entries:
                if e.is_feasible and e.score < float("inf"):
                    raw_w.append(1.0 / max(e.score, 1e-6))
                else:
                    raw_w.append(1.0)

            scored_w = [w for e, w in zip(entries, raw_w)
                        if e.is_feasible and e.score < float("inf")]
            if scored_w:
                mean_w = sum(scored_w) / len(scored_w)
                raw_w  = [
                    w / mean_w if e.is_feasible and e.score < float("inf") else w
                    for e, w in zip(entries, raw_w)
                ]
            weights = torch.tensor(raw_w, dtype=torch.float32)

            # Diagnostic 3: weight effective sample size
            if n_steps == 0:  # once per outer iteration
                w_np  = weights.numpy()
                w_sum = w_np.sum()
                eff_n = (w_sum ** 2) / (w_np ** 2).sum() if w_sum > 0 else 0.0
                print(
                    f"  [DBG-WEIGHT] max_w={w_np.max():.2f}  min_w={w_np.min():.4f}"
                    f"  eff_n={eff_n:.1f}/{len(w_np)}"
                )

            self.gen_opt.zero_grad()
            _, fm_loss, clf_loss = self.gen_model.train_step(
                images, cond_input,
                weights            = weights,
                classifier         = self.clf_model,
                classifier_weight  = cfg.clf_guidance_weight if self.clf_model is not None else 0.0,
            )

            # Linearly ramp up clf contribution to avoid early-training domination
            effective_clf_weight = cfg.clf_guidance_weight * min(
                1.0, current_iter / max(cfg.clf_warmup_iters, 1)
            )

            use_clf = (
                self.clf_model is not None
                and effective_clf_weight > 0.0
                and clf_loss.item() > 0.0
            )

            # Phase 1: FM gradients — keep graph alive only if clf pass will follow
            fm_loss.backward(retain_graph=use_clf)
            torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), cfg.gen_fm_grad_clip)

            if use_clf:
                # Save FM grads, then compute clf grads with a separate, tighter budget
                fm_grads = {
                    n: p.grad.clone()
                    for n, p in self.gen_model.named_parameters()
                    if p.grad is not None
                }
                self.gen_opt.zero_grad()
                # Phase 2: clf gradients
                (effective_clf_weight * clf_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), cfg.gen_clf_grad_clip)
                # Merge: restore FM grads on top of the clipped clf grads
                for n, p in self.gen_model.named_parameters():
                    if n in fm_grads:
                        p.grad = (p.grad + fm_grads[n]) if p.grad is not None else fm_grads[n]

            self.gen_opt.step()

            # Reconstruct total loss scalar for logging (no backward)
            total_loss_val = fm_loss.item() + cfg.clf_guidance_weight * clf_loss.item()

            # Per-step TensorBoard logging
            if self._tb_writer is not None:
                self._tb_writer.add_scalar(
                    "loss/gen_model_step", total_loss_val, self._global_gen_step
                )
                self._tb_writer.add_scalar(
                    "loss/gen_fm_step", fm_loss.item(), self._global_gen_step
                )
                if self.clf_model is not None:
                    self._tb_writer.add_scalar(
                        "loss/gen_clf_step", clf_loss.item(), self._global_gen_step
                    )
                    self._tb_writer.add_scalar(
                        "train/effective_clf_weight", effective_clf_weight, self._global_gen_step
                    )
            self._global_gen_step += 1

            loss_accum += total_loss_val
            n_steps    += 1

        return loss_accum / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Step 3b: Train generative model via world model rollout
    # ------------------------------------------------------------------

    def _step3b_train_gen_wm_rollout(self, current_iter: int) -> float:
        """
        Closed-loop generative model training using world model feedback.

        Algorithm (one call per training iteration):
          1. Generate a pool of ``gen_wm_rollout_pool_size`` images from the gen
             model in a single ODE pass (no gradient).
          2. Decode each image; check design feasibility.
             - Feasible designs  → world model predicts S-params.
             - Infeasible designs → assign large penalty labels.
          3. Run ``gen_wm_rollout_n_steps`` gradient steps on the gen model
             using random mini-batches (size = gen_batch_size) drawn from the
             labeled pool.  No entries are added to the replay buffer.

        Because no real simulation is involved, this step is cheap and can run
        many more gradient steps than the real-sample-based Step 3.

        Returns the mean flow-matching loss over all rollout gradient steps.
        """
        cfg = self.cfg
        if not cfg.gen_wm_rollout_enabled:
            return 0.0

        pool_size = cfg.gen_wm_rollout_pool_size
        penalty   = cfg.gen_wm_rollout_infeasible_penalty

        # ------------------------------------------------------------------
        # 1. Generate a fresh pool in a single ODE pass (no grad)
        # ------------------------------------------------------------------
        cond_input = self._get_guidance_cond(pool_size)
        self.gen_model.eval()
        with torch.no_grad():
            raw_imgs = self.gen_model.sample(
                cond_input                 = cond_input,
                n_steps                    = cfg.gen_n_steps,
                guidance_scale             = cfg.gen_guidance_scale,
                classifier                 = self.clf_model,
                classifier_guidance_weight = (
                    cfg.clf_sample_guidance_weight if self.clf_model is not None else 0.0
                ),
            )  # (pool_size, 1, H, W)
        self.gen_model.train()

        # ------------------------------------------------------------------
        # 2. Decode + feasibility check; collect feasible indices
        # ------------------------------------------------------------------
        decoded_imgs: List[torch.Tensor] = []
        feasible_mask: List[bool]         = []
        feasible_indices: List[int]       = []

        for i in range(pool_size):
            img    = raw_imgs[i]   # (1, H, W)
            is_ok  = False
            try:
                candidate = self.encoder.decode(img)
                ok, _     = check_design_feasibility(
                    matrix         = candidate.matrix,
                    switch_states  = candidate.switch_states,
                    n_switches_min = cfg.n_switches_min,
                    n_switches_max = cfg.n_switches_max,
                    max_inductors  = cfg.max_inductors,
                    feas_cfg       = cfg.feas_cfg,
                )
                if ok:
                    is_ok = True
                    img   = self.encoder.encode(candidate)  # snapped version
            except Exception:
                pass
            decoded_imgs.append(img.to(self.device))
            feasible_mask.append(is_ok)
            if is_ok:
                feasible_indices.append(i)

        n_feasible   = len(feasible_indices)
        feas_pct     = 100.0 * n_feasible / pool_size if pool_size else 0.0
        pool_images  = torch.stack(decoded_imgs, dim=0)   # (pool_size, 1, H, W)

        # ------------------------------------------------------------------
        # 3. World model labeling
        # ------------------------------------------------------------------
        # s_params_pool : (pool_size, n_states, n_freq) — WM pred or penalty
        s_params_pool = torch.zeros(
            pool_size, cfg.n_states, cfg.s_param_n_freq,
            dtype=torch.float32, device=self.device,
        )

        if feasible_indices:
            feas_imgs = pool_images[feasible_indices]   # (n_feas, 1, H, W)
            self.world_model.eval()
            with torch.no_grad():
                wm_preds, _ = self.world_model.predict(feas_imgs)   # (n_feas, S, F)
            self.world_model.train()
            for k, idx in enumerate(feasible_indices):
                s_params_pool[idx] = wm_preds[k]

        # Infeasible indices → fill with penalty labels
        infeasible_indices = [i for i in range(pool_size) if not feasible_mask[i]]
        if infeasible_indices:
            if cfg.guidance_mode == "score_based":
                # penalty is a large cost → fill s_params with a large positive value
                # (compute_score_tensor will map this to large penalty scores)
                fill_val = penalty
            else:
                # s_param mode: large negative dB = terrible passband insertion loss
                fill_val = -penalty
            s_params_pool[infeasible_indices] = fill_val

        # Build conditioning tensor for the whole pool
        if cfg.guidance_mode == "score_based":
            cond_pool_rows = []
            infeasible_score = torch.full(
                (cfg.n_states, cfg.n_score_components),
                penalty,
                dtype=torch.float32,
                device=self.device,
            )
            for i in range(pool_size):
                if feasible_mask[i]:
                    cond_pool_rows.append(
                        compute_score_tensor(
                            s_params_pool[i].cpu(), self.freq_points, self.specs
                        ).to(self.device)
                    )
                else:
                    cond_pool_rows.append(infeasible_score)
            cond_pool = torch.stack(cond_pool_rows, dim=0)  # (pool_size, S, n_score_components)
        else:
            cond_pool = s_params_pool   # (pool_size, S, F)

        # ------------------------------------------------------------------
        # 4. Multiple gradient steps using mini-batches from the labeled pool
        # ------------------------------------------------------------------
        B           = min(cfg.gen_batch_size, pool_size)
        loss_accum  = 0.0
        n_steps     = 0

        self.gen_model.train()
        for _ in range(cfg.gen_wm_rollout_n_steps):
            perm  = torch.randperm(pool_size, device=self.device)[:B]
            imgs  = pool_images[perm]   # (B, 1, H, W)
            conds = cond_pool[perm]     # (B, S, ...)

            effective_clf_weight = cfg.clf_guidance_weight * min(
                1.0, current_iter / max(cfg.clf_warmup_iters, 1)
            )

            self.gen_opt.zero_grad()
            _, fm_loss, clf_loss = self.gen_model.train_step(
                imgs, conds,
                classifier        = self.clf_model,
                classifier_weight = (
                    cfg.clf_guidance_weight if self.clf_model is not None else 0.0
                ),
            )

            use_clf = (
                self.clf_model is not None
                and effective_clf_weight > 0.0
                and clf_loss.item() > 0.0
            )

            fm_loss.backward(retain_graph=use_clf)
            torch.nn.utils.clip_grad_norm_(
                self.gen_model.parameters(), cfg.gen_wm_rollout_grad_clip
            )

            if use_clf:
                fm_grads = {
                    n: p.grad.clone()
                    for n, p in self.gen_model.named_parameters()
                    if p.grad is not None
                }
                self.gen_opt.zero_grad()
                (effective_clf_weight * clf_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.gen_model.parameters(), cfg.gen_clf_grad_clip
                )
                for n, p in self.gen_model.named_parameters():
                    if n in fm_grads:
                        p.grad = (
                            p.grad + fm_grads[n]
                            if p.grad is not None else fm_grads[n]
                        )

            self.gen_opt.step()

            total_loss_val = fm_loss.item() + (
                effective_clf_weight * clf_loss.item() if use_clf else 0.0
            )

            if self._tb_writer is not None:
                self._tb_writer.add_scalar(
                    "loss/gen_wm_rollout_step", total_loss_val, self._global_gen_step
                )
            self._global_gen_step += 1

            loss_accum += total_loss_val
            n_steps    += 1

        avg_loss = loss_accum / max(n_steps, 1)
        print(
            f"  [WM-Rollout] steps={n_steps}  loss={avg_loss:.4f}  "
            f"feasible={n_feasible}/{pool_size}({feas_pct:.0f}%)"
        )
        return avg_loss

    # ------------------------------------------------------------------
    # Step 4: Train world model
    # ------------------------------------------------------------------

    def _step4_train_world_model(self) -> float:
        cfg = self.cfg
        total_loss = 0.0
        n_steps    = 0

        self.world_model.train()
        for _ in range(cfg.wm_train_steps):
            entries = self.buffer.sample_for_world_model(cfg.wm_batch_size)
            if not entries:
                break

            images   = torch.stack([e.image    for e in entries]).to(self.device)
            s_params = torch.stack([e.s_params for e in entries]).to(self.device)

            # World model NLL loss
            self.wm_opt.zero_grad()
            loss = self.world_model.train_step(images, s_params)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.wm_opt.step()

            # RND predictor update (on the same batch of real images)
            if self.rnd_network is not None and self.rnd_opt is not None:
                self.rnd_opt.zero_grad()
                rnd_loss = self.rnd_network.train_step(images.detach())
                rnd_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.rnd_network.parameters() if p.requires_grad], 1.0
                )
                self.rnd_opt.step()

            # Prediction error metrics (separate no-grad forward pass on the same batch)
            with torch.no_grad():
                mean_pred, _ = self.world_model.predict(images)
                err      = mean_pred - s_params                          # (B, S, F)
                mae      = err.abs().mean().item()
                rmse     = err.pow(2).mean().sqrt().item()

            # Per-step TensorBoard logging
            if self._tb_writer is not None:
                self._tb_writer.add_scalar(
                    "loss/world_model_step", loss.item(), self._global_wm_step
                )
                self._tb_writer.add_scalar(
                    "world_model/mae_step", mae, self._global_wm_step
                )
                self._tb_writer.add_scalar(
                    "world_model/rmse_step", rmse, self._global_wm_step
                )
            self._global_wm_step += 1

            total_loss += loss.item()
            n_steps    += 1

        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Classifier training step
    # ------------------------------------------------------------------

    def _step_train_classifier(self) -> None:
        """
        Train FeasibilityClassifier on feasible buffer entries (label=1)
        vs. infeasible images from _infeasible_store (label=0).

        Skipped if either pool is too small.
        """
        cfg = self.cfg
        if self.clf_model is None or self.clf_opt is None:
            return
        if len(self._infeasible_store) < cfg.clf_min_store_size:
            return

        half = cfg.gen_batch_size // 2
        feasible_entries = self.buffer.sample_for_world_model(half)
        if not feasible_entries:
            return

        infeasible_imgs = list(self._infeasible_store)
        if len(infeasible_imgs) > half:
            infeasible_imgs = self._infeasible_store._rng_sample(half) if hasattr(
                self._infeasible_store, '_rng_sample') else infeasible_imgs[-half:]

        feasible_x   = torch.stack([e.image for e in feasible_entries]).to(self.device)
        infeasible_x = torch.stack(infeasible_imgs).to(self.device)

        self.clf_model.train()
        for _ in range(cfg.clf_n_train_steps):
            self.clf_opt.zero_grad()
            loss = self.clf_model.train_step(feasible_x, infeasible_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.clf_model.parameters(), 1.0)
            self.clf_opt.step()

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("loss/clf_step", loss.item(), self._global_wm_step)

        # Classifier accuracy metrics (computed on the last training batch)
        self.clf_model.eval()
        with torch.no_grad():
            prob_f = self.clf_model.classify(feasible_x)    # (N,) — label=1
            prob_i = self.clf_model.classify(infeasible_x)  # (M,) — label=0
            pred_f = (prob_f > 0.5).float()
            pred_i = (prob_i > 0.5).float()

            tp = pred_f.sum().item()
            tn = (1.0 - pred_i).sum().item()
            fp = pred_i.sum().item()           # infeasible predicted as feasible
            fn = (1.0 - pred_f).sum().item()   # feasible predicted as infeasible

            total     = pred_f.numel() + pred_i.numel()
            accuracy  = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        self.clf_model.train()

        if self._tb_writer is not None:
            step = self._global_wm_step
            self._tb_writer.add_scalar("classifier/accuracy",  accuracy,  step)
            self._tb_writer.add_scalar("classifier/precision", precision, step)
            self._tb_writer.add_scalar("classifier/recall",    recall,    step)

    # ------------------------------------------------------------------
    # Step 5: Correction — re-simulate world-model-labeled entries
    # ------------------------------------------------------------------

    def _submit_correction_sims(self, current_iter: int) -> None:
        """
        Submit correction simulations non-blockingly.

        Samples high-confidence WM-labeled buffer entries, decodes them, and
        submits re-simulations to the engine.  Futures + metadata are stored in
        ``self._pending_correction`` for collection at the start of the next
        iteration via ``_collect_correction_sims()``.
        """
        cfg = self.cfg
        entries = self.buffer.sample_for_correction(
            n                    = cfg.wm_correction_n,
            confidence_threshold = 0.0,
        )
        if not entries:
            return

        # Get WM predictions BEFORE re-simulation (for later comparison plot)
        self.world_model.eval()
        wm_preds_before: List[torch.Tensor] = []
        with torch.no_grad():
            try:
                entry_images = torch.stack([e.image for e in entries]).to(self.device)
                mean_before, _ = self.world_model.confidence(entry_images)
                for k in range(mean_before.shape[0]):
                    wm_preds_before.append(mean_before[k].cpu())
            except Exception:
                wm_preds_before = [e.s_params for e in entries]
        self.world_model.train()

        sim_dir   = self.output_dir / "correction" / f"iter{current_iter}"
        overrides = []
        for i, entry in enumerate(entries):
            try:
                design = self.encoder.decode(entry.image)
            except Exception:
                continue
            overrides.append((
                i,
                entry.entry_id,
                {
                    "mode":      "pixelated_design_simple",
                    "pixelated": design.to_pixelated_config(),
                    "project":   {"parent_output_dir": str(sim_dir / f"corr{i:04d}")},
                },
            ))

        if not overrides:
            return

        futures = [self.engine.submit(o[2]) for o in overrides]
        self._pending_correction = {
            "futures":        futures,
            "overrides":      overrides,
            "entries":        entries,
            "wm_preds_before": wm_preds_before,
            "current_iter":   current_iter,
        }
        print(f"  [ML] Submitted {len(futures)} correction sim(s) (non-blocking).")

    def _collect_correction_sims(self) -> int:
        """
        Collect correction simulation results submitted by ``_submit_correction_sims()``.

        Called at the start of each iteration.  Updates buffer labels, plots
        WM-vs-sim comparison, and fine-tunes the world model on corrected data.

        Returns
        -------
        n_corrected : number of successfully collected correction simulations.
        """
        if self._pending_correction is None:
            return 0

        pending         = self._pending_correction
        futures         = pending["futures"]
        overrides       = pending["overrides"]
        entries         = pending["entries"]
        wm_preds_before = pending["wm_preds_before"]
        current_iter    = pending["current_iter"]
        cfg             = self.cfg

        not_done = sum(1 for f in futures if not f.done())
        if not_done:
            print(f"  [ML] Waiting for {not_done} pending correction sim(s) from iter {current_iter} ...")
        results = [f.result() for f in futures]
        self._pending_correction = None

        n_correction_total = len(results)
        correction_images   = []
        correction_s_params = []
        plot_wm_list:  List[torch.Tensor] = []
        plot_sim_list: List[torch.Tensor] = []
        plot_eids:     List[int]          = []
        n_corrected = 0

        for (i, entry_id, _), result in zip(overrides, results):
            s_params = result_to_s_param_tensor(result, self.freq_points, cfg.n_states)
            if s_params is None:
                continue

            cost = compute_total_cost(result.s_param_files, self.specs)
            self.buffer.update_entry(
                entry_id   = entry_id,
                s_params   = s_params,
                is_real    = True,
                confidence = 0.0,
                score      = cost,
            )

            correction_images.append(entries[i].image)
            correction_s_params.append(s_params)

            if i < len(wm_preds_before):
                plot_wm_list.append(wm_preds_before[i])
                plot_sim_list.append(s_params)
                plot_eids.append(entry_id)

            n_corrected += 1

        print(f"  [ML] Correction collected (iter {current_iter}). {n_corrected}/{n_correction_total} succeeded.")

        if plot_wm_list:
            plot_wm_vs_sim(
                wm_preds     = plot_wm_list,
                sim_preds    = plot_sim_list,
                entry_ids    = plot_eids,
                current_iter = current_iter,
                freq_points  = self.freq_points,
                target_s     = self.target_s,
                output_dir   = self.output_dir,
                tb_writer    = self._tb_writer,
                debug        = cfg.debug,
            )

        # Include best known real designs so the world model does not forget them
        best_real_entries = self.buffer.sample_best_real(n=cfg.wm_correction_n)
        for entry in best_real_entries:
            correction_images.append(entry.image)
            correction_s_params.append(entry.s_params)

        if correction_images:
            imgs    = torch.stack(correction_images).to(self.device)
            sparams = torch.stack(correction_s_params).to(self.device)
            self.world_model.train()
            for _ in range(2):
                self.wm_opt.zero_grad()
                loss = self.world_model.train_step(imgs, sparams)
                loss.backward()
                self.wm_opt.step()

        return n_corrected

    # ------------------------------------------------------------------
    # Layout-only warmup pretraining (no simulation required)
    # ------------------------------------------------------------------

    def _pretrain_with_random_layouts(self) -> None:
        """
        Generate ``warmup_n_random_feasible`` random feasible designs and pre-train
        the generative model on them **unconditionally** (null guidance).

        These designs have no S-parameter data, so they are trained with
        ``force_uncond=True``, teaching the model the feasible design space
        before any conditioning signal is introduced.  They are never added
        to the replay buffer and are discarded after this call.
        """
        cfg = self.cfg
        n   = cfg.warmup_n_random_feasible
        if n <= 0:
            return

        print(f"[ML] Generating {n} random feasible layouts for unconditional pretraining ...")
        designs = []
        attempts = 0
        max_attempts = n * 50
        while len(designs) < n and attempts < max_attempts:
            attempts += 1
            d = generate_random_design(
                rows              = cfg.rows,
                cols              = cfg.cols,
                n_switches        = cfg.n_switches,
                max_inductors     = cfg.max_inductors,
                n_states          = cfg.n_states,
                cap_min           = cfg.cap_min,
                cap_max           = cfg.cap_max,
                rng               = self.rng,
                n_inductor_values = cfg.n_inductor_values,
            )
            if d is not None:
                designs.append(d)

        if not designs:
            print("[ML] WARNING: Could not generate any random feasible layouts.")
            return

        print(f"[ML] Generated {len(designs)} layouts.  "
              f"Pre-training generative model unconditionally "
              f"({cfg.warmup_random_feasible_train_steps} steps) ...")

        all_images = torch.stack(
            [self.encoder.encode(d) for d in designs]
        ).to(self.device)   # (n, 1, H, W)

        # Dummy conditioning tensor — bypassed by force_uncond=True
        n_cond = cfg.n_score_components if cfg.guidance_mode == "score_based" else cfg.s_param_n_freq
        dummy_cond = torch.zeros(
            cfg.gen_batch_size, cfg.n_states, n_cond, device=self.device,
        )

        self.gen_model.train()
        for _ in range(cfg.warmup_random_feasible_train_steps):
            idx   = torch.randperm(len(all_images), device=self.device)[:cfg.gen_batch_size]
            batch = all_images[idx]
            B     = batch.shape[0]
            self.gen_opt.zero_grad()
            total_loss, _, _ = self.gen_model.train_step(
                batch, dummy_cond[:B], force_uncond=True,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), 1.0)
            self.gen_opt.step()

        print(f"[ML] Layout-only pretraining done.")

    # ------------------------------------------------------------------
    # Score-based guidance helpers
    # ------------------------------------------------------------------

    def _update_best_scores(self, s_params: torch.Tensor) -> None:
        """
        Update ``_best_scores`` element-wise with a new real simulation result.

        Parameters
        ----------
        s_params : (n_states, n_freq) S21 tensor from a real simulation.
        """
        if self.cfg.guidance_mode != "score_based":
            return
        new_scores = compute_score_tensor(s_params, self.freq_points, self.specs)
        if self._best_scores is None:
            self._best_scores = new_scores.clone()
        else:
            self._best_scores = torch.minimum(self._best_scores, new_scores)

    def _init_best_scores_from_buffer(self) -> None:
        """
        Initialise ``_best_scores`` from the best real entry in the warmup buffer.
        Called once after the warmup phase when in score_based mode.
        """
        if self.cfg.guidance_mode != "score_based":
            return
        best_entries = self.buffer.sample_best_real(n=1)
        if not best_entries:
            return
        s_params = best_entries[0].s_params.to(self.device)
        self._best_scores = compute_score_tensor(
            s_params, self.freq_points, self.specs
        )
        print(f"[ML] Initialised score guidance from best warmup entry: "
              f"{self._best_scores.tolist()}")

    def _get_guidance_cond(self, batch_size: int) -> torch.Tensor:
        """
        Return the conditioning tensor for flow matching model sampling.

        In 's_param' mode    : returns (B, n_states, n_freq) — the ideal target S-params.
        In 'score_based' mode: returns (B, n_states, 4)      — gradual score target.
        """
        cfg = self.cfg
        if cfg.guidance_mode == "score_based":
            if self._best_scores is None:
                # No real data yet — use all-zeros (perfect score) as placeholder
                target = torch.zeros(
                    cfg.n_states, cfg.n_score_components, device=self.device
                )
            else:
                # Guide toward a slightly better score than the current best
                target = self._best_scores.to(self.device) * cfg.score_guidance_improvement_factor
            # Diagnostic 2: log inference target (mean over states/components)
            print(
                f"  [DBG-COND] inference target (mean per state): "
                f"{target.mean(dim=-1).tolist()}  "
                f"(best_scores×{cfg.score_guidance_improvement_factor})"
            )
            return target.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            return self.target_s.unsqueeze(0).expand(batch_size, -1, -1)

    def _get_score_cond_from_entries(self, entries) -> torch.Tensor:
        """
        Compute score-based conditioning tensors from a list of buffer entries.

        Feasible entries get their per-region score tensor computed from s_params.
        Infeasible entries get a fixed large "infeasible" label (all components = 100)
        that is clearly distinct from any physically realizable design's scores.

        Returns (B, n_states, 4).
        Used in ``_step3_train_generative`` when ``guidance_mode == 'score_based'``.
        """
        cfg = self.cfg
        infeasible_score = torch.full(
            (cfg.n_states, cfg.n_score_components), 100.0, dtype=torch.float32
        )
        rows = []
        for e in entries:
            if e.is_feasible:
                rows.append(compute_score_tensor(e.s_params, self.freq_points, self.specs))
            else:
                rows.append(infeasible_score)
        return torch.stack(rows, dim=0)   # (B, n_states, 4)

    def _get_score_cond_from_s_params(
        self,
        entries,
        s_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score-based conditioning from a pre-built (B, n_states, n_freq)
        s_params tensor rather than reading each entry's buffer field.

        This is used in ``_step3_train_generative`` so that freshly re-predicted
        s_params (from the current world model) are used for conditioning instead
        of the potentially stale values stored in each BufferEntry.

        Infeasible entries still receive the fixed large infeasible score regardless
        of their row in s_params (infeasible sentinel is not a valid S-parameter).
        """
        cfg = self.cfg
        infeasible_score = torch.full(
            (cfg.n_states, cfg.n_score_components), 100.0, dtype=torch.float32
        )
        rows = []
        for i, e in enumerate(entries):
            if e.is_feasible:
                rows.append(compute_score_tensor(
                    s_params[i].cpu(), self.freq_points, self.specs
                ))
            else:
                rows.append(infeasible_score)
        return torch.stack(rows, dim=0)   # (B, n_states, 4)

    # ------------------------------------------------------------------
    # Logging + checkpointing
    # ------------------------------------------------------------------

    def _log_iteration(self, it: int, metrics: dict) -> None:
        self._progress.append(metrics)
        # Save progress JSON
        prog_path = self.output_dir / "ml_progress.json"
        prog_path.write_text(json.dumps(self._progress, indent=2))

        if self._tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(f"ml/{k}", v, it)

    def _save_warmup_checkpoint(self, ckpt_dir: Path) -> None:
        """Save buffer immediately after warmup data collection (no model weights)."""
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.buffer.save(ckpt_dir / "buffer.pkl")
        print(f"  [ML] Warmup buffer saved → {ckpt_dir}  "
              f"({len(self.buffer)} entries, {self.buffer.n_real()} real)")

    def _load_warmup_checkpoint(self, ckpt_dir: Path) -> None:
        """Restore buffer from warmup checkpoint; models start from random weights."""
        self.buffer = ReplayBuffer.load(ckpt_dir / "buffer.pkl")
        print(f"  [ML] Warmup buffer loaded  "
              f"({len(self.buffer)} entries, {self.buffer.n_real()} real)")

    def _checkpoint(self, it: int) -> None:
        ckpt_dir = self.output_dir / "checkpoints" / f"iter{it:05d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.gen_model.state_dict(),   ckpt_dir / "gen_model.pt")
        torch.save(self.world_model.state_dict(), ckpt_dir / "world_model.pt")
        torch.save(self.gen_opt.state_dict(),     ckpt_dir / "gen_opt.pt")
        torch.save(self.wm_opt.state_dict(),      ckpt_dir / "wm_opt.pt")
        if self.rnd_network is not None:
            torch.save(self.rnd_network.state_dict(), ckpt_dir / "rnd_network.pt")
        if self.clf_model is not None:
            torch.save(self.clf_model.state_dict(), ckpt_dir / "clf_model.pt")
        self.buffer.save(ckpt_dir / "buffer.pkl")
        # Save iteration so resume knows where to continue from
        (ckpt_dir / "iter.json").write_text(json.dumps({"iteration": it}))
        print(f"  [ML] Checkpoint saved → {ckpt_dir}")

    def _save_best(self, design, cost: float, it: int) -> None:
        data = {
            "iteration":       it,
            "cost":            cost,
            "matrix":          design.matrix,
            "switch_states":   design.switch_states,
            "cap_values":      design.cap_values,
            "inductor_values": design.inductor_values,
        }
        path = self.output_dir / "best_design.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"  [ML] Best design saved → {path}  (cost={cost:.4f})")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """
        Resume training from a saved checkpoint directory.

        Restores model weights, optimizer states, and the replay buffer.
        Warmup is skipped and the training loop resumes from the saved iteration.
        """
        ckpt = Path(checkpoint_dir)
        self.gen_model.load_state_dict(
            torch.load(ckpt / "gen_model.pt",   map_location=self.device)
        )
        self.world_model.load_state_dict(
            torch.load(ckpt / "world_model.pt", map_location=self.device)
        )
        self.gen_opt.load_state_dict(
            torch.load(ckpt / "gen_opt.pt",     map_location=self.device)
        )
        self.wm_opt.load_state_dict(
            torch.load(ckpt / "wm_opt.pt",      map_location=self.device)
        )
        if self.rnd_network is not None and (ckpt / "rnd_network.pt").exists():
            self.rnd_network.load_state_dict(
                torch.load(ckpt / "rnd_network.pt", map_location=self.device)
            )
        if self.clf_model is not None and (ckpt / "clf_model.pt").exists():
            self.clf_model.load_state_dict(
                torch.load(ckpt / "clf_model.pt", map_location=self.device)
            )
        self.buffer = ReplayBuffer.load(ckpt / "buffer.pkl")

        # Restore iteration counter; fall back to 0 if iter.json is absent
        # (for backward compatibility with checkpoints saved before this change)
        iter_file = ckpt / "iter.json"
        if iter_file.exists():
            saved_it = json.loads(iter_file.read_text())["iteration"]
            self._start_iter = saved_it + 1   # resume on the NEXT iteration
        else:
            self._start_iter = 0

        self._skip_warmup = True
        print(f"[ML] Resumed from checkpoint {ckpt}  (starting at iter {self._start_iter})")
