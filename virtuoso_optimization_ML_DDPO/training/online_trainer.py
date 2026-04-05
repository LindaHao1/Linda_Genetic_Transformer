"""
online_trainer.py — Main 5-step online training loop for the ML optimizer.

Algorithm
---------
Step 0 (warmup)  : simulate ~100 random feasible designs; populate buffer.

Repeat for n_iterations:
  Step 1: Sample n_gen_samples designs from the flow matching model.
          - Try up to max_resample_attempts to get feasible decoded designs.
          - Remaining infeasible designs → add to buffer with sentinel label.
  Step 2: World model predicts S-params + confidence for feasible designs.
          - High confidence → relabel + add to buffer (no simulation).
          - Low confidence  → simulate with Virtuoso; add as real data.
  Step 3: Train generative model with priority-sampled buffer data.
  Step 4: Train world model on real + feasible buffer data.
  Step 5: (every wm_correction_interval iters)
          Re-simulate high-confidence world-model-labeled entries;
          update buffer labels; fine-tune world model on corrected data.

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

from virtuoso_optimization_SA.cost import specs_from_config, compute_total_cost
from virtuoso_optimization_SA.feasibility import (
    is_feasible, check_switch_states_distinct, generate_random_design,
)
from virtuoso_optimization_SA.feasibility import (
    check_feed_columns, check_connectivity, check_switch_spacing,
    check_inductor_spacing, check_inductor_count, check_switch_count_range,
)

from ..dataset.replay_buffer import ReplayBuffer, make_infeasible_sentinel
from ..encoding.design_encoder import DesignEncoder, EncoderConfig
from ..models.diffusion_model import DiffusionModel
from ..models.world_model import EnsembleWorldModel
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
    wm_n_members:                   int
    wm_lr:                          float
    wm_batch_size:                  int
    wm_std_scale:                   float
    wm_confidence_threshold:        float
    wm_confidence_schedule_iters:   int    # iterations to ramp threshold 1.0 → target
    wm_use_bagging:                 bool
    wm_correction_interval:         int

    # Replay buffer
    buf_max_size:                   int
    buf_alpha_recency:              float
    buf_alpha_real:                 float
    buf_real_bonus:                 float
    buf_decay_rate:                 float
    buf_infeasible_weight:          float
    buf_infeasible_min_fraction:    float  # guaranteed infeasible fraction per gen batch

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
    # DDPO fine-tuning
    ddpo_n_steps:            int   = 20     # stochastic DDPM steps for DDPO trajectory
    ddpo_batch_size:         int   = 8      # designs sampled per DDPO step
    ddpo_interval:           int   = 5      # run DDPO every N iterations
    ddpo_infeasible_penalty: float = 1.0    # reward penalty per rule violation
    ddpo_eta:                float = 0.7    # stochasticity: 1=full DDPM, <1 → closer to DDIM
    seed:             Optional[int] = None
    debug:            bool          = False   # print pixel-type counts for each generated design

    # Guidance mode
    guidance_mode:                     str   = "s_param"   # "s_param" or "score_based"
    score_guidance_improvement_factor: float = 0.9         # target = best_scores × factor
    n_score_components:                int   = 4

    # Layout-only warmup pretraining
    warmup_n_random_feasible:          int   = 1000  # random feasible layouts, no simulation
    warmup_random_feasible_train_steps: int  = 50    # gen model train steps on these layouts

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
            gen_n_steps        = int(gm.get("n_ddim_steps", gm.get("n_steps", 50))),
            gen_guidance_scale    = float(gm.get("guidance_scale",    2.0)),
            gen_p_uncond          = float(gm.get("p_uncond",          0.1)),
            gen_lr                = float(gm.get("lr",                1e-4)),
            gen_batch_size        = int(gm.get("batch_size",          32)),
            wm_n_members            = int(wm.get("n_members",            5)),
            wm_lr                   = float(wm.get("lr",                  1e-4)),
            wm_batch_size           = int(wm.get("batch_size",            32)),
            wm_std_scale            = float(wm.get("std_scale",           3.0)),
            wm_confidence_threshold       = float(wm.get("confidence_threshold",       0.7)),
            wm_confidence_schedule_iters  = int(wm.get("confidence_schedule_iters",   200)),
            wm_use_bagging                = bool(wm.get("use_bagging",                True)),
            wm_correction_interval  = int(wm.get("correction_interval",  10)),

            buf_max_size          = int(rb.get("max_size",           10000)),
            buf_alpha_recency     = float(rb.get("alpha_recency",      0.6)),
            buf_alpha_real        = float(rb.get("alpha_real",         0.4)),
            buf_real_bonus        = float(rb.get("real_bonus",         2.0)),
            buf_decay_rate        = float(rb.get("decay_rate",         0.05)),
            buf_infeasible_weight        = float(rb.get("infeasible_weight",        0.1)),
            buf_infeasible_min_fraction  = float(rb.get("infeasible_min_fraction",  0.15)),

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
            ddpo_n_steps            = int(d.get("ddpo", {}).get("n_steps",            20)),
            ddpo_batch_size         = int(d.get("ddpo", {}).get("batch_size",          8)),
            ddpo_interval           = int(d.get("ddpo", {}).get("interval",            5)),
            ddpo_infeasible_penalty = float(d.get("ddpo", {}).get("infeasible_penalty", 1.0)),
            ddpo_eta                = float(d.get("ddpo", {}).get("eta",               0.7)),
            guidance_mode                     = str(d.get("guidance", {}).get("mode",               "s_param")),
            score_guidance_improvement_factor = float(d.get("guidance", {}).get("improvement_factor", 0.9)),
            n_score_components                = int(d.get("guidance", {}).get("n_score_components",    4)),
            warmup_n_random_feasible          = int(tr.get("warmup_n_random_feasible",           1000)),
            warmup_random_feasible_train_steps = int(tr.get("warmup_random_feasible_train_steps",  50)),
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
        self.gen_model = DiffusionModel(
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
        ).to(self.device)

        self.world_model = EnsembleWorldModel(
            n_members   = cfg.wm_n_members,
            img_h       = img_h,
            img_w_total = img_w,
            n_states    = cfg.n_states,
            n_freq      = cfg.s_param_n_freq,
            std_scale   = cfg.wm_std_scale,
            use_bagging = cfg.wm_use_bagging,
        ).to(self.device)

        # Optimisers
        self.gen_opt = optim.Adam(self.gen_model.parameters(), lr=cfg.gen_lr)
        self.wm_opt  = optim.Adam(self.world_model.parameters(), lr=cfg.wm_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(
            max_size          = cfg.buf_max_size,
            alpha_recency     = cfg.buf_alpha_recency,
            alpha_real        = cfg.buf_alpha_real,
            real_bonus        = cfg.buf_real_bonus,
            decay_rate        = cfg.buf_decay_rate,
            infeasible_weight = cfg.buf_infeasible_weight,
            n_states          = cfg.n_states,
            n_freq            = cfg.s_param_n_freq,
            seed              = cfg.seed,
        )

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
        warmup_ckpt     = self.output_dir / "checkpoints" / "warmup"
        warmup_sim_dir  = self.output_dir / "warmup_sims"

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
                # No existing sim folder — run fresh warmup and save checkpoint
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

        # Pre-train generative model on random feasible layouts (no s-params, null guidance)
        # This is done before any s-param conditioned pre-training so the model first learns
        # the feasible design space structure, then learns to condition on s-params/scores.
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

            # Step 1: Sample designs from generative model
            feasible_designs, infeasible_images = self._step1_sample_designs(
                n=cfg.n_gen_samples
            )

            # Add infeasible designs to buffer with sentinel label (capped per iteration)
            sentinel = make_infeasible_sentinel(cfg.n_states, cfg.s_param_n_freq)
            for img in infeasible_images[:cfg.max_infeasible_per_iter]:
                self.buffer.add(
                    image       = img,
                    s_params    = sentinel,
                    is_real     = False,
                    is_feasible = False,
                    iteration   = it,
                    confidence  = 0.0,
                )

            # Step 2: World model confidence split + simulation
            n_sim = self._step2_evaluate(feasible_designs, it)

            # Step 3: Train generative model
            gen_loss = self._step3_train_generative(it)
            # gen_loss = 0

            # Step 4: Train world model
            wm_loss = self._step4_train_world_model()

            # Step 5: Periodic correction
            n_corrected = 0
            if it > 0 and it % cfg.wm_correction_interval == 0:
                n_corrected = self._step5_correction(it)

            # DDPO: fine-tune diffusion model with RL feedback from world model
            # ddpo_loss_val = 0.0
            # if it > 0 and it % cfg.ddpo_interval == 0 and self.buffer.n_real() > 0:
                # ddpo_loss_val = self._step_ddpo(it)
            ddpo_loss_val = 0

            # Checkpoint
            if it % cfg.checkpoint_interval == 0:
                self._checkpoint(it)

            elapsed = time.time() - iter_start
            total   = time.time() - start_time
            info = {
                "iteration":   it,
                "gen_loss":    gen_loss,
                "wm_loss":     wm_loss,
                "ddpo_loss":   ddpo_loss_val,
                "n_feasible":  len(feasible_designs),
                "n_infeasible": len(infeasible_images),
                "n_sim":       n_sim,
                "n_corrected": n_corrected,
                "buf_size":    len(self.buffer),
                "buf_real":    self.buffer.n_real(),
                "elapsed_s":   elapsed,
                "total_s":     total,
            }
            self._log_iteration(it, info)

            print(
                f"[ML] iter {it:4d}/{cfg.n_iterations}  "
                f"gen_loss={gen_loss:.4f}  wm_loss={wm_loss:.4f}  "
                f"feasible={len(feasible_designs)}  sim={n_sim}  "
                f"buf={len(self.buffer)}(real={self.buffer.n_real()})  "
                f"{elapsed:.1f}s"
            )

        # Final checkpoint
        self._checkpoint(cfg.n_iterations)
        print(f"\n[ML] Training complete.  Best cost: {self._best_cost:.4f}")
        if self._tb_writer:
            self._tb_writer.close()

    # ------------------------------------------------------------------
    # Confidence threshold schedule
    # ------------------------------------------------------------------

    def _current_confidence_threshold(self, it: int) -> float:
        """
        Linear schedule: confidence threshold starts at 1.0 (iter 0) and
        decreases to ``wm_confidence_threshold`` over
        ``wm_confidence_schedule_iters`` iterations, then stays constant.

        Starting at 1.0 means no design is ever treated as high-confidence
        early in training, forcing more real simulations and better
        initialisation of the world model before it is trusted.
        """
        target       = self.cfg.wm_confidence_threshold
        sched_iters  = self.cfg.wm_confidence_schedule_iters
        if sched_iters <= 0 or it >= sched_iters:
            return target
        frac = it / sched_iters           # 0 → 1 over the schedule window
        return 1.0 - frac * (1.0 - target)  # 1.0 → target

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
        n_total  = n * cfg.max_resample_attempts
        cond_input = self._get_guidance_cond(n_total)

        # Single batch inference — no mode switching inside the loop
        self.gen_model.eval()
        with torch.no_grad():
            raw_imgs = self.gen_model.sample(
                cond_input      = cond_input,
                n_steps         = cfg.gen_n_steps,
                guidance_scale  = cfg.gen_guidance_scale,
            )  # (n_total, 1, H, W)
        self.gen_model.train()

        feasible_designs: List = []
        infeasible_images: List = []

        for i in range(n_total):
            img = raw_imgs[i]   # (1, H, W)
            if len(feasible_designs) < n:
                # Still looking for feasible designs
                design = None
                try:
                    candidate = self.encoder.decode(img)
                    if (
                        is_feasible(
                            candidate.matrix,
                            cfg.n_switches,
                            cfg.max_inductors,
                            min_switches = cfg.n_switches_min,
                            max_switches = cfg.n_switches_max,
                        )
                        and check_switch_states_distinct(candidate.switch_states)
                    ):
                        design = candidate
                except Exception:
                    pass

                if design is not None:
                    snapped_img = self.encoder.encode(design)
                    if cfg.debug:
                        self._debug_pixel_counts(i, design, feasible=True)
                    feasible_designs.append((design, snapped_img))
                else:
                    if cfg.debug:
                        self._debug_pixel_counts_raw(i, img)
                    infeasible_images.append(img.cpu())
            else:
                # We have enough feasible designs; remaining images go to infeasible
                infeasible_images.append(img.cpu())

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
    ) -> int:
        """
        Run world model confidence split.  High-confidence designs are
        relabeled and added to buffer.  Low-confidence designs are sent to
        the real simulator (up to n_sim_max per iteration).

        Returns number of real simulations run.
        """
        if not feasible_designs:
            return 0

        cfg    = self.cfg
        images = torch.stack([img for _, img in feasible_designs], dim=0).to(self.device)

        self.world_model.eval()
        mean_pred, conf_score = self.world_model.confidence(images)
        self.world_model.train()

        # Proxy costs from WM predictions (used for logging and per-iter plot)
        wm_costs = [
            compute_cost_from_tensor(mean_pred[i], self.freq_points, self.specs)
            for i in range(len(feasible_designs))
        ]

        if cfg.debug:
            for i in range(len(feasible_designs)):
                conf_label = (
                    "HIGH" if conf_score[i].item() >= cfg.wm_confidence_threshold
                    else "low"
                )
                print(
                    f"  [DBG] design {i:3d} WM_cost≈{wm_costs[i]:.4f}  "
                    f"conf={conf_score[i].item():.3f}({conf_label})"
                )

        # Save S-param plots for the best and worst WM-predicted designs this iteration
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

        conf_threshold = self._current_confidence_threshold(current_iter)

        high_conf_idx = []
        low_conf_idx  = []
        for i, conf in enumerate(conf_score.tolist()):
            if conf >= conf_threshold:
                high_conf_idx.append(i)
            else:
                low_conf_idx.append(i)

        # High-confidence → add to buffer with world model label + WM predicted cost
        for i in high_conf_idx:
            _, img = feasible_designs[i]
            self.buffer.add(
                image       = img,
                s_params    = mean_pred[i].cpu(),
                is_real     = False,
                is_feasible = True,
                iteration   = current_iter,
                confidence  = conf_score[i].item(),
                score       = wm_costs[i],
            )

        # Low-confidence → simulate (up to n_sim_max)
        to_simulate = [feasible_designs[i] for i in low_conf_idx[:cfg.n_sim_max]]
        n_sim, best_sim_s_params, best_sim_cost = self._simulate_and_add(
            to_simulate, current_iter, tag=f"iter{current_iter}"
        )

        # Plot best simulated design if any simulation succeeded
        if best_sim_s_params is not None:
            plot_iter_best_sim(
                s_params     = best_sim_s_params,
                current_iter = current_iter,
                freq_points  = self.freq_points,
                target_s     = self.target_s,
                output_dir   = self.output_dir,
                cost         = best_sim_cost,
                tb_writer    = self._tb_writer,
                debug        = cfg.debug,
            )

        return n_sim

    def _simulate_and_add(
        self,
        design_img_pairs: List,
        current_iter: int,
        tag: str,
    ) -> Tuple[int, Optional[torch.Tensor], Optional[float]]:
        """
        Simulate a batch of designs and add real results to buffer.

        Returns
        -------
        n_ok              : number of successful simulations.
        best_sim_s_params : (n_states, n_freq) S-param tensor for the best design,
                            or None if no simulation succeeded.
        best_sim_cost     : cost of the best design, or None if no simulation succeeded.
        """
        if not design_img_pairs:
            return 0, None, None

        sim_dir   = self.output_dir / "iter_sims" / tag
        overrides = []
        for i, (design, _) in enumerate(design_img_pairs):
            overrides.append({
                "mode":      "pixelated_design_simple",
                "pixelated": design.to_pixelated_config(),
                "project":   {"parent_output_dir": str(sim_dir / f"cand{i:04d}")},
            })

        n_total = len(overrides)
        print(f"  [ML] Sending {n_total} sample(s) to simulation ...")
        results = self.engine.run_batch(overrides)

        n_ok              = 0
        best_sim_s_params = None
        best_sim_cost     = None

        for (design, img), result in zip(design_img_pairs, results):
            s_params = result_to_s_param_tensor(result, self.freq_points, self.cfg.n_states)
            if s_params is None:
                continue

            # Track best cost across all iterations
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
            # Update per-region best scores for score-based guidance
            self._update_best_scores(s_params)

            # Track best design in this simulation batch
            if best_sim_cost is None or cost < best_sim_cost:
                best_sim_cost     = cost
                best_sim_s_params = s_params

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

        print(f"  [ML] Simulation done. {n_ok}/{n_total} succeeded.")
        return n_ok, best_sim_s_params, best_sim_cost

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
                min_infeasible_fraction=cfg.buf_infeasible_min_fraction,
            )
            if not entries:
                break

            # Count sample types for diagnostics
            n_real_samples       = sum(1 for e in entries if e.is_real and e.is_feasible)
            n_wm_samples         = sum(1 for e in entries if not e.is_real and e.is_feasible)
            n_infeasible_samples = sum(1 for e in entries if not e.is_feasible)
            print(
                f"  [DBG] gen step: real={n_real_samples}  "
                f"wm={n_wm_samples}  infeasible={n_infeasible_samples}"
            )

            images   = torch.stack([e.image    for e in entries]).to(self.device)
            s_params = torch.stack([e.s_params for e in entries]).to(self.device)

            # Conditioning: use score tensors in score_based mode, s_params otherwise
            if cfg.guidance_mode == "score_based":
                cond_input = self._get_score_cond_from_entries(entries).to(self.device)
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

            self.gen_opt.zero_grad()
            loss = self.gen_model.train_step(images, cond_input, weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), 1.0)
            self.gen_opt.step()

            # Per-step TensorBoard logging
            if self._tb_writer is not None:
                self._tb_writer.add_scalar(
                    "loss/gen_model_step", loss.item(), self._global_gen_step
                )
            self._global_gen_step += 1

            loss_accum += loss.item()
            n_steps    += 1

        return loss_accum / max(n_steps, 1)

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

            self.wm_opt.zero_grad()
            loss = self.world_model.train_step(images, s_params)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.wm_opt.step()

            # Per-step TensorBoard logging
            if self._tb_writer is not None:
                self._tb_writer.add_scalar(
                    "loss/world_model_step", loss.item(), self._global_wm_step
                )
            self._global_wm_step += 1

            total_loss += loss.item()
            n_steps    += 1

        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Step 5: Correction — re-simulate high-confidence entries
    # ------------------------------------------------------------------

    def _step5_correction(self, current_iter: int) -> int:
        cfg = self.cfg
        entries = self.buffer.sample_for_correction(
            n                    = cfg.wm_correction_n,
            confidence_threshold = self._current_confidence_threshold(current_iter),
        )
        if not entries:
            return 0

        # Get WM predictions for selected entries BEFORE re-simulation
        self.world_model.eval()
        wm_preds_before: List[torch.Tensor] = []
        with torch.no_grad():
            try:
                entry_images = torch.stack([e.image for e in entries]).to(self.device)
                mean_before, _ = self.world_model.confidence(entry_images)
                # mean_before: (n_entries, n_states, n_freq)
                for k in range(mean_before.shape[0]):
                    wm_preds_before.append(mean_before[k].cpu())
            except Exception:
                wm_preds_before = [e.s_params for e in entries]  # fallback: stored WM label
        self.world_model.train()

        sim_dir   = self.output_dir / "correction" / f"iter{current_iter}"
        overrides = []
        for i, entry in enumerate(entries):
            # Decode image back to design for simulation override
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
            return 0

        sim_overrides = [o[2] for o in overrides]
        n_correction_total = len(sim_overrides)
        print(f"  [ML] Sending {n_correction_total} sample(s) to simulation (correction) ...")
        results = self.engine.run_batch(sim_overrides)

        # Fine-tune world model on corrected data
        correction_images   = []
        correction_s_params = []
        # Collect (wm_pred_before, sim_s_params, entry_id) for plotting
        plot_wm_list:  List[torch.Tensor] = []
        plot_sim_list: List[torch.Tensor] = []
        plot_eids:     List[int]          = []
        n_corrected = 0

        for (i, entry_id, _), result in zip(overrides, results):
            s_params = result_to_s_param_tensor(result, self.freq_points, cfg.n_states)
            if s_params is None:
                continue

            # Compute real cost and update buffer entry (label + score)
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

            # Collect for WM-vs-sim plot
            if i < len(wm_preds_before):
                plot_wm_list.append(wm_preds_before[i])
                plot_sim_list.append(s_params)
                plot_eids.append(entry_id)

            n_corrected += 1

        print(f"  [ML] Simulation done. {n_corrected}/{n_correction_total} succeeded (correction).")

        # Plot WM predictions vs real simulation for corrected entries
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

        # Include best known real designs so the world model does not forget
        # high-performing designs that were scored by the real simulator.
        best_real_entries = self.buffer.sample_best_real(n=cfg.wm_correction_n)
        for entry in best_real_entries:
            correction_images.append(entry.image)
            correction_s_params.append(entry.s_params)

        # Fine-tune world model on corrected + best-real entries
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
    # DDPO: RL fine-tuning of diffusion model via world model feedback
    # ------------------------------------------------------------------

    def _infeasible_penalty(self, design) -> float:
        """
        Compute a composite infeasibility penalty for a DesignState.

        Returns a positive float that combines:
          - Binary violations: 1.0 per failed rule (feed cols, connectivity,
            switch/inductor spacing, switch states distinct).
          - Proportional excess for count violations:
              * Each switch above n_switches_max counts as +1 additional penalty.
              * Each inductor above max_inductors counts as +1 additional penalty.
        This makes the reward signal graded: designs further from feasibility
        receive larger punishment, providing a more informative gradient signal.
        """
        cfg = self.cfg
        penalty = 0.0
        m = design.matrix

        # Binary rule violations
        if not check_feed_columns(m):     penalty += 10.0
        if not check_connectivity(m):     penalty += 10.0
        if not check_switch_spacing(m):   penalty += 10.0
        if not check_inductor_spacing(m): penalty += 10.0
        if not check_switch_states_distinct(design.switch_states): penalty += 30.0

        # Proportional switch count excess: each switch above max_switches adds 1
        n_sw = sum(v == 3 for row in m for v in row)
        if n_sw > cfg.n_switches_max:
            penalty += float(n_sw - cfg.n_switches_max)
        elif n_sw < cfg.n_switches_min:
            penalty += 1.0   # binary: too few switches

        # Proportional inductor count excess: each inductor above max adds 1
        n_ind = sum(v == 2 for row in m for v in row)
        if n_ind > cfg.max_inductors:
            penalty += float(n_ind - cfg.max_inductors)

        return max(penalty, 1.0)   # at least 1.0 for any infeasible design

    def _step_ddpo(self, it: int) -> float:
        """
        DDPO fine-tuning step.

        1. Sample B_ddpo trajectories from the diffusion model (stochastic DDPM).
        2. Decode each x_0 and compute a reward:
             - Infeasible design  → R = -penalty * n_violations  (always contributes)
             - Feasible + confident world model → R = -cost       (contributes)
             - Feasible + uncertain world model → masked out       (no gradient)
        3. If no designs are confident/infeasible: skip the update.
        4. Normalise rewards over contributing designs; compute DDPO policy gradient.
        5. Update diffusion model parameters.

        Returns the DDPO loss value (0.0 if skipped).
        """
        cfg    = self.cfg
        B      = cfg.ddpo_batch_size
        device = self.device

        # --- Sample trajectories ---
        self.gen_model.eval()
        cond_batch = self._get_guidance_cond(B)
        x0_samples, trajectories = self.gen_model.ddpo_sample(
            cond_input     = cond_batch,
            n_steps        = cfg.ddpo_n_steps,
            guidance_scale = cfg.gen_guidance_scale,
            eta            = cfg.ddpo_eta,
        )
        self.gen_model.train()

        # --- Compute rewards ---
        rewards:   list = []
        mask:      list = []   # True = contributes to DDPO loss
        active_idx: list = []  # which samples are active

        self.world_model.eval()
        for i in range(B):
            x0 = x0_samples[i]         # (1, H, W)
            design     = None
            is_feasible_flag = False
            try:
                candidate = self.encoder.decode(x0)
                is_feasible_flag = (
                    is_feasible(
                        candidate.matrix,
                        cfg.n_switches,
                        cfg.max_inductors,
                        min_switches = cfg.n_switches_min,
                        max_switches = cfg.n_switches_max,
                    )
                    and check_switch_states_distinct(candidate.switch_states)
                )
                design = candidate
            except Exception:
                pass

            if not is_feasible_flag:
                pen = self._infeasible_penalty(design) if design is not None else 6.0
                r = -cfg.ddpo_infeasible_penalty * pen
                print(f"  [DDPO] iter {it}: infeasible design sampled. with penalty={pen:.1f}  → reward={r:.1f}")
                rewards.append(r)
                mask.append(True)
                active_idx.append(i)
            else:
                # Query world model confidence
                snapped_img = self.encoder.encode(design).unsqueeze(0).to(device)
                with torch.no_grad():
                    mean_pred, conf = self.world_model.confidence(snapped_img)
                conf_val = conf.item()
                thresh   = self._current_confidence_threshold(it)

                if conf_val >= thresh:
                    cost = compute_cost_from_tensor(mean_pred[0], self.freq_points, self.specs)
                    rewards.append(-cost)
                    mask.append(True)
                    active_idx.append(i)
                else:
                    rewards.append(0.0)
                    mask.append(False)
        self.world_model.train()

        n_active = sum(mask)
        if n_active == 0:
            print(f"  [DDPO] iter {it}: no confident designs — skipping update.")
            return 0.0

        # --- Raw rewards (before normalisation) — for logging ---
        raw_rewards = torch.tensor(
            [r for r, m in zip(rewards, mask) if m], dtype=torch.float32
        )
        r_mean = raw_rewards.mean().item()
        r_min  = raw_rewards.min().item()
        r_max  = raw_rewards.max().item()
        r_std  = raw_rewards.std().item() if n_active > 1 else 0.0
        n_infeas_active = sum(
            1 for r, m in zip(rewards, mask) if m and r < 0 and r != -float("inf")
        )

        # --- Normalise rewards ---
        active_rewards = raw_rewards.to(device)
        std = active_rewards.std()
        if std > 1e-8:
            active_rewards = (active_rewards - active_rewards.mean()) / (std + 1e-8)
        else:
            active_rewards = active_rewards - active_rewards.mean()

        # --- DDPO loss ---
        active_traj       = [trajectories[i] for i in active_idx]
        active_cond_input = cond_batch[active_idx]

        self.gen_opt.zero_grad()
        loss = self.gen_model.ddpo_loss(
            trajectories   = active_traj,
            rewards        = active_rewards,
            cond_input     = active_cond_input,
            guidance_scale = cfg.gen_guidance_scale,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), 1.0)
        self.gen_opt.step()

        loss_val = loss.item()
        print(
            f"  [DDPO] iter {it}: loss={loss_val:.4f}  active={n_active}/{B}  "
            f"reward  mean={r_mean:.4f}  min={r_min:.4f}  max={r_max:.4f}  std={r_std:.4f}"
        )

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("loss/ddpo",          loss_val,       it)
            self._tb_writer.add_scalar("ddpo/n_active",      n_active,       it)
            self._tb_writer.add_scalar("ddpo/reward_mean",   r_mean,         it)
            self._tb_writer.add_scalar("ddpo/reward_min",    r_min,          it)
            self._tb_writer.add_scalar("ddpo/reward_max",    r_max,          it)
            self._tb_writer.add_scalar("ddpo/reward_std",    r_std,          it)
            self._tb_writer.add_scalar("ddpo/n_infeasible",  n_infeas_active, it)

        return loss_val

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

        # Dummy conditioning tensor — shape doesn't matter since force_uncond=True
        # bypasses the encoder, but the tensor must exist for the call signature.
        dummy_cond = torch.zeros(
            cfg.gen_batch_size,
            cfg.n_states,
            cfg.n_score_components if cfg.guidance_mode == "score_based" else cfg.s_param_n_freq,
            device=self.device,
        )

        self.gen_model.train()
        for step in range(cfg.warmup_random_feasible_train_steps):
            idx   = torch.randperm(len(all_images), device=self.device)[:cfg.gen_batch_size]
            batch = all_images[idx]
            B     = batch.shape[0]
            self.gen_opt.zero_grad()
            loss = self.gen_model.train_step(
                batch,
                dummy_cond[:B],
                force_uncond=True,
            )
            loss.backward()
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
        Return the conditioning tensor for diffusion model sampling.

        In 's_param' mode  : returns (B, n_states, n_freq) — the ideal target S-params.
        In 'score_based' mode: returns (B, n_states, 4) — gradual score target.
        """
        cfg = self.cfg
        if cfg.guidance_mode == "score_based":
            if self._best_scores is None:
                # No real data yet — use all-zeros (perfect design) as a placeholder
                target = torch.zeros(
                    cfg.n_states, cfg.n_score_components, device=self.device
                )
            else:
                # Guide toward a slightly better score than the current best
                target = self._best_scores.to(self.device) * cfg.score_guidance_improvement_factor
            return target.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            return self.target_s.unsqueeze(0).expand(batch_size, -1, -1)

    def _get_score_cond_from_entries(self, entries) -> torch.Tensor:
        """
        Compute score-based conditioning tensors from a list of buffer entries.

        Feasible entries get their per-region score tensor computed from s_params.
        Infeasible entries get a fixed large "infeasible" label (all components = 100)
        that is clearly distinct from any physically realizable design's scores.
        This ensures the model learns a dedicated conditioning cluster for infeasible
        designs, rather than inheriting the ambiguous values from the +1 dB sentinel.

        Returns (B, n_states, 4) — one row of four score components per entry.
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
