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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml

from virtuoso_optimization_SA.cost import specs_from_config, compute_total_cost
from virtuoso_optimization_SA.feasibility import is_feasible

from ..dataset.replay_buffer import ReplayBuffer, make_infeasible_sentinel
from ..encoding.design_encoder import DesignEncoder, EncoderConfig
from ..models.flow_matching import FlowMatchingModel
from ..models.world_model import EnsembleWorldModel
from ..training.warmup import run_warmup
from ..utils.s_param_utils import (
    build_target_s_param_tensor,
    result_to_s_param_tensor,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLConfig:
    # Grid / design space
    rows:              int
    cols:              int
    n_switches:        int
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
    wm_n_members:             int
    wm_lr:                    float
    wm_batch_size:            int
    wm_std_scale:             float
    wm_confidence_threshold:  float
    wm_use_bagging:           bool
    wm_correction_interval:   int

    # Replay buffer
    buf_max_size:           int
    buf_alpha_recency:      float
    buf_alpha_real:         float
    buf_real_bonus:         float
    buf_decay_rate:         float
    buf_infeasible_weight:  float

    # Training loop
    warmup_n_designs:      int
    n_iterations:          int
    n_gen_samples:         int
    n_sim_max:             int
    gen_train_steps:       int
    wm_train_steps:        int
    wm_correction_n:       int
    max_resample_attempts: int

    # Paths
    virtuoso_config: str
    output_dir:      str

    # Parallel simulation
    parallel_num_workers: int

    # Misc
    checkpoint_interval: int
    seed:                Optional[int] = None

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

            wm_n_members            = int(wm.get("n_members",            5)),
            wm_lr                   = float(wm.get("lr",                  1e-4)),
            wm_batch_size           = int(wm.get("batch_size",            32)),
            wm_std_scale            = float(wm.get("std_scale",           3.0)),
            wm_confidence_threshold = float(wm.get("confidence_threshold",0.7)),
            wm_use_bagging          = bool(wm.get("use_bagging",          True)),
            wm_correction_interval  = int(wm.get("correction_interval",  10)),

            buf_max_size          = int(rb.get("max_size",           10000)),
            buf_alpha_recency     = float(rb.get("alpha_recency",      0.6)),
            buf_alpha_real        = float(rb.get("alpha_real",         0.4)),
            buf_real_bonus        = float(rb.get("real_bonus",         2.0)),
            buf_decay_rate        = float(rb.get("decay_rate",         0.05)),
            buf_infeasible_weight = float(rb.get("infeasible_weight",  0.1)),

            warmup_n_designs      = int(tr.get("warmup_n_designs",        100)),
            n_iterations          = int(tr.get("n_iterations",            500)),
            n_gen_samples         = int(tr.get("n_gen_samples",            20)),
            n_sim_max             = int(tr.get("n_sim_max",                 8)),
            gen_train_steps       = int(tr.get("gen_train_steps",          10)),
            wm_train_steps        = int(tr.get("wm_train_steps",           10)),
            wm_correction_n       = int(tr.get("wm_correction_n",           5)),
            max_resample_attempts = int(tr.get("max_resample_attempts",     5)),

            virtuoso_config      = str(d["virtuoso_config"]),
            output_dir           = str(d["output_dir"]),

            parallel_num_workers = int(par.get("num_workers", 4)),
            checkpoint_interval  = int(d.get("checkpoint_interval", 50)),
            seed                 = d.get("seed"),
            filter_specs_raw     = list(d.get("filter_specs", [])),
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
            n_states      = cfg.n_states,
            n_freq        = cfg.s_param_n_freq,
            img_h         = img_h,
            img_w         = img_w,
            cond_dim      = cfg.gen_cond_dim,
            t_dim         = cfg.gen_t_dim,
            base_channels = cfg.gen_base_channels,
            p_uncond      = cfg.gen_p_uncond,
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

        # Step 0: Warmup
        n_warm = run_warmup(
            cfg         = cfg,
            encoder     = self.encoder,
            buffer      = self.buffer,
            freq_points = self.freq_points,
            rng         = self.rng,
            output_dir  = self.output_dir,
        )
        if n_warm == 0:
            print("[ML] WARNING: Warmup produced no real data — generative model "
                  "will start from scratch.")

        # Initial world model and generative model warm-up training
        if n_warm > 0:
            print("[ML] Pre-training world model on warmup data ...")
            for _ in range(50):
                self._step4_train_world_model()
            print("[ML] Pre-training generative model on warmup data ...")
            for _ in range(50):
                self._step3_train_generative(current_iter=0)

        start_time = time.time()

        for it in range(cfg.n_iterations):
            iter_start = time.time()

            # Step 1: Sample designs from generative model
            feasible_designs, infeasible_images = self._step1_sample_designs(
                n=cfg.n_gen_samples, current_iter=it
            )

            # Add infeasible designs to buffer with sentinel label
            sentinel = make_infeasible_sentinel(cfg.n_states, cfg.s_param_n_freq)
            for img in infeasible_images:
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

            # Step 4: Train world model
            wm_loss = self._step4_train_world_model()

            # Step 5: Periodic correction
            n_corrected = 0
            if it > 0 and it % cfg.wm_correction_interval == 0:
                n_corrected = self._step5_correction(it)

            # Checkpoint
            if it % cfg.checkpoint_interval == 0:
                self._checkpoint(it)

            elapsed = time.time() - iter_start
            total   = time.time() - start_time
            info = {
                "iteration":   it,
                "gen_loss":    gen_loss,
                "wm_loss":     wm_loss,
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
    # Step 1: Sample designs from generative model
    # ------------------------------------------------------------------

    def _step1_sample_designs(
        self, n: int, current_iter: int
    ) -> Tuple[List, List]:
        """
        Sample n designs from the flow matching model.

        Returns
        -------
        feasible_designs : list of (DesignState, image_tensor) — feasible
        infeasible_images: list of image tensors — could not be made feasible
        """
        cfg = self.cfg
        target_s = self.target_s.unsqueeze(0).expand(n, -1, -1)  # (n, S, F)

        # Generate raw images
        self.gen_model.eval()
        with torch.no_grad():
            raw_imgs = self.gen_model.sample(
                target_s        = target_s,
                n_steps         = cfg.gen_n_steps,
                guidance_scale  = cfg.gen_guidance_scale,
            )  # (n, 1, H, W)
        self.gen_model.train()

        feasible_designs: List = []
        infeasible_images: List = []

        for i in range(n):
            img = raw_imgs[i]     # (1, H, W)
            design = None

            # Try re-sampling to get a feasible decode
            for attempt in range(cfg.max_resample_attempts):
                try:
                    candidate = self.encoder.decode(img)
                    if is_feasible(
                        candidate,
                        n_switches    = cfg.n_switches,
                        max_inductors = cfg.max_inductors,
                    ):
                        design = candidate
                        break
                except Exception:
                    pass

                # Resample if still infeasible
                if attempt < cfg.max_resample_attempts - 1:
                    with torch.no_grad():
                        new_img = self.gen_model.sample(
                            target_s       = self.target_s.unsqueeze(0),
                            n_steps        = cfg.gen_n_steps,
                            guidance_scale = cfg.gen_guidance_scale,
                        )  # (1, 1, H, W)
                    img = new_img[0]

            if design is not None:
                feasible_designs.append((design, img.cpu()))
            else:
                infeasible_images.append(img.cpu())

        return feasible_designs, infeasible_images

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

        high_conf_idx = []
        low_conf_idx  = []
        for i, conf in enumerate(conf_score.tolist()):
            if conf >= cfg.wm_confidence_threshold:
                high_conf_idx.append(i)
            else:
                low_conf_idx.append(i)

        # High-confidence → add to buffer with world model label
        for i in high_conf_idx:
            _, img = feasible_designs[i]
            self.buffer.add(
                image       = img,
                s_params    = mean_pred[i].cpu(),
                is_real     = False,
                is_feasible = True,
                iteration   = current_iter,
                confidence  = conf_score[i].item(),
            )

        # Low-confidence → simulate (up to n_sim_max)
        to_simulate = [feasible_designs[i] for i in low_conf_idx[:cfg.n_sim_max]]
        n_sim = self._simulate_and_add(to_simulate, current_iter, tag=f"iter{current_iter}")
        return n_sim

    def _simulate_and_add(
        self,
        design_img_pairs: List,
        current_iter: int,
        tag: str,
    ) -> int:
        """Simulate a batch of designs and add real results to buffer."""
        if not design_img_pairs:
            return 0

        from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine  # deferred

        sim_dir   = self.output_dir / "iter_sims" / tag
        overrides = []
        for i, (design, _) in enumerate(design_img_pairs):
            overrides.append({
                "mode":      "pixelated_design_simple",
                "pixelated": design.to_pixelated_config(),
                "project":   {"parent_output_dir": str(sim_dir / f"cand{i:04d}")},
            })

        with ParallelVirtuosoEngine(
            config_path = self.cfg.virtuoso_config,
            num_workers = self.cfg.parallel_num_workers,
            log_path    = self.output_dir / "iter_engine.log",
        ) as engine:
            results = engine.run_batch(overrides)

        n_ok = 0
        for (design, img), result in zip(design_img_pairs, results):
            s_params = result_to_s_param_tensor(result, self.freq_points, self.cfg.n_states)
            if s_params is None:
                continue

            # Track best cost
            cost = compute_total_cost(result.s_param_files, self.specs)
            if cost < self._best_cost:
                self._best_cost   = cost
                self._best_design = design
                self._save_best(design, cost, current_iter)

            self.buffer.add(
                image       = img,
                s_params    = s_params,
                is_real     = True,
                is_feasible = True,
                iteration   = current_iter,
                confidence  = 0.0,
            )
            n_ok += 1

        return n_ok

    # ------------------------------------------------------------------
    # Step 3: Train generative model
    # ------------------------------------------------------------------

    def _step3_train_generative(self, current_iter: int) -> float:
        cfg = self.cfg
        total_loss = 0.0
        n_steps    = 0

        self.gen_model.train()
        for _ in range(cfg.gen_train_steps):
            entries = self.buffer.sample_for_generative(cfg.gen_batch_size, current_iter)
            if not entries:
                break

            images   = torch.stack([e.image    for e in entries]).to(self.device)
            s_params = torch.stack([e.s_params for e in entries]).to(self.device)

            self.gen_opt.zero_grad()
            loss = self.gen_model.train_step(images, s_params)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), 1.0)
            self.gen_opt.step()

            total_loss += loss.item()
            n_steps    += 1

        return total_loss / max(n_steps, 1)

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

            total_loss += loss.item()
            n_steps    += 1

        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Step 5: Correction — re-simulate high-confidence entries
    # ------------------------------------------------------------------

    def _step5_correction(self, current_iter: int) -> int:
        cfg = self.cfg
        entries = self.buffer.sample_high_confidence_real(
            n                    = cfg.wm_correction_n,
            confidence_threshold = cfg.wm_confidence_threshold,
        )
        if not entries:
            return 0

        from virtuoso_engine.parallel_engine import ParallelVirtuosoEngine  # deferred

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
        with ParallelVirtuosoEngine(
            config_path = cfg.virtuoso_config,
            num_workers = cfg.parallel_num_workers,
            log_path    = self.output_dir / "correction_engine.log",
        ) as engine:
            results = engine.run_batch(sim_overrides)

        # Fine-tune world model on corrected data
        correction_images   = []
        correction_s_params = []
        n_corrected = 0

        for (i, entry_id, _), result in zip(overrides, results):
            s_params = result_to_s_param_tensor(result, self.freq_points, cfg.n_states)
            if s_params is None:
                continue

            # Update buffer entry
            self.buffer.update_entry(
                entry_id   = entry_id,
                s_params   = s_params,
                is_real    = True,
                confidence = 0.0,
            )

            correction_images.append(entries[i].image)
            correction_s_params.append(s_params)
            n_corrected += 1

        # Fine-tune world model on corrected entries
        if correction_images:
            imgs   = torch.stack(correction_images).to(self.device)
            sparams = torch.stack(correction_s_params).to(self.device)
            self.world_model.train()
            for _ in range(5):
                self.wm_opt.zero_grad()
                loss = self.world_model.train_step(imgs, sparams)
                loss.backward()
                self.wm_opt.step()

        return n_corrected

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

    def _checkpoint(self, it: int) -> None:
        ckpt_dir = self.output_dir / "checkpoints" / f"iter{it:05d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.gen_model.state_dict(),   ckpt_dir / "gen_model.pt")
        torch.save(self.world_model.state_dict(), ckpt_dir / "world_model.pt")
        torch.save(self.gen_opt.state_dict(),     ckpt_dir / "gen_opt.pt")
        torch.save(self.wm_opt.state_dict(),      ckpt_dir / "wm_opt.pt")
        self.buffer.save(ckpt_dir / "buffer.pkl")
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
        print(f"[ML] Resumed from checkpoint {ckpt}")
