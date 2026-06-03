"""
Phase 2: iterated surrogate-guided fine-tuning (no RL).

Loop: sample per-state-ratcheted PV-score targets -> generate pool ->
rank by GA score (-soft_max cost on WM-predicted S21) -> soft-label SFT
conditioned on the target score vector -> ASYNC simulator recalibration ->
held-out eval.  Visualizes the best generated design each iter and the best
simulated design when real results land.  Flags reward hacking (surrogate
score up, true flat).

Recalibration runs against the real parallel Virtuoso engine
(`AsyncSimManager`), started once before the loop and kept alive in the
background.  Every `K_RECAL` iterations a wave of up to `n_parallel` selected
designs is submitted non-blocking; whenever simulations finish (over the next
iterations) the realized (grid-snapped) designs + true S21 are collected and
used to retrain BOTH the surrogate AND the LM (hindsight SFT on real data).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval import run_eval  # noqa: E402
from src.constraints import ConstraintConfig, design_is_feasible  # noqa: E402
from src.dataset import LMDataset  # noqa: E402
from src.factory import (build_model, build_score_norm, build_specs,  # noqa: E402
                         build_surrogate, build_tokenizer, freq_points,
                         load_config)
from src.scoring import (SCORE_SCALARS, PerStateBest,  # noqa: E402
                         design_score, design_score_matrix)
from src.simulator import AsyncSimManager  # noqa: E402
from src.training import (collate, compute_soft_weights,  # noqa: E402
                          decode_to_design, sample_score_targets,
                          select_for_simulation, sft_step, token_type_weights)
from src.types import SParams, TrainingExample  # noqa: E402
from src.visualize import plot_best_generated, plot_best_simulated  # noqa: E402


def _log(backend: str, it: int, metrics: Dict[str, float]) -> None:
    print(f"[iter {it}] " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
    # TODO(user): backend in {tensorboard, wandb} -> write metrics there.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sft-ckpt", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny pool/steps, K_RECAL=1 (pipeline dry run)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ic, sc = cfg["iterated"], cfg["spec"]
    agg = cfg["scoring"]["aggregation_mode"]
    temp = cfg["scoring"]["aggregation_temperature"]
    device = torch.device(cfg["surrogate"].get("device", "cpu"))
    out_dir = cfg["logging"]["dir"]
    viz_on = cfg.get("viz", {}).get("enabled", False)

    tok = build_tokenizer(cfg)
    model = build_model(cfg, tok).to(device)
    if args.sft_ckpt:
        model.load_state_dict(torch.load(args.sft_ckpt,
                                         map_location="cpu")["model"])
    surrogate = build_surrogate(cfg)
    specs = build_specs(cfg)
    score_norm = build_score_norm(cfg)
    fpts = freq_points(cfg)
    ccfg = ConstraintConfig.from_cfg(cfg)       # GA/SA design rules for generation
    K = cfg["grid"]["n_states"]
    type_w = token_type_weights(tok, cfg["sft"].get("type_weights"))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["sft"]["lr"]),
                            weight_decay=1e-4)
    rng = np.random.default_rng(0)

    def _targets(per_best, n, r):
        return sample_score_targets(per_best, K, n, r, sc["target_margins"],
                                    sc["bounds"], sc["cold_start_target"],
                                    sc["stretch_frac"], score_norm)

    if args.smoke:
        pool_mult, steps_iter = 4, 3
        n_iters, k_recal, n_targets = 2, 1, 2
        n_parallel, ground_steps = 2, 2
    else:
        pool_mult = ic["POOL_MULTIPLIER"]
        steps_iter = ic["STEPS_PER_ITER"]
        n_iters = args.iterations or ic["N_ITERATIONS"]
        k_recal = ic["K_RECAL"]
        n_targets = ic["n_targets"]
        n_parallel = ic["n_parallel"]
        ground_steps = ic.get("ground_steps_per_recal", 20)
    # `m_recal` (designs simulated per wave) is aligned to the live worker count
    # after the engine starts; see `m_recal = sim_mgr.n_workers` below.
    ground_replay = ic.get("ground_replay", True)

    dataset = LMDataset()                       # cumulative REAL data
    per_state_best = PerStateBest(K)            # per-state, per-scalar best
    holdout = _targets(PerStateBest(K), n_targets, np.random.default_rng(999))
    surr_hist: List[float] = []
    true_hist: List[float] = []

    # ---- recalibration helpers (close over the loop state) ---------------
    def _calib_metrics(designs, true_sp, metrics):
        """Surrogate-vs-true MSE/R2 on a batch (call BEFORE retrain)."""
        preds = [surrogate.predict_multistate(d)[0] for d in designs]
        P = np.concatenate([p.ravel() for p in preds])
        Tr = np.concatenate([s.s21_db.ravel() for s in true_sp])
        ss_res = float(np.sum((P - Tr) ** 2))
        ss_tot = float(np.sum((Tr - Tr.mean()) ** 2)) + 1e-9
        metrics["calib_mse"] = float(np.mean((P - Tr) ** 2))
        metrics["calib_r2"] = 1.0 - ss_res / ss_tot

    def _ground_lm(designs, true_sp, it):
        """Hindsight SFT on ground-truth designs: condition each design on its
        OWN achieved score (spec_vectors=None), uniform weight.  Grounds the LM
        in real physics, breaking the surrogate-only feedback loop."""
        fresh = [TrainingExample(d, sp, source="simulated", iteration_added=it)
                 for d, sp in zip(designs, true_sp)]
        pool = list(dataset.real_only()) if ground_replay else fresh
        if not pool:
            return
        state_rng = np.random.default_rng(10_000 + it)
        bs = cfg["sft"]["batch_size"]
        for _ in range(ground_steps):
            bidx = rng.integers(0, len(pool), size=min(bs, len(pool)))
            batch = collate([pool[int(i)] for i in bidx], tok, specs, fpts,
                            score_norm, K, device,
                            spec_vectors=None, state_rng=state_rng)
            sft_step(model, batch, type_w, opt)

    def _ingest_real(designs, true_sp, it, metrics):
        """Consume a collected batch of real (design, true S21) pairs: log
        calibration, grow the dataset, ratchet targets, retrain surrogate AND
        ground the LM."""
        if not designs:
            return
        _calib_metrics(designs, true_sp, metrics)           # predict-before-retrain
        dataset.add_relabeled(designs, true_sp, iteration=it)
        for spm in true_sp:                                 # per-state ratchet
            per_state_best.update(design_score_matrix(fpts, spm.s21_db, specs))
        surrogate.retrain(dataset.real_only(), epochs=1)    # world-model grounding
        _ground_lm(designs, true_sp, it)                    # LM grounding
        true_scores = [design_score(fpts, s.s21_db, specs, agg, temp)
                       for s in true_sp]
        metrics["mean_true_score"] = float(np.mean(true_scores))
        metrics["best_true_score"] = float(np.max(true_scores))
        surr_hist.append(metrics.get("mean_surrogate_score", float("nan")))
        true_hist.append(metrics["mean_true_score"])
        if viz_on:
            jt = int(np.argmax(true_scores))
            plot_best_simulated(
                designs[jt], true_sp[jt].s21_db, specs, fpts, it, out_dir,
                achieved=design_score_matrix(fpts, true_sp[jt].s21_db, specs))

    # per-iteration score record (JSONL): world-model best/per-state best each
    # iteration, plus the real-simulation best/per-state best whenever a wave
    # was collected that iteration.
    score_log_path = os.path.join(out_dir, "iteration_scores.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    def _write_scores(it, metrics, pool_scores, pool_mean):
        jb = int(np.argmax(pool_scores))
        rec = {
            "iter": int(it),
            "scalars": list(SCORE_SCALARS),
            "wm_best_score": float(pool_scores.max()),
            "wm_mean_score": float(pool_scores.mean()),
            # per-state, per-scalar GA score of the best WM-ranked design
            "wm_best_per_state": design_score_matrix(
                fpts, pool_mean[jb], specs).tolist(),
        }
        if "n_collected" in metrics:        # a real wave landed this iteration
            psb = per_state_best.best_matrix()
            rec["real"] = {
                "n_collected": metrics.get("n_collected"),
                "best_score": metrics.get("best_true_score"),
                "mean_score": metrics.get("mean_true_score"),
                "calib_r2": metrics.get("calib_r2"),
                # running per-state, per-scalar best from SIMULATED designs
                "per_state_best": (psb.tolist() if psb is not None else None),
            }
        with open(score_log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    sim_mgr = AsyncSimManager(
        engine_config_path=cfg["data"]["engine_config_path"],
        freq_hz=fpts, num_workers=n_parallel,
        output_root=os.path.join(out_dir, "sim_runs"),
        log_path=os.path.join(out_dir, "parallel_engine.log"),
        quiet=True)

    with sim_mgr:
        m_recal = sim_mgr.n_workers          # one wave fills all workers
        for it in range(n_iters):
            targets = _targets(per_state_best, n_targets, rng)   # (n,K,n_sc) norm

            # 1+2. generate the whole pool in ONE batched call (all targets x
            #     pool_mult at once -> far better CPU/GPU utilization than one
            #     call per target), then rank by GA score on WM-predicted S21.
            pool_designs, pool_mean, pool_scores, pool_group = [], [], [], []
            spec_all = torch.tensor(
                np.concatenate([np.repeat(targets[g][None], pool_mult, axis=0)
                                for g in range(n_targets)], axis=0)
            ).float().to(device)
            print(f"[iter {it}] generating pool B={spec_all.shape[0]} "
                  f"({n_targets} targets x {pool_mult}) ...", flush=True)
            seqs = model.generate_batch(
                spec_all, max_len=model.mcfg.max_seq_len,
                temperature=ic["temperature"], cfg_scale=ic["cfg_scale"],
                apply_constraints=True, constraint_config=ccfg)
            n_decoded = n_infeasible = 0
            for i, s in enumerate(seqs):
                d = decode_to_design(tok, s)
                if d is None:
                    continue
                n_decoded += 1
                if not design_is_feasible(d, ccfg):       # global rules (connectivity)
                    n_infeasible += 1
                    continue
                try:
                    mean, _ = surrogate.predict_multistate(d)
                except Exception:
                    continue
                pool_designs.append(d)
                pool_mean.append(mean)
                pool_scores.append(design_score(fpts, mean, specs, agg, temp))
                pool_group.append(i // pool_mult)        # target group of seq i
            print(f"[iter {it}] pool -> {len(pool_designs)}/{len(seqs)} feasible "
                  f"({n_decoded} decoded, {n_infeasible} infeasible)", flush=True)
            if not pool_designs:
                _log(cfg["logging"]["backend"], it, {"valid_pool": 0})
                continue
            pool_scores = np.array(pool_scores)

            # 3. soft-label weights per target group
            weights = np.zeros(len(pool_scores), dtype=np.float32)
            for g in range(n_targets):
                sel = [i for i, gg in enumerate(pool_group) if gg == g]
                if sel:
                    weights[sel] = compute_soft_weights(
                        pool_scores[sel], mode=ic["label_mode"],
                        temperature=ic["soft_label_temperature"],
                        top_frac=1.0 - ic.get("explore_frac", 0.2))

            # 3b. SFT on the pool, conditioned on the TARGET score vector
            pool_ex = [TrainingExample(d, SParams(fpts, m))
                       for d, m in zip(pool_designs, pool_mean)]
            pool_tvec = np.stack([targets[g] for g in pool_group])   # (P,K,n_sc)
            print(f"[iter {it}] SFT {steps_iter} steps on {len(pool_ex)} designs ...",
                  flush=True)
            log_every = max(1, steps_iter // 10)
            for st in range(steps_iter):
                bidx = rng.integers(0, len(pool_ex),
                                    size=min(cfg["sft"]["batch_size"], len(pool_ex)))
                batch = collate([pool_ex[int(i)] for i in bidx], tok, specs, fpts,
                                score_norm, K, device,
                                spec_vectors=pool_tvec[bidx])
                sw = torch.tensor(weights[bidx])
                out = sft_step(model, batch, type_w, opt, sample_weights=sw)
                if (st + 1) % log_every == 0 or st == 0:
                    print(f"[iter {it}]   SFT {st + 1}/{steps_iter} "
                          f"loss={out['loss']:.4f}", flush=True)

            metrics = {"valid_pool": len(pool_designs),
                       "mean_surrogate_score": float(pool_scores.mean()),
                       "best_surrogate_score": float(pool_scores.max())}

            # best generated design this iteration (WM-predicted S21 + scores)
            if viz_on:
                jb = int(np.argmax(pool_scores))
                g = pool_group[jb]
                plot_best_generated(
                    pool_designs[jb], pool_mean[jb], specs, fpts, it, out_dir,
                    achieved=design_score_matrix(fpts, pool_mean[jb], specs),
                    target=score_norm.denormalize(targets[g]))

            # 4a. recalibration trigger: SELECT + SUBMIT (non-blocking).  The
            #     loop never waits on simulation; designs run in the background.
            if it % k_recal == 0:
                _, pool_var = surrogate.predict_multistate_batch(pool_designs)
                unc = np.array([np.mean(v) for v in pool_var])
                selidx = select_for_simulation(pool_scores, unc, m_recal,
                                               ic["explore_frac"], rng)
                to_sim = [pool_designs[i] for i in selidx]
                if sim_mgr.n_pending < sim_mgr.n_workers:     # self-throttle
                    n_sub = sim_mgr.submit(to_sim, it)
                    metrics["n_submitted"] = float(n_sub)
                    print(f"[iter {it}] submitted {n_sub} designs to engine "
                          f"(pending={sim_mgr.n_pending})", flush=True)
                else:
                    metrics["sim_backlog_skip"] = float(sim_mgr.n_pending)

            # 4b. COLLECT finished simulations (every iteration) and use the
            #     realized designs + true S21 to ground BOTH surrogate and LM.
            collected = sim_mgr.collect()
            if collected:
                c_designs = [d for d, _, _ in collected]
                c_sp = [sp for _, sp, _ in collected]
                _ingest_real(c_designs, c_sp, it, metrics)
                metrics["n_collected"] = float(len(collected))
            metrics["n_pending"] = float(sim_mgr.n_pending)

            # 5. held-out eval (also generates -> can be slow on CPU)
            print(f"[iter {it}] held-out eval ...", flush=True)
            ev = run_eval(model, surrogate, tok, holdout, specs, fpts,
                          n_per_target=max(2, pool_mult // 2),
                          cfg_scale=ic["cfg_scale"], agg=agg, temp=temp,
                          constraint_config=ccfg)
            metrics.update({f"eval_{k}": v for k, v in ev.items()})

            # reward-hacking warning
            if len(surr_hist) >= 3 and len(true_hist) >= 3:
                if (surr_hist[-1] > surr_hist[-3]) and \
                   (true_hist[-1] <= true_hist[-3] + 1e-6):
                    metrics["REWARD_HACK_WARNING"] = 1.0

            _log(cfg["logging"]["backend"], it, metrics)
            _write_scores(it, metrics, pool_scores, pool_mean)

        # drain any in-flight simulations submitted near the end of the loop
        remaining = sim_mgr.drain()
        if remaining:
            c_designs = [d for d, _, _ in remaining]
            c_sp = [sp for _, sp, _ in remaining]
            drain_metrics: Dict[str, float] = {}
            _ingest_real(c_designs, c_sp, n_iters - 1, drain_metrics)
            drain_metrics["final_drain"] = float(len(remaining))
            _log(cfg["logging"]["backend"], n_iters, drain_metrics)

    print("done.")


if __name__ == "__main__":
    main()
