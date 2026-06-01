"""
Phase 2: iterated surrogate-guided fine-tuning (no RL).

Loop: sample per-state-ratcheted PV-score targets -> generate pool ->
rank by GA score (-soft_max cost on WM-predicted S21) -> soft-label SFT
conditioned on the target score vector -> periodic simulator recalibration
(updates per-state best, retrains surrogate, logs calibration) -> held-out
eval.  Visualizes the best generated design each iter and the best simulated
design at recalibration.  Flags reward hacking (surrogate score up, true flat).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval import run_eval  # noqa: E402
from src.dataset import LMDataset  # noqa: E402
from src.factory import (build_model, build_score_norm, build_specs,  # noqa: E402
                         build_surrogate, build_tokenizer, freq_points,
                         load_config)
from src.scoring import (PerStateBest, design_score,  # noqa: E402
                         design_score_matrix)
from src.simulator import run_simulator, run_simulator_dev  # noqa: E402
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
        n_iters, k_recal, m_recal, n_targets = 1, 1, 4, 2
    else:
        pool_mult = ic["POOL_MULTIPLIER"]
        steps_iter = ic["STEPS_PER_ITER"]
        n_iters = args.iterations or ic["N_ITERATIONS"]
        k_recal, m_recal = ic["K_RECAL"], ic["M_RECAL"]
        n_targets = ic["n_targets"]

    dataset = LMDataset()                       # cumulative REAL data
    per_state_best = PerStateBest(K)            # per-state, per-scalar best
    holdout = _targets(PerStateBest(K), n_targets, np.random.default_rng(999))
    surr_hist: List[float] = []
    true_hist: List[float] = []

    for it in range(n_iters):
        targets = _targets(per_state_best, n_targets, rng)   # (n,K,n_sc) norm

        # 1+2. generate pool & rank by GA score on WM-predicted S21
        pool_designs, pool_mean, pool_scores, pool_group = [], [], [], []
        for g in range(n_targets):
            spec = torch.tensor(
                np.repeat(targets[g][None], pool_mult, axis=0)
            ).float().to(device)
            seqs = model.generate_batch(
                spec, max_len=model.mcfg.max_seq_len,
                temperature=ic["temperature"], cfg_scale=ic["cfg_scale"],
                apply_constraints=True)
            for s in seqs:
                d = decode_to_design(tok, s)
                if d is None:
                    continue
                try:
                    mean, _ = surrogate.predict_multistate(d)
                except Exception:
                    continue
                pool_designs.append(d)
                pool_mean.append(mean)
                pool_scores.append(design_score(fpts, mean, specs, agg, temp))
                pool_group.append(g)
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
        for _ in range(steps_iter):
            bidx = rng.integers(0, len(pool_ex),
                                size=min(cfg["sft"]["batch_size"], len(pool_ex)))
            batch = collate([pool_ex[int(i)] for i in bidx], tok, specs, fpts,
                            score_norm, K, device,
                            spec_vectors=pool_tvec[bidx])
            sw = torch.tensor(weights[bidx])
            sft_step(model, batch, type_w, opt, sample_weights=sw)

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

        # 4. periodic simulator recalibration
        if it % k_recal == 0:
            _, pool_var = surrogate.predict_multistate_batch(pool_designs)
            unc = np.array([np.mean(v) for v in pool_var])
            selidx = select_for_simulation(pool_scores, unc, m_recal,
                                           ic["explore_frac"], rng)
            to_sim = [pool_designs[i] for i in selidx]
            if cfg["data"].get("use_surrogate_sim", True) or args.smoke:
                true_sp = run_simulator_dev(to_sim, surrogate, fpts, seed=it)
            else:
                true_sp = run_simulator(to_sim)              # real EM (TODO)
            dataset.add_relabeled(to_sim, true_sp, iteration=it)
            for spm in true_sp:                              # per-state ratchet
                per_state_best.update(
                    design_score_matrix(fpts, spm.s21_db, specs))
            surrogate.retrain(dataset.real_only(), epochs=1)

            preds = [surrogate.predict_multistate(d)[0] for d in to_sim]
            P = np.concatenate([p.ravel() for p in preds])
            Tr = np.concatenate([s.s21_db.ravel() for s in true_sp])
            ss_res = float(np.sum((P - Tr) ** 2))
            ss_tot = float(np.sum((Tr - Tr.mean()) ** 2)) + 1e-9
            metrics["calib_mse"] = float(np.mean((P - Tr) ** 2))
            metrics["calib_r2"] = 1.0 - ss_res / ss_tot
            true_scores = [design_score(fpts, s.s21_db, specs, agg, temp)
                           for s in true_sp]
            metrics["mean_true_score"] = float(np.mean(true_scores))
            metrics["best_true_score"] = float(np.max(true_scores))
            surr_hist.append(metrics["mean_surrogate_score"])
            true_hist.append(metrics["mean_true_score"])

            if viz_on:
                jt = int(np.argmax(true_scores))
                plot_best_simulated(
                    to_sim[jt], true_sp[jt].s21_db, specs, fpts, it, out_dir,
                    achieved=design_score_matrix(
                        fpts, true_sp[jt].s21_db, specs))

        # 5. held-out eval
        ev = run_eval(model, surrogate, tok, holdout, specs, fpts,
                      n_per_target=max(2, pool_mult // 2),
                      cfg_scale=ic["cfg_scale"], agg=agg, temp=temp)
        metrics.update({f"eval_{k}": v for k, v in ev.items()})

        # reward-hacking warning
        if len(surr_hist) >= 3 and len(true_hist) >= 3:
            if (surr_hist[-1] > surr_hist[-3]) and \
               (true_hist[-1] <= true_hist[-3] + 1e-6):
                metrics["REWARD_HACK_WARNING"] = 1.0

        _log(cfg["logging"]["backend"], it, metrics)

    print("done.")


if __name__ == "__main__":
    main()
