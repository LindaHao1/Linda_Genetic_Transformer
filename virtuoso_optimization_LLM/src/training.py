"""
Shared SFT helpers: token-type loss weighting, batching, soft-label SFT step.

Hindsight relabeling: the spec fed to the model for an example is that
example's OWN achieved S21 (`TrainingExample.sparams.s21_db`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.model import FilterLM
from src.scoring import (
    DIRECTION, SCORE_SCALARS, PerStateBest, ScoreNorm, design_score_matrix,
)
from src.tokenizer import Tokenizer
from src.types import Design, TrainingExample

# default token-type loss weights (prompt: pixel 1, value 3, state 3)
DEFAULT_TYPE_WEIGHTS = {
    "structural": 1.0,
    "pixel": 1.0, "pixel_cap_pair": 1.0,
    "value": 3.0, "indval": 3.0,
    "state": 3.0,
}


def token_type_weights(tok: Tokenizer,
                        overrides: Optional[Dict[str, float]] = None
                        ) -> torch.Tensor:
    w = dict(DEFAULT_TYPE_WEIGHTS)
    if overrides:
        w.update(overrides)
    vec = torch.ones(tok.vocab_size)
    for tid in range(tok.vocab_size):
        vec[tid] = w.get(tok.get_token_type(tid), 1.0)
    return vec


def spec_score_vector(ex: TrainingExample, specs, freqs: np.ndarray,
                      score_norm: ScoreNorm, K: int,
                      scalars: Sequence[str] = SCORE_SCALARS) -> np.ndarray:
    """Normalized (K, n_scalars) PV-score conditioning derived from the
    example's OWN achieved S21 (Phase-1 hindsight)."""
    raw = design_score_matrix(freqs, np.asarray(ex.sparams.s21_db),
                              specs, scalars)               # (K_ex, n_sc)
    out = np.zeros((K, len(scalars)), dtype=np.float32)
    kk = min(K, raw.shape[0])
    out[:kk] = score_norm.normalize(raw[:kk])
    return out


def collate(examples: Sequence[TrainingExample], tok: Tokenizer,
            specs, freqs: np.ndarray, score_norm: ScoreNorm, K: int,
            device: torch.device,
            scalars: Sequence[str] = SCORE_SCALARS,
            spec_vectors: Optional[np.ndarray] = None
            ) -> Dict[str, torch.Tensor]:
    """`spec_vectors` (B,K,n_sc, already normalized) overrides the hindsight
    spec — used in Phase 2 to condition on the TARGET score vector. When None,
    each example is conditioned on its OWN achieved scores (Phase-1 hindsight).
    """
    seqs = [tok.encode(ex.design.to_record()) for ex in examples]
    Tmax = max(len(s) for s in seqs)
    pad = tok.t("PAD")
    inp, tgt = [], []
    for s in seqs:
        s = s + [pad] * (Tmax - len(s))
        inp.append(s[:-1])
        tgt.append(s[1:])
    inp = torch.tensor(inp, device=device)
    tgt = torch.tensor(tgt, device=device)
    if spec_vectors is not None:
        spec = torch.tensor(np.asarray(spec_vectors, np.float32), device=device)
    else:
        spec = torch.tensor(
            np.stack([spec_score_vector(e, specs, freqs, score_norm, K, scalars)
                      for e in examples]),
            device=device)
    pad_mask = tgt.ne(pad)
    return {"input_ids": inp, "target_ids": tgt,
            "pad_mask": pad_mask, "spec": spec}


def sft_step(
    model: FilterLM,
    batch: Dict[str, torch.Tensor],
    type_w: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    sample_weights: Optional[torch.Tensor] = None,   # (B,) soft-label weights
) -> Dict[str, float]:
    """One next-token CE step with token-type + optional per-sample weighting.

    Returns total loss and per-token-type loss breakdown (floats)."""
    model.train()
    logits = model(batch["input_ids"], batch["spec"])      # (B,T,V)
    B, T, V = logits.shape
    tgt = batch["target_ids"]
    pad_mask = batch["pad_mask"].float()

    ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                         reduction="none").view(B, T)
    tw = type_w.to(logits.device)[tgt]                     # (B,T)
    w = tw * pad_mask
    if sample_weights is not None:
        w = w * sample_weights.to(logits.device)[:, None]
    loss = (ce * w).sum() / w.sum().clamp_min(1.0)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # per-type breakdown (unweighted CE, for logging)
    out = {"loss": float(loss.detach())}
    with torch.no_grad():
        types = np.array([model.tok.get_token_type(int(t))
                          for t in tgt.reshape(-1).tolist()])
        ce_flat = ce.reshape(-1).detach().cpu().numpy()
        pm = batch["pad_mask"].reshape(-1).cpu().numpy()
        for fam in ("pixel", "pixel_cap_pair", "value", "state",
                    "indval", "structural"):
            sel = (types == fam) & (pm > 0)
            if sel.any():
                out[f"loss_{fam}"] = float(ce_flat[sel].mean())
    return out


def decode_to_design(tok: Tokenizer, tokens) -> Optional[Design]:
    """Decode a generated token sequence to a Design; None if ungrammatical."""
    try:
        return Design.from_record(tok.decode(list(tokens)))
    except Exception:
        return None


def _cold_start_matrix(cold_start, K: int, n_sc: int) -> np.ndarray:
    arr = np.asarray(cold_start, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.tile(arr, (K, 1))
    return arr[:K, :n_sc]


def sample_score_targets(per_state_best: PerStateBest, K: int, n: int,
                         rng: np.random.Generator, margins: dict, bounds: dict,
                         cold_start, stretch_frac: float,
                         score_norm: ScoreNorm,
                         scalars: Sequence[str] = SCORE_SCALARS) -> np.ndarray:
    """(n, K, n_scalars) NORMALIZED target score vectors.

    Per-state, per-scalar ratchet from the running best (simulated designs
    only) plus a margin on "stretch" targets, with physical clamps:
      pv_pb_level <= bounds.pv_pb_level.max (0 dB, passive filter)
      pv_pb_std   >= bounds.pv_pb_std.min   (0)
    `cold_start` (raw) is used until any simulated design exists.
    """
    n_sc = len(scalars)
    base = per_state_best.best_matrix()
    if base is None:
        base = _cold_start_matrix(cold_start, K, n_sc)
    base = base.astype(np.float32)

    out = []
    for _ in range(n):
        stretch = rng.random() < stretch_frac
        tgt = base.copy()
        if stretch:
            for j, name in enumerate(scalars):
                m = margins.get(name, {})
                if DIRECTION.get(name, "max") == "max":
                    tgt[:, j] += float(m.get("add", 0.0))          # push up
                else:
                    if "mult" in m:
                        tgt[:, j] *= float(m["mult"])              # shrink
                    tgt[:, j] -= float(m.get("add", 0.0))          # push down
        for j, name in enumerate(scalars):
            b = bounds.get(name, {})
            if "max" in b:
                tgt[:, j] = np.minimum(tgt[:, j], float(b["max"]))
            if "min" in b:
                tgt[:, j] = np.maximum(tgt[:, j], float(b["min"]))
        out.append(score_norm.normalize(tgt).astype(np.float32))
    return np.stack(out)


def select_for_simulation(scores: np.ndarray, uncertainty: np.ndarray,
                          m: int, explore_frac: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Top-(1-explore_frac)*m by surrogate score + explore_frac*m by highest
    epistemic uncertainty (for surrogate recalibration)."""
    n = len(scores)
    m = min(m, n)
    n_explore = int(round(explore_frac * m))
    n_exploit = m - n_explore
    by_score = np.argsort(scores)[::-1]
    chosen = list(by_score[:n_exploit])
    rest = [i for i in range(n) if i not in set(chosen)]
    if rest and n_explore:
        unc = np.asarray(uncertainty)[rest]
        chosen += list(np.array(rest)[np.argsort(unc)[::-1][:n_explore]])
    return np.array(sorted(set(chosen)), dtype=int)


def compute_soft_weights(scores: np.ndarray, mode: str = "soft_weighted",
                         temperature: float = 0.1,
                         top_frac: float = 0.2) -> np.ndarray:
    """Per-candidate SFT weights within one target-spec group.

    mode: soft_weighted | rejection_topk | rejection_topp."""
    s = np.asarray(scores, dtype=np.float64)
    if mode == "soft_weighted":
        z = (s - s.max()) / max(temperature, 1e-8)
        w = np.exp(z)
        return (w / w.sum() * len(w)).astype(np.float32)
    if mode == "rejection_topk":
        k = max(1, int(round(top_frac * len(s))))
        thr = np.sort(s)[::-1][k - 1]
        return (s >= thr).astype(np.float32)
    if mode == "rejection_topp":
        order = np.argsort(s)[::-1]
        keep = order[:max(1, int(round(top_frac * len(s))))]
        w = np.zeros_like(s, dtype=np.float32)
        w[keep] = 1.0
        return w
    raise ValueError(f"unknown label mode {mode!r}")
