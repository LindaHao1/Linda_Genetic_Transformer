"""
Surrogate = wrapper around the pretrained world model.

Reuses (does NOT duplicate) world_model_test/encode.py + model.py.  The world
model is single-state, so a K-state filter is evaluated by encoding each state
and calling the ensemble K times (batched).
"""

from __future__ import annotations

import logging
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.types import Design

log = logging.getLogger(__name__)

_WM_DIR = "/disk/mcao53/RFIC_RL/world_model_test"
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)

import encode as wm_encode          # noqa: E402
from model import EnsembleModel, WorldModel  # noqa: E402


def neg_mse_db(pred: np.ndarray, target: np.ndarray) -> float:
    """Default closeness score: negative MSE in dB (higher is better)."""
    return float(-np.mean((pred - target) ** 2))


class Surrogate:
    def __init__(
        self,
        encode_config: dict,
        checkpoint_path: Optional[str] = None,
        ensemble_size: int = 5,
        n_freq: int = 71,
        device: str = "cpu",
        score_fn: Callable[[np.ndarray, np.ndarray], float] = neg_mse_db,
    ) -> None:
        self.cfg = wm_encode.build_encode_config(encode_config)
        self.device = torch.device(device)
        self.score_fn = score_fn
        self.n_freq = n_freq

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu",
                              weights_only=False)
            self.n_freq = ckpt["n_freq"]
            members = []
            for sd in ckpt["member_states"]:
                m = WorldModel(n_freq=self.n_freq)
                m.load_state_dict(sd)
                members.append(m)
            log.info("Loaded %d-member ensemble from %s",
                     len(members), checkpoint_path)
        else:
            members = [WorldModel(n_freq=n_freq) for _ in range(ensemble_size)]
            log.warning(
                "No surrogate checkpoint provided -- using %d UNTRAINED "
                "members. Predictions are meaningless; smoke-test only. "
                "TODO(user): set surrogate.checkpoint_path in config.",
                ensemble_size)
        if len(members) == 1:
            log.warning("Single-member ensemble -- epistemic variance is 0; "
                        "uncertainty-based selection disabled.")
        self.ensemble = EnsembleModel(members).to(self.device).eval()

    # ---- encoding --------------------------------------------------------

    def _encode_states(self, design: Design) -> torch.Tensor:
        """Return (K, 3, H, W) float32 tensor for all K states of `design`."""
        rec = design.to_record()
        K = design.n_states
        imgs = [wm_encode.encode_sample(rec, s, self.cfg) for s in range(K)]
        return torch.from_numpy(np.stack(imgs, axis=0)).float()

    # ---- prediction ------------------------------------------------------

    def predict(self, design: Design, state_idx: int) -> np.ndarray:
        """Mean predicted S21 (dB) for one state. shape (n_freq,)."""
        rec = design.to_record()
        img = torch.from_numpy(
            wm_encode.encode_sample(rec, state_idx, self.cfg)
        ).float().unsqueeze(0).to(self.device)
        mean, _ = self.ensemble.predict(img)
        return mean[0].cpu().numpy()

    def predict_multistate(self, design: Design
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """(mean (K,n_freq), epistemic_var (K,n_freq))."""
        x = self._encode_states(design).to(self.device)
        mean, var = self.ensemble.predict(x)
        return mean.cpu().numpy(), var.cpu().numpy()

    def predict_multistate_batch(
        self, designs: Sequence[Design]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Per-design (mean, var) lists; designs may have differing K."""
        means, vars = [], []
        # group encodings into one big batch, then split back
        chunks, ks = [], []
        for d in designs:
            x = self._encode_states(d)
            chunks.append(x)
            ks.append(x.shape[0])
        big = torch.cat(chunks, dim=0).to(self.device)
        m, v = self.ensemble.predict(big)
        m = m.cpu().numpy()
        v = v.cpu().numpy()
        off = 0
        for k in ks:
            means.append(m[off:off + k])
            vars.append(v[off:off + k])
            off += k
        return means, vars

    # ---- scoring ---------------------------------------------------------

    def score(self, pred_sparams: np.ndarray, target_sparams: np.ndarray) -> float:
        """LEGACY curve-closeness (default negative MSE in dB).

        Candidate ranking now uses GA PV-contrast scoring
        (`src/scoring.py:design_score`) on predicted S21 + FilterStateSpec —
        see scripts/train_iterated.py. This is kept only for ad-hoc curve
        comparisons / back-compat.
        """
        return self.score_fn(np.asarray(pred_sparams), np.asarray(target_sparams))

    # ---- recalibration ---------------------------------------------------

    def retrain(self, examples, epochs: int = 1, lr: float = 3e-4,
                batch_size: int = 64) -> None:
        """
        Fine-tune ensemble members on real (image, S21) pairs.

        Minimal but real: each member does next-epoch MSE SGD on the cumulative
        real dataset.  `examples` is an iterable of TrainingExample with
        source == 'simulated'.  For large-scale retraining prefer
        world_model_test/train.py (TODO(user)).
        """
        data = []
        for ex in examples:
            rec = ex.design.to_record()
            for s in range(ex.design.n_states):
                img = wm_encode.encode_sample(rec, s, self.cfg)
                data.append((img, ex.sparams.s21_db[s]))
        if not data:
            log.warning("retrain: no simulated examples; skipping")
            return
        X = torch.from_numpy(np.stack([d[0] for d in data])).float()
        Y = torch.from_numpy(np.stack([d[1] for d in data])).float()
        for m in self.ensemble.members:
            m.train()
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
            for _ in range(epochs):
                perm = torch.randperm(len(X))
                for i in range(0, len(X), batch_size):
                    idx = perm[i:i + batch_size]
                    opt.zero_grad()
                    loss = m.train_step(X[idx].to(self.device),
                                        Y[idx].to(self.device))
                    loss.backward()
                    opt.step()
            m.eval()
