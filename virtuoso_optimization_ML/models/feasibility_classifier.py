"""
feasibility_classifier.py — Binary CNN classifier for design feasibility.

The classifier is trained on:
  - Feasible designs sampled from the replay buffer  (label = 1)
  - Infeasible decoded images collected during Step 1 (label = 0)

It provides two complementary uses:

1. **Training guidance** (flow_matching.train_step):
   The estimated x1 from the current velocity prediction is scored by the
   classifier; the log-probability of feasibility is added as an auxiliary
   loss, pushing the generative model toward feasible regions.

2. **Sampling guidance** (flow_matching.sample):
   At each Euler integration step the gradient of log P(feasible | x_t)
   w.r.t. x_t is added to the velocity field, steering the ODE trajectory
   into feasible regions.

Architecture
------------
Shares the same CNN encoder backbone as WorldModel and RNDNetwork
(via _build_cnn_encoder from world_model.py), followed by a single
linear head with sigmoid output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .world_model import _build_cnn_encoder, _FLAT_DIM, _LATENT_DIM


class FeasibilityClassifier(nn.Module):
    """
    Binary classifier that estimates P(feasible | design_image).

    Parameters
    ----------
    (none — architecture is fixed to match the WorldModel encoder)
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem, self.encoder, self.pool = _build_cnn_encoder()
        self.head = nn.Sequential(
            nn.Linear(_FLAT_DIM, _LATENT_DIM),
            nn.GELU(),
            nn.Linear(_LATENT_DIM, 1),   # raw logit
        )

    def _logits(self, x: Tensor) -> Tensor:
        """
        Compute raw (pre-sigmoid) logit for each sample.

        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        (B,) logit — positive = likely feasible.
        """
        B = x.shape[0]
        h = self.stem(x)
        h = self.encoder(h)
        h = self.pool(h)
        h = h.reshape(B, -1)
        return self.head(h).squeeze(-1)   # (B,)

    def classify(self, x: Tensor) -> Tensor:
        """
        Estimate feasibility probability for each sample.

        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        (B,) probability in [0, 1].  Higher = more likely feasible.
        """
        return torch.sigmoid(self._logits(x))

    def train_step(
        self,
        feasible_x:   Tensor,
        infeasible_x: Tensor,
    ) -> Tensor:
        """
        Binary cross-entropy loss over a balanced feasible / infeasible batch.

        Parameters
        ----------
        feasible_x   : (N, 1, H, W+S)  — images of feasible designs.
        infeasible_x : (M, 1, H, W+S)  — images of infeasible designs.

        Returns
        -------
        Scalar BCE loss.
        """
        x      = torch.cat([feasible_x, infeasible_x], dim=0)
        labels = torch.cat([
            torch.ones(len(feasible_x),   device=x.device),
            torch.zeros(len(infeasible_x), device=x.device),
        ])
        logits = self._logits(x)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def log_prob_gradient(self, x: Tensor) -> Tensor:
        """
        Compute ∂ log P(feasible | x) / ∂x for use as classifier guidance.

        Enables gradient-based steering of the ODE trajectory toward feasible
        regions during sampling.

        Parameters
        ----------
        x : (B, 1, H, W+S)   — must have requires_grad=True or be a leaf tensor.

        Returns
        -------
        (B, 1, H, W+S) gradient tensor (same shape as x), detached.
        """
        x_req = x.detach().requires_grad_(True)
        logits = self._logits(x_req)
        log_p  = F.logsigmoid(logits).sum()
        grad   = torch.autograd.grad(log_p, x_req)[0]
        return grad.detach()
