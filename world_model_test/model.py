"""
Deterministic CNN surrogate (world model) + deep ensemble for epistemic uncertainty.

Single member:
  Input:  (B, 3, H, W)
  Output: (B, n_freq)  — predicted S21 mean (dB), trained with MSE

Ensemble:
  Wraps N independently-trained members.
  predict() returns (ensemble_mean, epistemic_variance) — variance across members,
  reflecting how much the models disagree, i.e. how well this region of design
  space is covered by training data.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.gelu(out + self.skip(x))


# ---------------------------------------------------------------------------
# Single deterministic member
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """
    Single CNN surrogate member. Outputs a deterministic S21 prediction.
    Trained with MSE — no stochastic loss term, since simulations are deterministic.

    Args:
        n_freq: Number of output frequency points.
    """

    _POOL_SIZE = 4
    _FLAT_DIM = 256 * 4 * 4  # = 4096 after AdaptiveAvgPool2d(4,4)
    _LATENT_DIM = 512

    def __init__(self, n_freq: int) -> None:
        super().__init__()
        self.n_freq = n_freq

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            _ResBlock(64, 128, stride=2),
            _ResBlock(128, 256, stride=2),
            _ResBlock(256, 256, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((self._POOL_SIZE, self._POOL_SIZE))
        self.latent_mlp = nn.Sequential(
            nn.Linear(self._FLAT_DIM, self._LATENT_DIM),
            nn.GELU(),
        )
        self.head = nn.Linear(self._LATENT_DIM, n_freq)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, n_freq) predicted S21 in dB
        """
        h = self.stem(x)
        h = self.encoder(h)
        h = self.pool(h).flatten(1)
        h = self.latent_mlp(h)
        return self.head(h)

    def train_step(self, x: Tensor, y_true: Tensor) -> Tensor:
        """MSE loss. Returns scalar."""
        return F.mse_loss(self(x), y_true)


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

class EnsembleModel(nn.Module):
    """
    Deep ensemble of N independently-trained WorldModel members.

    Epistemic uncertainty is estimated from the variance of member predictions:
    high inter-member disagreement indicates the design is poorly covered by
    training data (out-of-distribution), not that the simulation is noisy.

    Args:
        members: List of trained WorldModel instances.
        std_scale: Scale factor for confidence scoring.
    """

    def __init__(self, members: List[WorldModel], std_scale: float = 1.0) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.std_scale = std_scale

    @property
    def n_members(self) -> int:
        return len(self.members)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            mean:              (B, n_freq) ensemble mean prediction
            epistemic_variance:(B, n_freq) variance across members
        """
        # Stack member predictions: (n_members, B, n_freq)
        preds = torch.stack([m(x) for m in self.members], dim=0)
        mean = preds.mean(dim=0)
        epistemic_variance = preds.var(dim=0, unbiased=True)
        return mean, epistemic_variance

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Same as forward but with no_grad. Returns (mean, epistemic_variance)."""
        return self(x)

    @torch.no_grad()
    def confidence(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns (mean_pred, conf_score).
        conf_score = exp(-mean(epistemic_std) / std_scale) in [0, 1].
        High score → members agree → design is well-covered by training data.
        Low score  → high inter-member disagreement → likely OOD.
        """
        mean, variance = self(x)
        mean_std = variance.sqrt().mean(dim=-1)   # (B,)
        score = torch.exp(-mean_std / self.std_scale)
        return mean, score
