"""
world_model.py — Ensemble surrogate model that predicts S-parameters from
a pixel design image and provides a confidence score for active learning.

Architecture (per ensemble member)
------------------------------------
Input : (B, 1, H, W+S)   — encoded design image
Steps :
  Conv2d(1→64, 3×3) → GroupNorm → GELU
  ResBlock(64→128,  stride=2)
  ResBlock(128→256, stride=2)
  ResBlock(256→256, stride=1)   ← no 3rd downsample; keeps spatial dims ≥ 4×4
  AdaptiveAvgPool2d(4, 4) → flatten → Linear(4096, 512) → GELU
  Linear(512, n_states * n_freq)  → reshape to (B, n_states, n_freq)

Confidence
----------
Each member independently predicts the S-parameters.  The ensemble mean
gives the prediction; the ensemble std across members provides uncertainty.

  conf_score = exp( -mean(std_pred) / std_scale )

where mean(std_pred) is averaged over all (state, freq) positions.
High score → low uncertainty → high confidence → relabel without simulation.

Bagging
-------
When use_bagging=True, each member in train_step() trains on a random 80%
subset of the batch to improve member diversity.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """2-D residual block with optional stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.gn2   = nn.GroupNorm(groups, out_ch)

        self.skip: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


# ---------------------------------------------------------------------------
# Single ensemble member
# ---------------------------------------------------------------------------

class WorldModelMember(nn.Module):
    """
    One member of the ensemble world model.

    Parameters
    ----------
    img_h       : image height (rows).
    img_w_total : image width (cols + n_states).
    n_states    : number of filter states.
    n_freq      : number of frequency points in canonical grid.
    """

    def __init__(
        self,
        img_h:       int,
        img_w_total: int,
        n_states:    int,
        n_freq:      int,
    ) -> None:
        super().__init__()
        self.n_states = n_states
        self.n_freq   = n_freq

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            _ResBlock(64,  128, stride=2),
            _ResBlock(128, 256, stride=2),
            _ResBlock(256, 256, stride=1),  # stride=1: keeps spatial detail for small grids
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        flat_dim = 256 * 4 * 4   # 4096
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.GELU(),
            nn.Linear(512, n_states * n_freq),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        (B, n_states, n_freq)
        """
        B  = x.shape[0]
        h  = self.stem(x)
        h  = self.encoder(h)
        h  = self.pool(h)
        h  = h.reshape(B, -1)
        h  = self.head(h)
        return h.reshape(B, self.n_states, self.n_freq)


# ---------------------------------------------------------------------------
# Ensemble world model
# ---------------------------------------------------------------------------

class EnsembleWorldModel(nn.Module):
    """
    Ensemble of K independent WorldModelMember networks.

    Parameters
    ----------
    n_members   : ensemble size (default 5).
    img_h       : image height (rows).
    img_w_total : image width (cols + n_states).
    n_states    : number of filter states.
    n_freq      : number of frequency points.
    std_scale   : calibration parameter for confidence (dB).
    use_bagging : if True, each member trains on a random 80% subset in
                  train_step() to encourage diversity.
    """

    def __init__(
        self,
        n_members:   int   = 5,
        img_h:       int   = 25,
        img_w_total: int   = 29,
        n_states:    int   = 2,
        n_freq:      int   = 201,
        std_scale:   float = 3.0,
        use_bagging: bool  = True,
    ) -> None:
        super().__init__()
        self.n_members   = n_members
        self.n_states    = n_states
        self.n_freq      = n_freq
        self.std_scale   = std_scale
        self.use_bagging = use_bagging

        self.members = nn.ModuleList([
            WorldModelMember(img_h, img_w_total, n_states, n_freq)
            for _ in range(n_members)
        ])

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Run all members and return mean + std across the ensemble.

        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        mean_pred : (B, n_states, n_freq)
        std_pred  : (B, n_states, n_freq)
        """
        preds = torch.stack([m(x) for m in self.members], dim=0)  # (K, B, S, F)
        mean  = preds.mean(dim=0)
        std   = preds.std(dim=0, unbiased=False)
        return mean, std

    @torch.no_grad()
    def confidence(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute mean prediction and a scalar confidence score per sample.

        conf_score = exp( -mean_over_SF(std_pred) / std_scale )

        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        mean_pred  : (B, n_states, n_freq)
        conf_score : (B,)  in [0, 1]
        """
        mean, std  = self.predict(x)
        mean_std   = std.mean(dim=(-1, -2))           # (B,)
        conf_score = torch.exp(-mean_std / self.std_scale)
        return mean, conf_score

    def train_step(
        self,
        x:             Tensor,
        s_params_true: Tensor,
    ) -> Tensor:
        """
        Compute the training loss for all ensemble members.

        Each member is trained independently.  With use_bagging=True, each
        member sees a random 80% of the batch to encourage diversity.

        Parameters
        ----------
        x             : (B, 1, H, W+S)
        s_params_true : (B, n_states, n_freq)

        Returns
        -------
        Scalar mean MSE loss across members.
        """
        B    = x.shape[0]
        losses: List[Tensor] = []

        for member in self.members:
            if self.use_bagging and B > 1:
                n_sub = max(1, int(0.8 * B))
                idx   = torch.randperm(B, device=x.device)[:n_sub]
                x_sub = x[idx]
                y_sub = s_params_true[idx]
            else:
                x_sub = x
                y_sub = s_params_true

            pred = member(x_sub)                    # (n_sub, n_states, n_freq)
            loss = F.mse_loss(pred, y_sub)
            losses.append(loss)

        return torch.stack(losses).mean()
