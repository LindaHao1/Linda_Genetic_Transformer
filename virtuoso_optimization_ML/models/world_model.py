"""
world_model.py — Single heteroscedastic surrogate model + RND for OOD detection.

Architecture
------------
WorldModel
  Input : (B, 1, H, W+S)
  Encoder:
    Conv2d(1→64, 3×3) → GroupNorm → GELU
    ResBlock(64→128,  stride=2)
    ResBlock(128→256, stride=2)
    ResBlock(256→256, stride=1)
    AdaptiveAvgPool2d(4, 4) → flatten → Linear(4096, 512) → GELU  [latent]
  Two parallel prediction heads:
    mean_head   : Linear(512, n_states * n_freq)
    logvar_head : Linear(512, n_states * n_freq)  clamped to [-6, 6]
  Loss: Gaussian NLL = 0.5 * (logvar + (y - mean)^2 * exp(-logvar))

Confidence
----------
  variance   = exp(logvar)
  conf_score = exp(-mean(sqrt(variance)) / std_scale)

High conf_score → low predictive uncertainty → relabel without simulation.
Low  conf_score (or high RND score) → trigger real simulation.

RNDNetwork (Random Network Distillation)
----------------------------------------
Two copies of the CNN encoder body:
  target   : randomly initialised, frozen (never updated).
  predictor: trainable — learns to predict the target's features.

  rnd_score(x) = mean( (predictor(x) - target(x))^2 )  per sample

High rnd_score → input is OOD relative to the training distribution.
The predictor is trained by calling train_step() on each real-data batch.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared building blocks
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


def _build_cnn_encoder() -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Build the shared CNN encoder body used by WorldModel, RNDNetwork, and
    FeasibilityClassifier.

    Returns
    -------
    (stem, encoder, pool) — three sequential nn.Modules.

    After pool, call  h.reshape(B, -1)  to get a (B, 4096) flat latent vector.
    """
    stem = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, 64),
        nn.GELU(),
    )
    encoder = nn.Sequential(
        _ResBlock(64,  128, stride=2),
        _ResBlock(128, 256, stride=2),
        _ResBlock(256, 256, stride=1),
    )
    pool = nn.AdaptiveAvgPool2d((4, 4))
    return stem, encoder, pool


_FLAT_DIM   = 256 * 4 * 4   # 4096
_LATENT_DIM = 512


# ---------------------------------------------------------------------------
# WorldModel — single model, heteroscedastic (mean + variance) output
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """
    Single CNN surrogate model that predicts both the mean and variance of the
    S-parameter response given an encoded design image.

    Parameters
    ----------
    n_states    : number of filter states.
    n_freq      : number of frequency points in the canonical grid.
    std_scale   : calibration parameter for the confidence score (dB).
    """

    def __init__(
        self,
        n_states:  int   = 2,
        n_freq:    int   = 101,
        std_scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.n_states  = n_states
        self.n_freq    = n_freq
        self.std_scale = std_scale

        self.stem, self.encoder, self.pool = _build_cnn_encoder()

        self.latent_mlp = nn.Sequential(
            nn.Linear(_FLAT_DIM, _LATENT_DIM),
            nn.GELU(),
        )
        out_dim = n_states * n_freq
        self.mean_head   = nn.Linear(_LATENT_DIM, out_dim)
        self.logvar_head = nn.Linear(_LATENT_DIM, out_dim)

    def _encode(self, x: Tensor) -> Tensor:
        """Shared encoder pass.  Returns (B, _LATENT_DIM) latent vector."""
        B = x.shape[0]
        h = self.stem(x)
        h = self.encoder(h)
        h = self.pool(h)
        h = h.reshape(B, -1)
        return self.latent_mlp(h)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        mean    : (B, n_states, n_freq)
        logvar  : (B, n_states, n_freq)   clamped to [-6, 6]
        """
        latent = self._encode(x)
        mean   = self.mean_head(latent).reshape(-1, self.n_states, self.n_freq)
        logvar = self.logvar_head(latent).reshape(-1, self.n_states, self.n_freq)
        logvar = logvar.clamp(-6.0, 6.0)
        return mean, logvar

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Return mean prediction and predictive variance.

        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        mean     : (B, n_states, n_freq)
        variance : (B, n_states, n_freq)   = exp(logvar)
        """
        mean, logvar = self(x)
        return mean, torch.exp(logvar)

    @torch.no_grad()
    def confidence(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute mean prediction and a scalar confidence score per sample.

        conf_score = exp(-mean_over_SF(sqrt(variance)) / std_scale)

        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        mean_pred  : (B, n_states, n_freq)
        conf_score : (B,)  in [0, 1]
        """
        mean, variance = self.predict(x)
        mean_std       = variance.sqrt().mean(dim=(-1, -2))   # (B,)
        conf_score     = torch.exp(-mean_std / self.std_scale)
        return mean, conf_score

    def train_step(self, x: Tensor, s_params_true: Tensor) -> Tensor:
        """
        Gaussian NLL training loss.

        Loss = 0.5 * mean( logvar + (y - mean)^2 * exp(-logvar) )

        Parameters
        ----------
        x             : (B, 1, H, W+S)
        s_params_true : (B, n_states, n_freq)

        Returns
        -------
        Scalar NLL loss.
        """
        mean, logvar = self(x)
        nll = 0.5 * (logvar + (s_params_true - mean).pow(2) * torch.exp(-logvar))
        return nll.mean()


# ---------------------------------------------------------------------------
# RNDNetwork — Random Network Distillation for OOD detection
# ---------------------------------------------------------------------------

class RNDNetwork(nn.Module):
    """
    Random Network Distillation (Burda et al., 2018) for OOD detection.

    Two CNN encoders share the same architecture:
      target   : randomly initialised, permanently frozen.
      predictor: trained to predict the target's feature embedding.

    rnd_score(x) measures the prediction error per sample.  High error
    indicates the input x is OOD (far from the training distribution).

    Parameters
    ----------
    rnd_feature_dim : dimensionality of the RND feature embedding.
    """

    def __init__(self, rnd_feature_dim: int = 256) -> None:
        super().__init__()
        self.rnd_feature_dim = rnd_feature_dim

        # Target: frozen random encoder
        t_stem, t_enc, t_pool = _build_cnn_encoder()
        self.target_stem    = t_stem
        self.target_encoder = t_enc
        self.target_pool    = t_pool
        self.target_proj    = nn.Linear(_FLAT_DIM, rnd_feature_dim)

        # Predictor: trainable encoder
        p_stem, p_enc, p_pool = _build_cnn_encoder()
        self.predictor_stem    = p_stem
        self.predictor_encoder = p_enc
        self.predictor_pool    = p_pool
        self.predictor_proj    = nn.Sequential(
            nn.Linear(_FLAT_DIM, _LATENT_DIM),
            nn.GELU(),
            nn.Linear(_LATENT_DIM, rnd_feature_dim),
        )

        # Freeze target completely
        for p in self.target_stem.parameters():
            p.requires_grad_(False)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_pool.parameters():
            p.requires_grad_(False)
        for p in self.target_proj.parameters():
            p.requires_grad_(False)

    def _target_features(self, x: Tensor) -> Tensor:
        """(B, rnd_feature_dim) — no gradient, always frozen."""
        B = x.shape[0]
        h = self.target_stem(x)
        h = self.target_encoder(h)
        h = self.target_pool(h)
        h = h.reshape(B, -1)
        return self.target_proj(h)

    def _predictor_features(self, x: Tensor) -> Tensor:
        """(B, rnd_feature_dim) — trainable."""
        B = x.shape[0]
        h = self.predictor_stem(x)
        h = self.predictor_encoder(h)
        h = self.predictor_pool(h)
        h = h.reshape(B, -1)
        return self.predictor_proj(h)

    @torch.no_grad()
    def rnd_score(self, x: Tensor) -> Tensor:
        """
        Compute the per-sample OOD score (MSE between predictor and target).

        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        (B,) non-negative scalar OOD score per sample.
             Higher → more out-of-distribution.
        """
        target_feat    = self._target_features(x)
        predictor_feat = self._predictor_features(x)
        return F.mse_loss(predictor_feat, target_feat, reduction="none").mean(dim=-1)

    def train_step(self, x: Tensor) -> Tensor:
        """
        Update the predictor to better match the frozen target features.

        Parameters
        ----------
        x : (B, 1, H, W+S)

        Returns
        -------
        Scalar MSE loss (predictor only; target receives no gradient).
        """
        with torch.no_grad():
            target_feat = self._target_features(x)
        predictor_feat = self._predictor_features(x)
        return F.mse_loss(predictor_feat, target_feat)
