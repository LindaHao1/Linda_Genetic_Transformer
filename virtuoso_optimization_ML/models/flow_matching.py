"""
flow_matching.py — Conditional flow matching generative model.

Formulation (linear / OT-CFM)
------------------------------
Given a data point x_1 (encoded design image) and noise x_0 ~ N(0, I):

  Interpolation:  x_t = (1 - t) * x_0  +  t * x_1
  Target field:   u_t = x_1 - x_0          (constant along the path)
  Loss:           MSE( v_θ(t, x_t, c), u_t )

where c = SParamEncoder(target_s_params) is the conditioning vector.

Classifier-free guidance (CFG)
-------------------------------
During training, the conditioning vector is replaced by a learned null
embedding with probability p_uncond (default 0.1).  At inference:

  v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

This allows tuning conditioning strength at inference without retraining.

Network: U-Net with AdaGN conditioning
----------------------------------------
The vector field v_θ is a U-Net that injects both the time embedding and the
conditioning vector via Adaptive Group Normalization (AdaGN) at every
residual block.  AdaGN computes per-channel scale and shift from a linear
projection of the combined (time_emb || cond) vector, then applies them to
the output of a GroupNorm layer — identical to the approach used in ADM
(Dhariwal & Nichol, 2021).

Inference
---------
Simple Euler integration from x_0 ~ N(0, I) to x_1 over n_steps.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .s_param_encoder import SParamEncoder


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for scalar time t ∈ [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """t : (B,)  →  (B, dim)"""
        device  = t.device
        half    = self.dim // 2
        freqs   = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        args    = t[:, None] * freqs[None, :]   # (B, half)
        emb     = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return emb


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding followed by a 2-layer MLP."""

    def __init__(self, t_dim: int) -> None:
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(t_dim)
        self.mlp     = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2),
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sin_emb(t))


# ---------------------------------------------------------------------------
# AdaGN residual block
# ---------------------------------------------------------------------------

class AdaGNResBlock(nn.Module):
    """
    2-D residual block where GroupNorm scale + shift are predicted from an
    external conditioning vector (time embedding + S-param conditioning).

    Replaces the standard affine parameters of GroupNorm with
      (scale, shift) = Linear(cond_total_dim, 2 * out_ch)
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(groups, out_ch, affine=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(groups, out_ch, affine=False)

        self.adagn = nn.Linear(cond_dim, 4 * out_ch)   # scale+shift for gn1 and gn2

        self.skip: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        x    : (B, C_in, H, W)
        cond : (B, cond_dim)
        """
        ada = self.adagn(cond)                    # (B, 4*out_ch)
        s1, b1, s2, b2 = ada.chunk(4, dim=-1)    # each (B, out_ch)

        h = self.conv1(x)
        h = self.gn1(h)
        h = h * (1 + s1[:, :, None, None]) + b1[:, :, None, None]
        h = F.silu(h)

        h = self.conv2(h)
        h = self.gn2(h)
        h = h * (1 + s2[:, :, None, None]) + b2[:, :, None, None]

        return F.silu(h + self.skip(x))


# ---------------------------------------------------------------------------
# Self-attention block (bottleneck)
# ---------------------------------------------------------------------------

class SelfAttention2D(nn.Module):
    """Lightweight multi-head self-attention over spatial positions."""

    def __init__(self, ch: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm    = nn.GroupNorm(min(8, ch), ch)
        self.qkv     = nn.Conv2d(ch, ch * 3, 1)
        self.proj    = nn.Conv2d(ch, ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h    = self.norm(x)
        qkv  = self.qkv(h).reshape(B, 3, self.n_heads, C // self.n_heads, H * W)
        q, k, v = qkv.unbind(dim=1)   # each (B, heads, head_dim, HW)
        q = q.permute(0, 1, 3, 2)     # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        scale = (C // self.n_heads) ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale
        attn  = attn.softmax(dim=-1)
        out   = (attn @ v).permute(0, 1, 3, 2)  # (B, heads, head_dim, HW)
        out   = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# U-Net vector field
# ---------------------------------------------------------------------------

class ConditionalVectorField(nn.Module):
    """
    U-Net that predicts the conditional flow matching vector field
    v_θ(t, x_t, c).

    Parameters
    ----------
    img_h         : image height.
    img_w         : image width (cols + n_states).
    cond_dim      : conditioning vector dimension (from SParamEncoder).
    t_dim         : time embedding dimension.
    base_channels : channel count for first encoder stage.
    """

    def __init__(
        self,
        img_h:         int,
        img_w:         int,
        cond_dim:      int = 256,
        t_dim:         int = 128,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        C  = base_channels
        tc = t_dim + cond_dim    # total conditioning dimension

        self.time_emb = TimeEmbedding(t_dim)

        # --- Encoder ---
        self.enc_in  = nn.Conv2d(1, C, 3, padding=1)

        self.enc1 = AdaGNResBlock(C,     C,     tc)
        self.enc2 = AdaGNResBlock(C,     C * 2, tc, stride=2)
        self.enc3 = AdaGNResBlock(C * 2, C * 4, tc, stride=2)

        # --- Bottleneck ---
        self.bot1 = AdaGNResBlock(C * 4, C * 4, tc)
        self.attn = SelfAttention2D(C * 4)
        self.bot2 = AdaGNResBlock(C * 4, C * 4, tc)

        # --- Decoder ---
        self.dec3 = AdaGNResBlock(C * 4 + C * 4, C * 2, tc)
        self.dec2 = AdaGNResBlock(C * 2 + C * 2, C,     tc)
        self.dec1 = AdaGNResBlock(C + C,          C,     tc)

        self.out_conv = nn.Conv2d(C, 1, kernel_size=1)

        # Upsample helpers
        self._up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x_t: Tensor, t: Tensor, c: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x_t : (B, 1, H, W)
        t   : (B,)            scalar time in [0, 1]
        c   : (B, cond_dim)   conditioning vector

        Returns
        -------
        (B, 1, H, W)  — predicted vector field
        """
        t_emb = self.time_emb(t)              # (B, t_dim)
        cond  = torch.cat([t_emb, c], dim=-1) # (B, t_dim + cond_dim)

        # Encoder
        h0 = self.enc_in(x_t)                # (B, C, H, W)
        h1 = self.enc1(h0, cond)             # (B, C, H, W)
        h2 = self.enc2(h1, cond)             # (B, 2C, H/2, W/2)
        h3 = self.enc3(h2, cond)             # (B, 4C, H/4, W/4)

        # Bottleneck
        h  = self.bot1(h3, cond)
        h  = self.attn(h)
        h  = self.bot2(h, cond)

        # Decoder with skip connections
        h = self._up(h)
        h = self.dec3(torch.cat([h, h3], dim=1), cond)  # (B, 2C, H/4, W/4)

        h = self._up(h)
        h = self.dec2(torch.cat([h, h2], dim=1), cond)  # (B, C, H/2, W/2)

        h = self._up(h)
        h = self.dec1(torch.cat([h, h1], dim=1), cond)  # (B, C, H, W)

        return self.out_conv(h)                          # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Full flow matching model
# ---------------------------------------------------------------------------

class FlowMatchingModel(nn.Module):
    """
    Wraps SParamEncoder + ConditionalVectorField.

    Provides train_step() for supervised CFM training and sample() for
    design generation via Euler integration.

    Parameters
    ----------
    n_states      : number of filter states.
    n_freq        : number of canonical frequency points.
    img_h         : image height.
    img_w         : image width (cols + n_states).
    cond_dim      : S-param conditioning dimension.
    t_dim         : time embedding dimension.
    base_channels : U-Net base channel count.
    p_uncond      : CFG null-conditioning drop probability during training.
    """

    def __init__(
        self,
        n_states:      int   = 2,
        n_freq:        int   = 201,
        img_h:         int   = 25,
        img_w:         int   = 29,
        cond_dim:      int   = 256,
        t_dim:         int   = 128,
        base_channels: int   = 64,
        p_uncond:      float = 0.1,
    ) -> None:
        super().__init__()
        self.p_uncond = p_uncond
        self.img_h    = img_h
        self.img_w    = img_w

        self.s_encoder  = SParamEncoder(n_states, n_freq, cond_dim)
        self.vector_field = ConditionalVectorField(
            img_h, img_w, cond_dim, t_dim, base_channels
        )

        # Learnable null embedding for CFG
        self.null_emb = nn.Parameter(torch.zeros(cond_dim))
        nn.init.normal_(self.null_emb, std=0.02)

    def train_step(
        self,
        x_1:      Tensor,    # (B, 1, H, W)  real encoded design image
        target_s: Tensor,    # (B, n_states, n_freq)  target S-params (conditioning)
    ) -> Tensor:
        """
        One flow matching training step.

        1. Encode target S-params → conditioning vector c.
        2. With probability p_uncond, replace c with the null embedding (CFG).
        3. Sample x_0 ~ N(0, I), t ~ Uniform(0, 1).
        4. Compute x_t = (1 - t) * x_0 + t * x_1.
        5. Predict vector field; compute MSE against target u_t = x_1 - x_0.

        Returns scalar loss.
        """
        B      = x_1.shape[0]
        device = x_1.device

        # Conditioning
        c = self.s_encoder(target_s)           # (B, cond_dim)

        # CFG dropout: replace conditioning with null embedding
        if self.training and self.p_uncond > 0:
            mask = torch.rand(B, device=device) < self.p_uncond   # (B,)
            null = self.null_emb[None].expand(B, -1)              # (B, cond_dim)
            c    = torch.where(mask[:, None], null, c)

        # Sample t and x_0
        t   = torch.rand(B, device=device)                         # (B,)
        x_0 = torch.randn_like(x_1)                               # (B, 1, H, W)

        # Interpolate
        t4    = t[:, None, None, None]
        x_t   = (1.0 - t4) * x_0 + t4 * x_1

        # Target
        u_t = x_1 - x_0

        # Predict
        v_pred = self.vector_field(x_t, t, c)

        return F.mse_loss(v_pred, u_t)

    @torch.no_grad()
    def sample(
        self,
        target_s:       Tensor,       # (B, n_states, n_freq)
        n_steps:        int   = 50,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """
        Generate design images by Euler integration from N(0, I) to x_1.

        Classifier-free guidance is applied when guidance_scale != 1:
          v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

        Parameters
        ----------
        target_s        : (B, n_states, n_freq)
        n_steps         : Euler integration steps.
        guidance_scale  : CFG scale.  1.0 = no guidance.

        Returns
        -------
        (B, 1, H, W)  generated images in [0, 1] (approximately).
        """
        B      = target_s.shape[0]
        device = target_s.device

        c_cond  = self.s_encoder(target_s)                          # (B, cond_dim)
        c_uncond = self.null_emb[None].expand(B, -1)               # (B, cond_dim)

        x = torch.randn(B, 1, self.img_h, self.img_w, device=device)

        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = step / n_steps
            t_vec = torch.full((B,), t_val, device=device)

            v_cond   = self.vector_field(x, t_vec, c_cond)

            if abs(guidance_scale - 1.0) > 1e-6:
                v_uncond = self.vector_field(x, t_vec, c_uncond)
                v        = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            x = x + v * dt

        return x
