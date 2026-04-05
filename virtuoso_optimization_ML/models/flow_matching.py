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
from .score_encoder import ScoreEncoder
from .feasibility_classifier import FeasibilityClassifier


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
# Cross-attention block (spec-to-layout conditioning)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention from 2-D spatial features (queries) to a spec token sequence
    (keys/values).

    Spatial feature map positions attend to the spec token sequence, allowing
    each layout region to selectively pull information from the parts of the
    frequency response (or per-state scores) that are most relevant to it.

    Parameters
    ----------
    spatial_dim : channel count C of the spatial feature map (B, C, H, W).
    ctx_dim     : embedding dimension of the spec token sequence (B, L, ctx_dim).
    n_heads     : number of attention heads (spatial_dim must be divisible by n_heads).
    """

    def __init__(self, spatial_dim: int, ctx_dim: int, n_heads: int = 4) -> None:
        assert spatial_dim % n_heads == 0, (
            f"spatial_dim ({spatial_dim}) must be divisible by n_heads ({n_heads})"
        )
        super().__init__()
        self.n_heads  = n_heads
        self.norm_x   = nn.LayerNorm(spatial_dim)
        self.norm_ctx = nn.LayerNorm(ctx_dim)
        self.ctx_proj = (
            nn.Linear(ctx_dim, spatial_dim)
            if ctx_dim != spatial_dim else nn.Identity()
        )
        self.q_proj   = nn.Linear(spatial_dim, spatial_dim)
        self.k_proj   = nn.Linear(spatial_dim, spatial_dim)
        self.v_proj   = nn.Linear(spatial_dim, spatial_dim)
        self.out_proj = nn.Linear(spatial_dim, spatial_dim)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x   : (B, C, H, W)  spatial feature map (queries)
        ctx : (B, L, ctx_dim)  spec token sequence (keys and values)

        Returns
        -------
        (B, C, H, W)  with residual connection applied.
        """
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)         # (B, HW, C)

        # Pre-norm on both sides
        q  = self.q_proj(self.norm_x(x_flat))                     # (B, HW, C)
        kv = self.ctx_proj(self.norm_ctx(ctx))                    # (B, L, C)
        k  = self.k_proj(kv)                                      # (B, L, C)
        v  = self.v_proj(kv)                                      # (B, L, C)

        # Split into heads
        hd = C // self.n_heads
        L  = ctx.shape[1]

        def split_heads(t: Tensor, seq: int) -> Tensor:
            return t.reshape(B, seq, self.n_heads, hd).permute(0, 2, 1, 3)

        q = split_heads(q, H * W)    # (B, heads, HW, hd)
        k = split_heads(k, L)        # (B, heads, L,  hd)
        v = split_heads(v, L)        # (B, heads, L,  hd)

        scale = hd ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale    # (B, heads, HW, L)
        attn  = attn.softmax(dim=-1)
        out   = (attn @ v)                            # (B, heads, HW, hd)
        out   = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out   = self.out_proj(out)

        # Residual over the original (un-normed) spatial features
        out = (x_flat + out).permute(0, 2, 1).reshape(B, C, H, W)
        return out


# ---------------------------------------------------------------------------
# U-Net vector field
# ---------------------------------------------------------------------------

class ConditionalVectorField(nn.Module):
    """
    U-Net that predicts the conditional flow matching vector field
    v_θ(t, x_t, c) or v_θ(t, x_t, ctx) depending on conditioning mode.

    Parameters
    ----------
    img_h          : image height.
    img_w          : image width (cols + n_states).
    cond_dim       : conditioning vector dimension (legacy AdaGN mode).
    t_dim          : time embedding dimension.
    base_channels  : channel count for first encoder stage.
    use_cross_attn : if True, spec conditioning is done via cross-attention
                     instead of AdaGN.  AdaGN receives only the time embedding.
    ctx_dim        : spec token embedding dimension (used when use_cross_attn=True).
    n_heads        : number of attention heads in cross-attention blocks.
    """

    def __init__(
        self,
        img_h:          int,
        img_w:          int,
        cond_dim:       int  = 256,
        t_dim:          int  = 128,
        base_channels:  int  = 64,
        use_cross_attn: bool = False,
        ctx_dim:        int  = 128,
        n_heads:        int  = 4,
    ) -> None:
        super().__init__()
        C  = base_channels
        self.use_cross_attn = use_cross_attn

        # In cross-attn mode AdaGN only carries the time signal;
        # in legacy mode it carries time + spec vector.
        tc = t_dim if use_cross_attn else t_dim + cond_dim

        self.time_emb = TimeEmbedding(t_dim)

        # --- Encoder: 3 stride-2 stages → input must be padded to multiple of 8 ---
        # enc_in: no stride, lifts to C channels
        # enc1/enc2/enc3: each halves spatial dims
        # Skip connections saved: h0 (C), h1 (C), h2 (2C)
        self.enc_in = nn.Conv2d(1, C, 3, padding=1)
        self.enc1   = AdaGNResBlock(C,     C,     tc, stride=2)
        self.enc2   = AdaGNResBlock(C,     C * 2, tc, stride=2)
        self.enc3   = AdaGNResBlock(C * 2, C * 4, tc, stride=2)

        # --- Bottleneck ---
        self.bot1 = AdaGNResBlock(C * 4, C * 4, tc)
        self.attn = SelfAttention2D(C * 4)
        self.bot2 = AdaGNResBlock(C * 4, C * 4, tc)

        # --- Decoder: 3 up stages, skips from enc2/enc1/enc_in outputs ---
        # dec3: up from H/8 → H/4, cat with h2 (2C) → in: 4C+2C=6C, out: 2C
        # dec2: up from H/4 → H/2, cat with h1 (C)  → in: 2C+C=3C,  out: C
        # dec1: up from H/2 → H,   cat with h0 (C)  → in: C+C=2C,   out: C
        self.dec3 = AdaGNResBlock(C * 4 + C * 2, C * 2, tc)
        self.dec2 = AdaGNResBlock(C * 2 + C,     C,     tc)
        self.dec1 = AdaGNResBlock(C + C,         C,     tc)

        self.out_conv = nn.Conv2d(C, 1, kernel_size=1)

        # --- Cross-attention blocks (only created when use_cross_attn=True) ---
        if use_cross_attn:
            self.bot_xattn  = CrossAttentionBlock(C * 4, ctx_dim, n_heads)
            self.dec3_xattn = CrossAttentionBlock(C * 2, ctx_dim, n_heads)
            self.dec2_xattn = CrossAttentionBlock(C,     ctx_dim, n_heads)
            self.dec1_xattn = CrossAttentionBlock(C,     ctx_dim, n_heads)

    def forward(
        self,
        x_t: Tensor,
        t:   Tensor,
        c:   Tensor,
        ctx: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x_t : (B, 1, H, W)          any H, W; padded internally to multiple of 8
        t   : (B,)                   scalar time in [0, 1]
        c   : (B, cond_dim)          spec conditioning vector (legacy mode only)
        ctx : (B, L, ctx_dim) | None spec token sequence (cross-attn mode only)

        Returns
        -------
        (B, 1, H, W)  predicted vector field, same spatial size as input
        """
        B, _, H, W = x_t.shape
        t_emb = self.time_emb(t)                          # (B, t_dim)

        if self.use_cross_attn:
            cond = t_emb                                  # (B, t_dim)
        else:
            cond = torch.cat([t_emb, c], dim=-1)          # (B, t_dim + cond_dim)

        # Pad to the nearest multiple of 8 (3 stride-2 stages → divisible by 8)
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        x = F.pad(x_t, (0, pad_w, 0, pad_h))  # pad right / bottom only

        # Encoder — save outputs as skip connections
        h0 = self.enc_in(x)        # (B, C,   H',   W')
        h1 = self.enc1(h0, cond)   # (B, C,   H'/2, W'/2)
        h2 = self.enc2(h1, cond)   # (B, 2C,  H'/4, W'/4)
        h3 = self.enc3(h2, cond)   # (B, 4C,  H'/8, W'/8)

        # Bottleneck
        h = self.bot1(h3, cond)
        h = self.attn(h)
        if ctx is not None:
            h = self.bot_xattn(h, ctx)
        h = self.bot2(h, cond)     # (B, 4C, H'/8, W'/8)

        # Decoder — upsample to exact skip size to handle any padding remainder
        h = F.interpolate(h, size=h2.shape[2:], mode="bilinear", align_corners=False)
        h = self.dec3(torch.cat([h, h2], dim=1), cond)   # (B, 2C, H'/4, W'/4)
        if ctx is not None:
            h = self.dec3_xattn(h, ctx)

        h = F.interpolate(h, size=h1.shape[2:], mode="bilinear", align_corners=False)
        h = self.dec2(torch.cat([h, h1], dim=1), cond)   # (B, C, H'/2, W'/2)
        if ctx is not None:
            h = self.dec2_xattn(h, ctx)

        h = F.interpolate(h, size=h0.shape[2:], mode="bilinear", align_corners=False)
        h = self.dec1(torch.cat([h, h0], dim=1), cond)   # (B, C, H', W')
        if ctx is not None:
            h = self.dec1_xattn(h, ctx)

        out = self.out_conv(h)     # (B, 1, H', W')
        return out[:, :, :H, :W]   # crop padding back to original size


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
    n_states           : number of filter states.
    n_freq             : number of canonical frequency points.
    img_h              : image height.
    img_w              : image width (cols + n_states).
    cond_dim           : conditioning vector dimension (legacy AdaGN mode).
    t_dim              : time embedding dimension.
    base_channels      : U-Net base channel count.
    p_uncond           : CFG null-conditioning drop probability during training.
    guidance_mode      : "s_param" (default) or "score_based".
    n_score_components : number of score components per state (score_based mode).
    use_cross_attn     : if True, spec conditioning uses cross-attention instead of AdaGN.
                         The encoder outputs a token sequence (B, L, seq_d_model) and
                         cross-attention blocks in the U-Net attend to it spatially.
                         Works for both "s_param" and "score_based" guidance modes:
                           s_param    → L = n_states * 32 (per-freq-bin tokens)
                           score_based → L = n_states     (per-state tokens)
    seq_d_model        : token embedding dimension when use_cross_attn=True.
    ca_n_heads         : attention heads in cross-attention blocks.
    """

    def __init__(
        self,
        n_states:           int   = 2,
        n_freq:             int   = 201,
        img_h:              int   = 25,
        img_w:              int   = 29,
        cond_dim:           int   = 256,
        t_dim:              int   = 128,
        base_channels:      int   = 64,
        p_uncond:           float = 0.1,
        guidance_mode:      str   = "s_param",
        n_score_components: int   = 4,
        use_cross_attn:     bool  = False,
        seq_d_model:        int   = 128,
        ca_n_heads:         int   = 4,
    ) -> None:
        super().__init__()
        self.p_uncond       = p_uncond
        self.img_h          = img_h
        self.img_w          = img_w
        self.guidance_mode  = guidance_mode
        self.use_cross_attn = use_cross_attn

        self.s_encoder = SParamEncoder(
            n_states, n_freq, cond_dim,
            use_cross_attn=use_cross_attn, seq_d_model=seq_d_model,
        )
        if guidance_mode == "score_based":
            self.score_encoder = ScoreEncoder(
                n_states, n_score_components, cond_dim,
                use_cross_attn=use_cross_attn, seq_d_model=seq_d_model,
            )
        self.vector_field = ConditionalVectorField(
            img_h, img_w, cond_dim, t_dim, base_channels,
            use_cross_attn=use_cross_attn, ctx_dim=seq_d_model, n_heads=ca_n_heads,
        )

        # Learnable null embedding for CFG.
        # Legacy mode : (cond_dim,)     — a single conditioning vector.
        # Cross-attn  : (1, seq_d_model) — a single null token, expanded to (B, L, D)
        #               at runtime so it works regardless of the actual sequence length L.
        if use_cross_attn:
            self.null_emb = nn.Parameter(torch.zeros(1, seq_d_model))
        else:
            self.null_emb = nn.Parameter(torch.zeros(cond_dim))
        nn.init.normal_(self.null_emb, std=0.02)

    def _encode_cond(self, cond_input: Tensor) -> Tensor:
        """
        Encode conditioning input.

        Legacy mode (use_cross_attn=False):
          's_param'    : (B, n_states, n_freq) → (B, cond_dim)
          'score_based': (B, n_states, 4)      → (B, cond_dim)

        Cross-attn mode (use_cross_attn=True):
          's_param'    : (B, n_states, n_freq) → (B, n_states*32, seq_d_model)
          'score_based': (B, n_states, 4)      → (B, n_states,    seq_d_model)
        """
        if self.guidance_mode == "score_based":
            return self.score_encoder(cond_input)
        return self.s_encoder(cond_input)

    def train_step(
        self,
        x_1:                    Tensor,                          # (B, 1, H, W)
        cond_input:             Tensor,                          # (B, n_states, n_freq) or (B, n_states, 4)
        weights:                Optional[Tensor] = None,         # (B,) per-sample loss weights
        classifier:             Optional[FeasibilityClassifier] = None,
        classifier_weight:      float = 0.0,
        force_uncond:           bool  = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        One flow matching training step.

        1. Encode conditioning input → conditioning vector c.
           When force_uncond=True all samples use the null embedding (layout-only warmup).
        2. With probability p_uncond, replace c with the null embedding (CFG).
        3. Sample x_0 ~ N(0, I), t ~ Uniform(0, 1).
        4. Compute x_t = (1 - t) * x_0 + t * x_1.
        5. Predict vector field; compute (weighted) MSE against u_t = x_1 - x_0.
        6. Optionally add classifier feasibility guidance loss on
           x1_pred = x_t + (1-t)*v_pred  (suppressed at t < 0.2).

        Parameters
        ----------
        cond_input         : (B, n_states, n_freq) in s_param mode;
                             (B, n_states, n_components) in score_based mode.
        weights            : optional (B,) per-sample loss weights (pre-normalised).
        classifier         : FeasibilityClassifier instance or None.
        classifier_weight  : scalar multiplier for the classifier guidance term.
        force_uncond       : if True, bypass encoder and use null_emb for all samples.

        Returns
        -------
        (total_loss, fm_loss, clf_loss) — all scalar Tensors.
        """
        B      = x_1.shape[0]
        device = x_1.device

        # Conditioning — cross-attn and legacy paths share the same flow below;
        # the difference is only in what gets passed to vector_field.
        if self.use_cross_attn:
            if force_uncond:
                # Single null token expanded to (B, 1, seq_d_model)
                ctx = self.null_emb.expand(B, -1).unsqueeze(1)    # (B, 1, D)
            else:
                ctx = self._encode_cond(cond_input)                # (B, L, D)

            # CFG dropout: replace each sample's token sequence with the null token
            if self.training and self.p_uncond > 0:
                mask  = torch.rand(B, device=device) < self.p_uncond  # (B,)
                L     = ctx.shape[1]
                null  = self.null_emb.expand(B, L, -1)                # (B, L, D)
                ctx   = torch.where(mask[:, None, None], null, ctx)

            c = x_1.new_zeros(B, 0)   # dummy — not used in cross-attn mode
        else:
            if force_uncond:
                c = self.null_emb[None].expand(B, -1)               # (B, cond_dim)
            else:
                c = self._encode_cond(cond_input)                    # (B, cond_dim)

            # CFG dropout
            if self.training and self.p_uncond > 0:
                mask = torch.rand(B, device=device) < self.p_uncond  # (B,)
                null = self.null_emb[None].expand(B, -1)             # (B, cond_dim)
                c    = torch.where(mask[:, None], null, c)

            ctx = None

        # Sample t and x_0
        t   = torch.rand(B, device=device)                         # (B,)
        x_0 = torch.randn_like(x_1)                               # (B, 1, H, W)

        # Interpolate
        t4  = t[:, None, None, None]
        x_t = (1.0 - t4) * x_0 + t4 * x_1

        # Target
        u_t = x_1 - x_0

        # Predict
        v_pred = self.vector_field(x_t, t, c, ctx=ctx)

        # Per-sample flow matching MSE, then optional weighting
        per_sample = F.mse_loss(v_pred, u_t, reduction="none").mean(dim=[1, 2, 3])  # (B,)
        if weights is not None:
            fm_loss = (per_sample * weights.to(device)).mean()
        else:
            fm_loss = per_sample.mean()

        # Classifier feasibility guidance on predicted endpoint x1_pred
        clf_loss = x_1.new_zeros(1).squeeze()
        if classifier is not None and classifier_weight > 0.0:
            x1_pred = x_t + (1.0 - t4) * v_pred                  # (B, 1, H, W)
            logits  = classifier._logits(x1_pred)                  # (B,)
            # Negative log-probability of feasibility: 0 at t≤0.2, linearly ramps to 1 at t=1.
            # Soft ramp avoids the gradient discontinuity of a hard threshold mask.
            raw_clf     = -F.logsigmoid(logits)                    # (B,) ≥ 0
            clf_t_weight = ((t - 0.2) / 0.8).clamp(0.0, 1.0)     # (B,)
            clf_loss = (raw_clf * clf_t_weight).mean()

        total_loss = fm_loss + classifier_weight * clf_loss
        return total_loss, fm_loss, clf_loss

    def sample(
        self,
        cond_input:                 Tensor,
        n_steps:                    int   = 50,
        guidance_scale:             float = 1.0,
        classifier:                 Optional[FeasibilityClassifier] = None,
        classifier_guidance_weight: float = 0.0,
    ) -> Tensor:
        """
        Generate design images by Euler integration from N(0, I) to x_1.

        Classifier-free guidance is applied when guidance_scale != 1:
          v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

        Classifier guidance (optional) adds ∂ log P(feasible|x) / ∂x at each step,
        steering the ODE trajectory toward feasible regions.

        Parameters
        ----------
        cond_input                 : (B, n_states, n_freq) or (B, n_states, n_components).
        n_steps                    : Euler integration steps.
        guidance_scale             : CFG scale.  1.0 = no CFG guidance.
        classifier                 : FeasibilityClassifier instance or None.
        classifier_guidance_weight : scale for classifier gradient guidance.

        Returns
        -------
        (B, 1, H, W)  generated images in [0, 1] (approximately).
        """
        B      = cond_input.shape[0]
        device = cond_input.device

        use_classifier = (
            classifier is not None
            and classifier_guidance_weight > 0.0
        )

        with torch.no_grad():
            if self.use_cross_attn:
                ctx_cond   = self._encode_cond(cond_input)           # (B, L, D)
                L          = ctx_cond.shape[1]
                ctx_uncond = self.null_emb.expand(B, L, -1)          # (B, L, D)
                c_dummy    = cond_input.new_zeros(B, 0)
            else:
                c_cond   = self._encode_cond(cond_input)             # (B, cond_dim)
                c_uncond = self.null_emb[None].expand(B, -1)         # (B, cond_dim)

        x  = torch.randn(B, 1, self.img_h, self.img_w, device=device)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = step / n_steps
            t_vec = torch.full((B,), t_val, device=device)

            with torch.no_grad():
                if self.use_cross_attn:
                    v_cond = self.vector_field(x, t_vec, c_dummy, ctx=ctx_cond)
                    if abs(guidance_scale - 1.0) > 1e-6:
                        v_uncond = self.vector_field(x, t_vec, c_dummy, ctx=ctx_uncond)
                        v        = v_uncond + guidance_scale * (v_cond - v_uncond)
                    else:
                        v = v_cond
                else:
                    v_cond = self.vector_field(x, t_vec, c_cond)
                    if abs(guidance_scale - 1.0) > 1e-6:
                        v_uncond = self.vector_field(x, t_vec, c_uncond)
                        v        = v_uncond + guidance_scale * (v_cond - v_uncond)
                    else:
                        v = v_cond

            x = x + v * dt

            # Classifier gradient guidance (requires grad, done outside no_grad)
            if use_classifier:
                grad = classifier.log_prob_gradient(x)
                x    = x + classifier_guidance_weight * grad

        return x.detach()
