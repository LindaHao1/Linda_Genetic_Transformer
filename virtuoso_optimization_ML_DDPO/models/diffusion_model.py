"""
diffusion_model.py — Conditional DDPM diffusion model for design generation.

Replaces the flow-matching model.  Same U-Net backbone (AdaGN conditioning),
different generative formulation.

Forward process (fixed):
  q(x_t | x_0) = N(sqrt(ā_t)·x_0, (1-ā_t)·I)
  x_t = sqrt(ā_t)·x_0 + sqrt(1-ā_t)·ε,   ε ~ N(0,I)

Reverse process (learned):
  p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t,c), σ_t²·I)
  μ_θ = 1/√α_t · (x_t - β_t/√(1-ā_t) · ε_θ(x_t,t,c))
  σ_t² = (1-ā_{t-1})/(1-ā_t) · β_t     (posterior variance)

Training:   E[ ||ε_θ(x_t, t, c) - ε||² ]    (epsilon prediction)

Sampling:
  sample()       — DDIM (deterministic, fast, n_steps << T)
  ddpo_sample()  — strided stochastic DDPM; stores trajectories for DDPO

DDPO fine-tuning:
  ddpo_loss() re-evaluates ε_θ on stored (x_t, t) pairs with gradients,
  then computes REINFORCE-style policy gradient:
    L = −E_trajectory[ R · Σ_t log p_θ(x_{t-1}|x_t) ]
  where R is the reward (world-model cost or infeasibility penalty).
  Designs with uncertain world-model predictions are masked out (R=0 → no gradient).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .s_param_encoder import SParamEncoder
from .score_encoder import ScoreEncoder


# ---------------------------------------------------------------------------
# Time embedding (identical to flow_matching.py)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(-math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
        args   = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, t_dim: int) -> None:
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(t_dim)
        self.mlp = nn.Sequential(nn.Linear(t_dim, t_dim * 2), nn.SiLU(), nn.Linear(t_dim * 2, t_dim))

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sin_emb(t))


# ---------------------------------------------------------------------------
# AdaGN residual block (identical to flow_matching.py)
# ---------------------------------------------------------------------------

class AdaGNResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(groups, out_ch, affine=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(groups, out_ch, affine=False)
        self.adagn = nn.Linear(cond_dim, 4 * out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False))
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        ada = self.adagn(cond)
        s1, b1, s2, b2 = ada.chunk(4, dim=-1)
        h = F.silu(self.gn1(self.conv1(x)) * (1 + s1[:, :, None, None]) + b1[:, :, None, None])
        h = self.gn2(self.conv2(h)) * (1 + s2[:, :, None, None]) + b2[:, :, None, None]
        return F.silu(h + self.skip(x))


# ---------------------------------------------------------------------------
# Self-attention block (identical to flow_matching.py)
# ---------------------------------------------------------------------------

class SelfAttention2D(nn.Module):
    def __init__(self, ch: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.n_heads, C // self.n_heads, H * W)
        q, k, v = qkv.unbind(dim=1)
        q = q.permute(0, 1, 3, 2); k = k.permute(0, 1, 3, 2); v = v.permute(0, 1, 3, 2)
        scale = (C // self.n_heads) ** -0.5
        attn  = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
        out   = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# U-Net noise predictor  (same architecture as ConditionalVectorField,
# but predicts ε instead of a velocity field)
# ---------------------------------------------------------------------------

class ConditionalEpsNet(nn.Module):
    """
    U-Net that predicts the noise ε_θ(x_t, t, c) at each diffusion timestep.
    Architecture is identical to ConditionalVectorField in flow_matching.py.
    """

    def __init__(self, img_h: int, img_w: int, cond_dim: int = 256,
                 t_dim: int = 128, base_channels: int = 64) -> None:
        super().__init__()
        C  = base_channels
        tc = t_dim + cond_dim

        self.time_emb = TimeEmbedding(t_dim)
        self.enc_in   = nn.Conv2d(1, C, 3, padding=1)
        self.enc1     = AdaGNResBlock(C,     C,     tc, stride=2)
        self.enc2     = AdaGNResBlock(C,     C * 2, tc, stride=2)
        self.enc3     = AdaGNResBlock(C * 2, C * 4, tc, stride=2)
        self.bot1     = AdaGNResBlock(C * 4, C * 4, tc)
        self.attn     = SelfAttention2D(C * 4)
        self.bot2     = AdaGNResBlock(C * 4, C * 4, tc)
        self.dec3     = AdaGNResBlock(C * 4 + C * 2, C * 2, tc)
        self.dec2     = AdaGNResBlock(C * 2 + C,     C,     tc)
        self.dec1     = AdaGNResBlock(C + C,         C,     tc)
        self.out_conv = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x_t: Tensor, t: Tensor, c: Tensor) -> Tensor:
        B, _, H, W = x_t.shape
        t_emb = self.time_emb(t)
        cond  = torch.cat([t_emb, c], dim=-1)

        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        x = F.pad(x_t, (0, pad_w, 0, pad_h))

        h0 = self.enc_in(x)
        h1 = self.enc1(h0, cond)
        h2 = self.enc2(h1, cond)
        h3 = self.enc3(h2, cond)
        h  = self.bot1(h3, cond)
        h  = self.attn(h)
        h  = self.bot2(h, cond)

        h = F.interpolate(h,  size=h2.shape[2:], mode="bilinear", align_corners=False)
        h = self.dec3(torch.cat([h, h2], dim=1), cond)
        h = F.interpolate(h,  size=h1.shape[2:], mode="bilinear", align_corners=False)
        h = self.dec2(torch.cat([h, h1], dim=1), cond)
        h = F.interpolate(h,  size=h0.shape[2:], mode="bilinear", align_corners=False)
        h = self.dec1(torch.cat([h, h0], dim=1), cond)

        out = self.out_conv(h)
        return out[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Full diffusion model
# ---------------------------------------------------------------------------

class DiffusionModel(nn.Module):
    """
    Conditional DDPM wrapping SParamEncoder + ConditionalEpsNet.

    Parameters
    ----------
    n_states      : number of filter states.
    n_freq        : canonical frequency-grid size.
    img_h, img_w  : design image spatial dimensions.
    cond_dim      : S-param conditioning dimension.
    t_dim         : time-embedding dimension.
    base_channels : U-Net base channel count.
    T             : number of diffusion timesteps.
    beta_start    : initial noise-schedule value.
    beta_end      : final  noise-schedule value.
    p_uncond      : classifier-free guidance null-conditioning probability.
    """

    def __init__(
        self,
        n_states:          int   = 2,
        n_freq:            int   = 201,
        img_h:             int   = 25,
        img_w:             int   = 29,
        cond_dim:          int   = 256,
        t_dim:             int   = 128,
        base_channels:     int   = 64,
        T:                 int   = 1000,
        beta_start:        float = 1e-4,
        beta_end:          float = 0.02,
        p_uncond:          float = 0.1,
        guidance_mode:     str   = "s_param",   # "s_param" or "score_based"
        n_score_components: int  = 4,
    ) -> None:
        super().__init__()
        self.T            = T
        self.img_h        = img_h
        self.img_w        = img_w
        self.p_uncond     = p_uncond
        self.guidance_mode = guidance_mode

        # ----- Noise schedule buffers -----
        betas              = torch.linspace(beta_start, beta_end, T)         # (T,)
        alphas             = 1.0 - betas                                       # (T,)
        alphas_cumprod     = torch.cumprod(alphas, dim=0)                      # ā_t  (T,)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)   # ā_{t-1} (T,)
        sqrt_abar          = alphas_cumprod.sqrt()
        sqrt_one_minus_abar = (1.0 - alphas_cumprod).sqrt()
        # Posterior variance: σ_t² = β_tilde = (1-ā_{t-1})/(1-ā_t) * β_t
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas",                betas)
        self.register_buffer("alphas",               alphas)
        self.register_buffer("alphas_cumprod",       alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",  alphas_cumprod_prev)
        self.register_buffer("sqrt_abar",            sqrt_abar)
        self.register_buffer("sqrt_one_minus_abar",  sqrt_one_minus_abar)
        self.register_buffer("posterior_variance",   posterior_variance)

        # ----- Learnable modules -----
        self.s_encoder = SParamEncoder(n_states, n_freq, cond_dim)
        if guidance_mode == "score_based":
            self.score_encoder = ScoreEncoder(n_states, n_score_components, cond_dim)
        self.eps_net   = ConditionalEpsNet(img_h, img_w, cond_dim, t_dim, base_channels)
        self.null_emb  = nn.Parameter(torch.zeros(cond_dim))
        nn.init.normal_(self.null_emb, std=0.02)

    # ------------------------------------------------------------------
    # Conditioning encoder dispatch
    # ------------------------------------------------------------------

    def _encode_cond(self, cond_input: Tensor) -> Tensor:
        """
        Encode conditioning input into a (B, cond_dim) vector.

        In 's_param' mode  : cond_input is (B, n_states, n_freq)  → SParamEncoder
        In 'score_based' mode: cond_input is (B, n_states, 4)     → ScoreEncoder
        """
        if self.guidance_mode == "score_based":
            return self.score_encoder(cond_input)
        return self.s_encoder(cond_input)

    # ------------------------------------------------------------------
    # Forward (noise injection helper)
    # ------------------------------------------------------------------

    def _q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Add noise: x_t = sqrt(ā_t)*x_0 + sqrt(1-ā_t)*ε"""
        s  = self.sqrt_abar[t][:, None, None, None]
        s1 = self.sqrt_one_minus_abar[t][:, None, None, None]
        return s * x_0 + s1 * noise

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        x_0:         Tensor,                   # (B, 1, H, W)  encoded design image
        cond_input:  Tensor,                   # (B, n_states, n_freq) or (B, n_states, 4)
        weights:     Optional[Tensor] = None,  # (B,) per-sample weights
        force_uncond: bool = False,            # if True, bypass encoder and use null_emb
    ) -> Tensor:
        """
        Standard DDPM training step: predict noise at random timestep.

        When ``force_uncond=True``, all samples in the batch are trained
        unconditionally (null embedding), regardless of ``cond_input``.
        Used for layout-only warmup pretraining where no S-params are available.

        Returns scalar MSE loss.
        """
        B      = x_0.shape[0]
        device = x_0.device

        # Conditioning with optional CFG dropout
        if force_uncond:
            c = self.null_emb[None].expand(B, -1)
        else:
            c = self._encode_cond(cond_input)
            if self.training and self.p_uncond > 0:
                mask = torch.rand(B, device=device) < self.p_uncond
                null = self.null_emb[None].expand(B, -1)
                c    = torch.where(mask[:, None], null, c)

        t     = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x_0)
        x_t   = self._q_sample(x_0, t, noise)

        t_norm   = t.float() / (self.T - 1)
        eps_pred = self.eps_net(x_t, t_norm, c)

        per_sample = F.mse_loss(eps_pred, noise, reduction="none").mean(dim=[1, 2, 3])
        if weights is not None:
            return (per_sample * weights.to(device)).mean()
        return per_sample.mean()

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _guided_eps(self, x: Tensor, t_norm: Tensor, c_cond: Tensor,
                    c_uncond: Tensor, guidance_scale: float) -> Tensor:
        """Compute classifier-free guided noise prediction."""
        eps_cond = self.eps_net(x, t_norm, c_cond)
        if abs(guidance_scale - 1.0) < 1e-6:
            return eps_cond
        eps_uncond = self.eps_net(x, t_norm, c_uncond)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    # ------------------------------------------------------------------
    # DDIM sampling  (fast, deterministic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        cond_input:     Tensor,       # (B, n_states, n_freq) or (B, n_states, 4)
        n_steps:        int   = 50,
        guidance_scale: float = 2.0,
    ) -> Tensor:
        """
        Generate design images via DDIM (deterministic, fast).

        Returns (B, 1, H, W).
        """
        B      = cond_input.shape[0]
        device = cond_input.device

        c_cond   = self._encode_cond(cond_input)
        c_uncond = self.null_emb[None].expand(B, -1)

        # Strided timesteps: T-1 down to 0 in n_steps steps
        ts = torch.linspace(self.T - 1, 0, n_steps + 1).long().to(device)

        x = torch.randn(B, 1, self.img_h, self.img_w, device=device)

        for i in range(n_steps):
            t_cur  = ts[i].item()
            t_prev = ts[i + 1].item()

            t_norm  = torch.full((B,), t_cur / (self.T - 1), device=device)
            eps_pred = self._guided_eps(x, t_norm, c_cond, c_uncond, guidance_scale)

            abar_t  = self.alphas_cumprod[t_cur]
            abar_tp = self.alphas_cumprod[t_prev] if t_prev > 0 else x.new_tensor(1.0)

            # x_0 prediction
            x0_pred = (x - self.sqrt_one_minus_abar[t_cur] * eps_pred) / self.sqrt_abar[t_cur]
            x0_pred = x0_pred.clamp(-3.0, 3.0)

            # DDIM step (eta=0): x_{t'} = sqrt(ā_{t'})*x0_pred + sqrt(1-ā_{t'})*eps_pred
            x = abar_tp.sqrt() * x0_pred + (1.0 - abar_tp).sqrt() * eps_pred

        return x

    # ------------------------------------------------------------------
    # DDPO sampling  (stochastic strided DDPM, returns trajectories)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddpo_sample(
        self,
        cond_input:     Tensor,       # (B, n_states, n_freq) or (B, n_states, 4)
        n_steps:        int   = 20,
        guidance_scale: float = 2.0,
        eta:            float = 1.0,  # stochasticity: 0=DDIM (invalid for DDPO), 1=full DDPM
    ) -> Tuple[Tensor, List[List[Dict]]]:
        """
        Stochastic strided DDPM sampling for DDPO trajectory collection.

        Uses the generalised DDIM reverse formula (Song et al. 2020) with
        configurable stochasticity eta ∈ (0, 1]:

          x_0_pred = (x_t - sqrt(1-ā_t)·ε_pred) / sqrt(ā_t)
          β_eff    = 1 - ā_t / ā_{t_prev}
          σ_t²     = eta² · (1-ā_{t_prev})/(1-ā_t) · β_eff
          μ_t      = sqrt(ā_{t_prev})·x_0_pred + sqrt(1-ā_{t_prev}-σ_t²)·ε_pred
          x_{t_prev} = μ_t + σ_t · z,   z ~ N(0,I)

        eta=1 → full stochastic DDPM (maximum noise per step).
        eta<1 → reduced variance; samples closer to DDIM while log-probs
                remain well-defined (sigma > 0 required for DDPO validity).
        eta=0 → DDIM (deterministic, sigma=0 → log-prob degenerate → NOT valid for DDPO).

        Returns
        -------
        x0_samples   : (B, 1, H, W) — final generated images (detached, on device)
        trajectories : list of B trajectory lists; each is a list of n_steps dicts:
            {'x_t': tensor(1,1,H,W cpu), 't_idx': int, 't_prev': int,
             'x_prev': tensor(1,1,H,W cpu), 'sigma_sq': float}
        """
        B      = cond_input.shape[0]
        device = cond_input.device

        c_cond   = self._encode_cond(cond_input)
        c_uncond = self.null_emb[None].expand(B, -1)

        # Strided timesteps
        ts = torch.linspace(self.T - 1, 0, n_steps + 1).long().tolist()
        # ts[0]=T-1, ts[-1]=0; steps go from ts[i] to ts[i+1]

        x = torch.randn(B, 1, self.img_h, self.img_w, device=device)
        per_sample_steps: List[List[Dict]] = [[] for _ in range(B)]

        for i in range(n_steps):
            t_cur  = int(ts[i])
            t_prev = int(ts[i + 1])

            t_norm   = torch.full((B,), t_cur / (self.T - 1), device=device)
            eps_pred = self._guided_eps(x, t_norm, c_cond, c_uncond, guidance_scale)

            abar_t  = self.alphas_cumprod[t_cur]
            abar_tp = self.alphas_cumprod[t_prev] if t_prev > 0 else x.new_tensor(1.0)

            x0_pred = (x - self.sqrt_one_minus_abar[t_cur] * eps_pred) / abar_t.sqrt()
            x0_pred = x0_pred.clamp(-3.0, 3.0)

            # Strided posterior variance, scaled by eta²
            beta_eff = (1.0 - abar_t / abar_tp).clamp(min=0.0)
            sigma_sq = (eta ** 2 * (1.0 - abar_tp) / (1.0 - abar_t) * beta_eff).clamp(min=1e-8)
            sigma    = sigma_sq.sqrt()

            # Posterior mean direction toward eps
            mean_eps_coef = (1.0 - abar_tp - sigma_sq).clamp(min=0.0).sqrt()
            mu    = abar_tp.sqrt() * x0_pred + mean_eps_coef * eps_pred

            z      = torch.randn_like(x)
            x_prev = mu + sigma * z

            # Store step info for each sample (CPU to save GPU memory)
            sigma_sq_val = sigma_sq.item()
            for b in range(B):
                per_sample_steps[b].append({
                    "x_t":      x[b:b+1].cpu(),
                    "t_idx":    t_cur,
                    "t_prev":   t_prev,
                    "x_prev":   x_prev[b:b+1].cpu(),
                    "sigma_sq": sigma_sq_val,
                })

            x = x_prev

        return x.detach(), per_sample_steps

    # ------------------------------------------------------------------
    # DDPO policy gradient loss
    # ------------------------------------------------------------------

    def ddpo_loss(
        self,
        trajectories:   List[List[Dict]],  # B_active trajectory lists
        rewards:        Tensor,             # (B_active,) normalised rewards
        cond_input:     Tensor,             # (B_active, n_states, n_freq) or (B_active, n_states, 4)
        guidance_scale: float = 2.0,
    ) -> Tensor:
        """
        Re-evaluate ε_θ on stored (x_t, t) pairs WITH gradients and compute
        the REINFORCE-style policy gradient loss.

          L = −E_B[ R · Σ_{t} log p_θ(x_{t-1}|x_t) ]
            = −E_B[ R · Σ_{t} −||x_{t-1} − μ_θ(x_t,t)||² / (2·σ_t²) ]

        Parameters
        ----------
        trajectories  : output of ddpo_sample() filtered to active (masked) designs.
        rewards       : (B_active,) normalised scalar reward per design.
        target_s      : (B_active, n_states, n_freq) conditioning for active designs.
        guidance_scale: CFG scale (should match the scale used during ddpo_sample).

        Returns
        -------
        Scalar loss tensor (backprop-ready).
        """
        B_active = len(trajectories)
        device   = rewards.device
        n_steps  = len(trajectories[0])

        # Conditioning — with gradients so they flow into the update
        c_cond   = self._encode_cond(cond_input)                    # (B_active, cond_dim)
        c_uncond = self.null_emb[None].expand(B_active, -1)         # (B_active, cond_dim)

        total_log_p = torch.zeros(B_active, device=device)

        for step_i in range(n_steps):
            # Gather stored tensors for this step across all active samples
            x_t_list    = [trajectories[b][step_i]["x_t"].to(device)   for b in range(B_active)]
            x_prev_list = [trajectories[b][step_i]["x_prev"].to(device) for b in range(B_active)]
            x_t    = torch.cat(x_t_list,    dim=0)   # (B_active, 1, H, W) — detached
            x_prev = torch.cat(x_prev_list, dim=0)   # (B_active, 1, H, W) — detached

            t_cur  = trajectories[0][step_i]["t_idx"]
            t_prev = trajectories[0][step_i]["t_prev"]
            sigma_sq = trajectories[0][step_i]["sigma_sq"]

            t_norm = torch.full((B_active,), t_cur / (self.T - 1), device=device)

            # Re-evaluate ε_θ WITH GRADIENTS
            eps_cond   = self.eps_net(x_t, t_norm, c_cond)
            if abs(guidance_scale - 1.0) > 1e-6:
                eps_uncond = self.eps_net(x_t, t_norm, c_uncond)
                eps_pred   = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = eps_cond

            abar_t  = self.alphas_cumprod[t_cur]
            abar_tp = self.alphas_cumprod[t_prev] if t_prev > 0 else x_t.new_tensor(1.0)
            sigma_sq_t = x_t.new_tensor(sigma_sq)

            # Re-derive μ_θ from ε_pred (same formula as ddpo_sample)
            x0_pred = (x_t - self.sqrt_one_minus_abar[t_cur] * eps_pred) / abar_t.sqrt()
            x0_pred = x0_pred.clamp(-3.0, 3.0)
            mean_eps_coef = (1.0 - abar_tp - sigma_sq_t).clamp(min=0.0).sqrt()
            mu_theta = abar_tp.sqrt() * x0_pred + mean_eps_coef * eps_pred

            # log p_θ(x_{t-1}|x_t) ∝ −||x_{t-1} − μ_θ||² / (2σ²)
            diff_sq   = ((x_prev - mu_theta) ** 2).sum(dim=[1, 2, 3])   # (B_active,)
            log_p_step = -diff_sq / (2.0 * sigma_sq_t.clamp(min=1e-8))
            total_log_p = total_log_p + log_p_step

        # Normalise by n_steps for gradient-magnitude stability across different schedules
        total_log_p = total_log_p / n_steps

        # DDPO loss: maximise E[R · log_p]  ≡  minimise −E[R · log_p]
        loss = -(rewards * total_log_p).mean()
        return loss
