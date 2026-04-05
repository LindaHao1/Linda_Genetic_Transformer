"""
s_param_encoder.py — 1-D CNN encoder for S-parameter conditioning.

Encodes a target S-parameter curve (S21 in dB for all states) into a
fixed-length conditioning vector used by the flow matching generative model.

Architecture
------------
Input : (B, n_states, n_freq)
Steps :
  1. Reshape to (B * n_states, 1, n_freq)
  2. Three Conv1d layers: 1→32→64→128, kernel sizes 7/5/3, padding maintains length
  3. AdaptiveAvgPool1d(32) → (B * n_states, 128, 32)
  4. Flatten → (B * n_states, 128 * 32)
  5. Reshape to (B, n_states * 128 * 32)
  6. MLP: Linear → LayerNorm → GELU → Linear → (B, cond_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SParamEncoder(nn.Module):
    """
    Encodes target S-parameter curves into a conditioning vector or token sequence.

    Parameters
    ----------
    n_states       : number of filter states.
    n_freq         : number of frequency points in the canonical grid.
    cond_dim       : output conditioning vector dimension (legacy mode, ignored when use_cross_attn=True).
    use_cross_attn : if True, output a token sequence (B, n_states*32, seq_d_model) instead of
                     a single vector (B, cond_dim).  Each of the 32 frequency-pool bins per state
                     becomes an independent token, preserving the spectral structure.
    seq_d_model    : token embedding dimension when use_cross_attn=True.
    """

    def __init__(
        self,
        n_states:       int,
        n_freq:         int,
        cond_dim:       int  = 256,
        use_cross_attn: bool = False,
        seq_d_model:    int  = 128,
    ) -> None:
        super().__init__()
        self.n_states       = n_states
        self.n_freq         = n_freq
        self.cond_dim       = cond_dim
        self.use_cross_attn = use_cross_attn

        # 1-D CNN over the frequency axis (shared by both modes)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64,  kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(32)   # → (*, 128, 32)

        if use_cross_attn:
            # Project each 128-dim frequency-bin feature to seq_d_model
            self.proj = nn.Linear(128, seq_d_model)
        else:
            flat_dim = n_states * 128 * 32
            self.mlp = nn.Sequential(
                nn.Linear(flat_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, cond_dim),
            )

    def forward(self, s_params: Tensor) -> Tensor:
        """
        Parameters
        ----------
        s_params : (B, n_states, n_freq)

        Returns
        -------
        use_cross_attn=False : (B, cond_dim)
        use_cross_attn=True  : (B, n_states * 32, seq_d_model)
                               — n_states × 32 frequency-bin tokens, each seq_d_model-dim.
        """
        B = s_params.shape[0]
        x = s_params.reshape(B * self.n_states, 1, self.n_freq)  # (B*S, 1, F)
        x = self.conv(x)                                          # (B*S, 128, F)
        x = self.pool(x)                                          # (B*S, 128, 32)

        if self.use_cross_attn:
            # Reshape so each (state, freq-bin) pair is a separate token
            x = x.permute(0, 2, 1)                               # (B*S, 32, 128)
            x = x.reshape(B, self.n_states * 32, 128)            # (B, S*32, 128)
            return self.proj(x)                                   # (B, S*32, seq_d_model)
        else:
            x = x.reshape(B, -1)                                  # (B, n_states * 128 * 32)
            return self.mlp(x)                                     # (B, cond_dim)
