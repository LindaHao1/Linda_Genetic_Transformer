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
    Encodes target S-parameter curves into a conditioning vector.

    Parameters
    ----------
    n_states : number of filter states.
    n_freq   : number of frequency points in the canonical grid.
    cond_dim : dimension of the output conditioning vector.
    """

    def __init__(self, n_states: int, n_freq: int, cond_dim: int = 256) -> None:
        super().__init__()
        self.n_states = n_states
        self.n_freq   = n_freq
        self.cond_dim = cond_dim

        # 1-D CNN over the frequency axis
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64,  kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(32)   # → (*, 128, 32)

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
        (B, cond_dim)
        """
        B = s_params.shape[0]
        x = s_params.reshape(B * self.n_states, 1, self.n_freq)  # (B*S, 1, F)
        x = self.conv(x)                       # (B*S, 128, F)
        x = self.pool(x)                       # (B*S, 128, 32)
        x = x.reshape(B, -1)                  # (B, n_states * 128 * 32)
        return self.mlp(x)                     # (B, cond_dim)
