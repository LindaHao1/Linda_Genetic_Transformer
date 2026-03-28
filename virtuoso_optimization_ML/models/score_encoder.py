"""
score_encoder.py — MLP encoder for score-based guidance.

Encodes a (B, n_states, n_components) tensor of per-state, per-region penalty
scores into a (B, cond_dim) conditioning vector for the flow matching model.

Score components per state (in order):
  0: lower_reject  — penalty from lower-frequency stopband
  1: passband      — penalty from passband (shortfall below passband_min_db)
  2: higher_reject — penalty from upper-frequency stopband(s)
  3: cross_band    — penalty from cross-state stopband regions

Inputs are log-transformed as log(1 + score) before encoding to compress
the dynamic range (scores can span several orders of magnitude).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class ScoreEncoder(nn.Module):
    """
    Encode per-region penalty scores into a conditioning vector.

    Parameters
    ----------
    n_states         : number of filter states.
    n_components     : number of score components per state (default 4).
    cond_dim         : output conditioning dimension (must match FlowMatchingModel cond_dim).
    """

    def __init__(
        self,
        n_states:     int,
        n_components: int = 4,
        cond_dim:     int = 256,
    ) -> None:
        super().__init__()
        in_dim = n_states * n_components
        hidden = cond_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cond_dim),
        )
        # Initialise with small weights so early conditioning is near-neutral
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, scores: Tensor) -> Tensor:
        """
        Parameters
        ----------
        scores : (B, n_states, n_components) — raw (non-negative) penalty scores.

        Returns
        -------
        (B, cond_dim) conditioning vector.
        """
        B = scores.shape[0]
        # Log-transform to compress dynamic range
        x = torch.log1p(scores.view(B, -1))
        return self.net(x)
