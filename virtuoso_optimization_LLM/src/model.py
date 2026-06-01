"""
Decoder-only transformer (GPT-style) for RF filter layout generation.

Conditioning: the multi-state target S21 (K x n_freq, dB) is encoded by a small
MLP into K continuous "spec" embeddings, prepended as a causal prefix
(prefix-LM).  Classifier-free guidance: with prob `spec_dropout` the spec is
replaced by a learned null embedding during training; `cfg_scale` mixes the
conditional and unconditional logits at generation time.

Positional info: a base 1-D learned position embedding for every token, plus an
additional 2-D (row, col) learned embedding on the grid segment, plus a learned
segment embedding (spec/grid/values/cappairs/states/structural).

`generate_batch` applies `get_invalid_token_mask` at every step
(`logits[mask] = -inf`) so only grammar/constraint-valid tokens are sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constraints import ConstraintConfig, get_invalid_token_mask
from src.tokenizer import Tokenizer

# segment ids
SEG_STRUCT, SEG_SPEC, SEG_GRID, SEG_STATES = range(4)
_N_SEG = 4


@dataclass
class ModelConfig:
    model_dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    ffn_mult: int = 4
    dropout: float = 0.0
    max_seq_len: int = 1024          # token positions (excl. spec prefix)
    spec_dropout: float = 0.15       # CFG: prob of dropping spec in training
    spec_score_dim: int = 3          # per-state conditioning scalars (PV scores)
    n_freq: int = 71                 # surrogate S21 width (NOT the spec encoder)


def segment_ids_for(names: List[str], tok: Tokenizer) -> List[int]:
    """Per-token segment id, derived from SEP markers. len == len(names)."""
    seg = []
    phase = "pre"
    for n in names:
        if n in ("BOS", "EOS", "PAD"):
            seg.append(SEG_STRUCT)
            if n == "BOS":
                phase = "spec"
            continue
        if n == "SEP_spec":
            seg.append(SEG_STRUCT); phase = "grid"; continue
        if n == "SEP_grid":
            seg.append(SEG_STRUCT); phase = "states"; continue
        if n == "SEP_states":
            seg.append(SEG_STRUCT); phase = "done"; continue
        if phase == "grid":
            # inline VAL/INDVAL share the grid segment with their pixel
            seg.append(SEG_GRID)
        elif phase == "states":
            seg.append(SEG_STATES)
        else:
            seg.append(SEG_STRUCT)
    return seg


class _Block(nn.Module):
    def __init__(self, d: int, h: int, mult: int, p: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))
        self.h = h
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class FilterLM(nn.Module):
    def __init__(self, tokenizer: Tokenizer, mcfg: ModelConfig) -> None:
        super().__init__()
        self.tok = tokenizer
        self.mcfg = mcfg
        d = mcfg.model_dim
        V = tokenizer.vocab_size
        tcfg = tokenizer.config

        self.token_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Embedding(mcfg.max_seq_len, d)
        self.row_emb = nn.Embedding(tcfg.rows, d)
        self.col_emb = nn.Embedding(tcfg.cols, d)
        self.seg_emb = nn.Embedding(_N_SEG, d)

        # spec encoder: per-state PV-score vector -> d ; learned null for CFG
        self.spec_mlp = nn.Sequential(
            nn.Linear(mcfg.spec_score_dim, d), nn.GELU(), nn.Linear(d, d))
        self.null_spec = nn.Parameter(torch.zeros(1, 1, d))

        self.blocks = nn.ModuleList(
            [_Block(d, mcfg.n_heads, mcfg.ffn_mult, mcfg.dropout)
             for _ in range(mcfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.token_emb.weight   # tie

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.token_emb.weight.device

    # ---- embedding assembly ---------------------------------------------

    def _spec_prefix(self, spec: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
        """spec: (B,K,spec_score_dim) -> (B,K,d); `drop` (B,) replaces w/ null."""
        e = self.spec_mlp(spec)                       # (B,K,d)
        null = self.null_spec.expand_as(e)
        return torch.where(drop[:, None, None], null, e)

    def _token_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        x = self.token_emb(tokens)
        pos = torch.arange(T, device=tokens.device)
        x = x + self.pos_emb(pos)[None]
        cols = self.tok.config.cols
        pixel_names = {"GROUND", "METAL", "INDUCTOR", "SWITCH", "VIA", "PORT"}
        pixel_names |= self.tok._cap_pair_names
        # per-sample 2D grid + segment embeddings.  Inline VAL/INDVAL share the
        # raster (r,c) of the preceding pixel-family token; the cursor advances
        # only on pixel-family tokens.
        for b in range(B):
            names = [self.tok.id2tok[int(t)] for t in tokens[b].tolist()]
            seg = torch.tensor(segment_ids_for(names, self.tok),
                               device=tokens.device)
            x[b] = x[b] + self.seg_emb(seg)
            cursor = 0
            cur_r, cur_c = 0, 0
            for p, n in enumerate(names):
                if seg[p].item() != SEG_GRID:
                    continue
                if n in pixel_names:
                    cur_r, cur_c = divmod(cursor, cols)
                    cursor += 1
                x[b, p] = x[b, p] + self.row_emb.weight[cur_r] + self.col_emb.weight[cur_c]
        return x

    def forward(self, tokens: torch.Tensor, spec: torch.Tensor,
                spec_drop: Optional[torch.Tensor] = None) -> torch.Tensor:
        """tokens (B,T); spec (B,K,spec_score_dim) -> logits (B,T,V)."""
        B = tokens.shape[0]
        if spec_drop is None:
            if self.training and self.mcfg.spec_dropout > 0:
                spec_drop = (torch.rand(B, device=tokens.device)
                             < self.mcfg.spec_dropout)
            else:
                spec_drop = torch.zeros(B, dtype=torch.bool,
                                        device=tokens.device)
        prefix = self._spec_prefix(spec, spec_drop)        # (B,K,d)
        tok_x = self._token_embeddings(tokens)             # (B,T,d)
        x = torch.cat([prefix, tok_x], dim=1)              # (B,K+T,d)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits[:, prefix.shape[1]:]                 # token slots only

    # ---- generation ------------------------------------------------------

    @torch.no_grad()
    def generate_batch(
        self,
        spec: torch.Tensor,               # (B,K,spec_score_dim)
        max_len: int = 1024,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        apply_constraints: bool = True,
        constraint_config: Optional[ConstraintConfig] = None,
    ) -> List[List[int]]:
        self.eval()
        B = spec.shape[0]
        bos, eos = self.tok.t("BOS"), self.tok.t("EOS")
        seqs = [[bos] for _ in range(B)]
        done = [False] * B

        for _ in range(max_len - 1):
            if all(done):
                break
            T = max(len(s) for s in seqs)
            pad = self.tok.t("PAD")
            batch = torch.tensor([s + [pad] * (T - len(s)) for s in seqs],
                                 device=self.device)
            cond = self.forward(batch, spec,
                                spec_drop=torch.zeros(B, dtype=torch.bool,
                                                      device=self.device))
            if cfg_scale != 1.0:
                uncond = self.forward(batch, spec,
                                      spec_drop=torch.ones(B, dtype=torch.bool,
                                                           device=self.device))
            for b in range(B):
                if done[b]:
                    continue
                last = len(seqs[b]) - 1
                lg = cond[b, last]
                if cfg_scale != 1.0:
                    lg = uncond[b, last] + cfg_scale * (lg - uncond[b, last])
                if apply_constraints:
                    m = get_invalid_token_mask(seqs[b], self.tok,
                                               constraint_config)
                    lg = lg.masked_fill(m.to(lg.device), float("-inf"))
                if temperature <= 0:
                    nxt = int(lg.argmax())
                else:
                    p = torch.softmax(lg / temperature, dim=-1)
                    nxt = int(torch.multinomial(p, 1))
                seqs[b].append(nxt)
                if nxt == eos:
                    done[b] = True
        return seqs
