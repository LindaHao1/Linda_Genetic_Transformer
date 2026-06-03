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

    def _sa(self, x, past_kv, is_causal):
        """Self-attention core, optionally concatenating a past K/V cache.
        Returns (attn_out, present_kv) where present_kv is the (k, v) including
        this step's tokens -- ready to be reused as the next step's cache."""
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, D // self.h).transpose(1, 2)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        a = a.transpose(1, 2).reshape(B, T, D)
        return a, present

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training / full-sequence path: causal attention, no cache."""
        a, _ = self._sa(x, None, is_causal=True)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

    def step(self, x: torch.Tensor, k_buf: torch.Tensor, v_buf: torch.Tensor,
             l_prev: int, l_new: int) -> torch.Tensor:
        """Incremental decode path with a PRE-ALLOCATED KV cache.

        Writes this chunk's K/V into `k_buf`/`v_buf` at [l_prev:l_new] (in
        place, no realloc) and attends the query over the live cache [:l_new].
        `l_prev == 0` is the prime chunk (prefix + BOS) and needs intra-chunk
        causal masking; later calls pass a single token (T=1) attending the
        whole cache, so no masking is needed.  Avoids the O(T^2) `torch.cat`
        that growing-cache decoding incurs."""
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        nh, hd = self.h, D // self.h
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        k_buf[:, :, l_prev:l_new] = k
        v_buf[:, :, l_prev:l_new] = v
        a = F.scaled_dot_product_attention(
            q, k_buf[:, :, :l_new], v_buf[:, :, :l_new],
            is_causal=(l_prev == 0))
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

    # ---- incremental embedding (KV-cache decode) ------------------------

    def _pixel_name_set(self) -> set:
        return ({"GROUND", "METAL", "INDUCTOR", "SWITCH", "VIA", "PORT"}
                | self.tok._cap_pair_names)

    @staticmethod
    def _advance_decode_state(st: dict, name: str, pixel_names: set,
                              cols: int) -> "tuple[int, bool, int, int]":
        """Update per-sequence decode state for one token and return
        (segment_id, add_row_col, row, col).  Mirrors `segment_ids_for` +
        the cursor logic in `_token_embeddings` exactly, one token at a time."""
        if name in ("BOS", "EOS", "PAD"):
            if name == "BOS":
                st["phase"] = "spec"
            return SEG_STRUCT, False, 0, 0
        if name == "SEP_spec":
            st["phase"] = "grid";   return SEG_STRUCT, False, 0, 0
        if name == "SEP_grid":
            st["phase"] = "states"; return SEG_STRUCT, False, 0, 0
        if name == "SEP_states":
            st["phase"] = "done";   return SEG_STRUCT, False, 0, 0
        if st["phase"] == "grid":
            if name in pixel_names:                       # cursor advances on a pixel
                st["cur_r"], st["cur_c"] = divmod(st["cursor"], cols)
                st["cursor"] += 1
            return SEG_GRID, True, st["cur_r"], st["cur_c"]
        if st["phase"] == "states":
            return SEG_STATES, False, 0, 0
        return SEG_STRUCT, False, 0, 0

    def _embed_step(self, token_ids: torch.Tensor, position: int,
                    states: List[dict], pixel_names: set) -> torch.Tensor:
        """Embed a single new token per sequence -> (B,1,d).  Equivalent to the
        slice of `_token_embeddings` for one column, given each sequence's
        running decode state."""
        dev = token_ids.device
        B = token_ids.shape[0]
        cols = self.tok.config.cols
        pos = min(position, self.mcfg.max_seq_len - 1)
        x = self.token_emb(token_ids) + self.pos_emb(
            torch.tensor(pos, device=dev))               # (B,d)
        seg = torch.empty(B, dtype=torch.long, device=dev)
        add_rc = torch.zeros(B, dtype=torch.bool, device=dev)
        rr = torch.zeros(B, dtype=torch.long, device=dev)
        cc = torch.zeros(B, dtype=torch.long, device=dev)
        for b in range(B):
            name = self.tok.id2tok[int(token_ids[b])]
            seg_id, grid, r, c = self._advance_decode_state(
                states[b], name, pixel_names, cols)
            seg[b] = seg_id; add_rc[b] = grid; rr[b] = r; cc[b] = c
        x = x + self.seg_emb(seg)
        rc = self.row_emb(rr) + self.col_emb(cc)
        x = x + torch.where(add_rc[:, None], rc, torch.zeros_like(rc))
        return x.unsqueeze(1)                            # (B,1,d)

    def _alloc_kv(self, B: int, cap: int, device, dtype):
        """Per-layer pre-allocated (k_buf, v_buf), each (B, n_heads, cap, hd)."""
        nh = self.mcfg.n_heads
        hd = self.mcfg.model_dim // nh
        return [(torch.empty(B, nh, cap, hd, device=device, dtype=dtype),
                 torch.empty(B, nh, cap, hd, device=device, dtype=dtype))
                for _ in range(self.mcfg.n_layers)]

    def _run_cached(self, x: torch.Tensor, kv_bufs, l_prev: int):
        """Run all blocks over `x` (B,T,d), writing into the pre-allocated KV
        buffers at [l_prev : l_prev+T].  Returns (logits (B,T,V), l_new)."""
        l_new = l_prev + x.shape[1]
        for (k_buf, v_buf), blk in zip(kv_bufs, self.blocks):
            x = blk.step(x, k_buf, v_buf, l_prev, l_new)
        x = self.ln_f(x)
        return self.head(x), l_new

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
        dev = self.device
        B = spec.shape[0]
        bos, eos, pad = self.tok.t("BOS"), self.tok.t("EOS"), self.tok.t("PAD")
        use_cfg = cfg_scale != 1.0
        pixel_names = self._pixel_name_set()

        seqs = [[bos] for _ in range(B)]
        done = [False] * B
        states = [{"phase": "pre", "cursor": 0, "cur_r": 0, "cur_c": 0}
                  for _ in range(B)]

        # pre-allocate the KV cache: capacity = spec prefix (K) + token budget
        K = spec.shape[1]
        cap = K + max_len
        dtype = self.token_emb.weight.dtype
        cond_kv = self._alloc_kv(B, cap, dev, dtype)
        unc_kv = self._alloc_kv(B, cap, dev, dtype) if use_cfg else None

        # ---- prime step: process [spec prefix ; BOS], fill the KV cache ----
        bos_ids = torch.full((B,), bos, dtype=torch.long, device=dev)
        bos_emb = self._embed_step(bos_ids, 0, states, pixel_names)   # (B,1,d)

        def _prime(kv, drop_val: float):
            drop = torch.full((B,), bool(drop_val), dtype=torch.bool, device=dev)
            prefix = self._spec_prefix(spec, drop)                    # (B,K,d)
            x = torch.cat([prefix, bos_emb], dim=1)                   # (B,K+1,d)
            logits, l_new = self._run_cached(x, kv, 0)
            return logits[:, -1], l_new                               # BOS-slot logits

        # cond cache (spec on); uncond cache (spec off) only when CFG is active
        cond_logits, cache_len = _prime(cond_kv, False)
        if use_cfg:
            unc_logits, _ = _prime(unc_kv, True)
        else:
            unc_logits = None

        def _sample(logits_c, logits_u):
            for b in range(B):
                if done[b]:
                    continue
                lg = logits_c[b]
                if use_cfg:
                    lg = logits_u[b] + cfg_scale * (lg - logits_u[b])
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

        _sample(cond_logits, unc_logits)

        # ---- incremental decode: one new token per step over the cache ----
        for step in range(1, max_len - 1):
            if all(done):
                break
            # last token of each sequence (PAD for finished ones keeps shapes)
            last_ids = torch.tensor(
                [seqs[b][-1] if not done[b] else pad for b in range(B)],
                dtype=torch.long, device=dev)
            x_new = self._embed_step(last_ids, step, states, pixel_names)
            l_prev = cache_len            # cond & uncond share the same length
            cond_logits, cache_len = self._run_cached(x_new, cond_kv, l_prev)
            cond_logits = cond_logits[:, -1]
            if use_cfg:
                unc_logits, _ = self._run_cached(x_new, unc_kv, l_prev)
                unc_logits = unc_logits[:, -1]
            _sample(cond_logits, unc_logits)
        return seqs
