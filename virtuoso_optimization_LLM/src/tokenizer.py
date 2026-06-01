"""
Token vocabulary + Tokenizer.

Sequence layout (spec is a CONTINUOUS prefix handled by the model, not
tokenized here -- only the design portion is tokenized):

    BOS  SEP_spec
    <grid: rows*cols cells in raster row-major; per cell:
        - GROUND / METAL / VIA / PORT                 (1 token)
        - SWITCH         VAL_b                        (2 tokens, inline)
        - INDUCTOR       INDVAL_i  VAL_b              (3 tokens, inline)
        - CAP_PAIR_{H|V}{L}  VAL_b                    (2 tokens, head cell only)
        - interior/dest cells of an already-started pair emit
          GROUND / METAL respectively (regular pixel tokens)>
    SEP_grid
    <states: K*(n_sw+n_ind) STATE_*; state-major, switches raster then inductors raster>
    SEP_states
    EOS

`decode` returns a record dict identical in shape to
`world_model_test/encode.py`'s expected input, so it round-trips through the
surrogate without adaptation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

# Pixel codes (must match world_model_test/encode.py)
P_GROUND, P_METAL, P_INDUCTOR, P_SWITCH, P_VIA = 0, 1, 2, 3, 4


@dataclass
class TokenizerConfig:
    rows: int = 21
    cols: int = 27
    n_states: int = 3                 # K (configurable, default 3)
    n_bins: int = 64                  # value discretization
    n_inductor_values: int = 1        # categorical inductor model count
    cap_min: float = 100e-15
    cap_max: float = 1000e-15
    inductor_cap_min: float = 200e-15
    inductor_cap_max: float = 1000e-15
    cap_pair_min: float = 50e-15
    cap_pair_max: float = 1000e-15
    cap_pair_min_dist: int = 2        # Chebyshev length range for a pair
    cap_pair_max_dist: int = 3
    max_cap_pairs: int = 8

    @property
    def lengths(self) -> List[int]:
        return list(range(self.cap_pair_min_dist, self.cap_pair_max_dist + 1))


# Token-family prefixes used in the vocabulary names
_STRUCTURAL = [
    "BOS", "EOS", "PAD",
    "SEP_spec", "SEP_grid", "SEP_states",
]
_PIXEL = ["GROUND", "METAL", "INDUCTOR", "SWITCH", "VIA", "PORT"]
_STATE = ["STATE_OFF", "STATE_ON"]

_PIXEL_CODE_TO_NAME = {
    P_GROUND: "GROUND", P_METAL: "METAL", P_INDUCTOR: "INDUCTOR",
    P_SWITCH: "SWITCH", P_VIA: "VIA",
}


def _cap_pair_names(lengths: List[int]) -> List[str]:
    return [f"CAP_PAIR_{o}{L}" for o in ("H", "V") for L in lengths]


class Tokenizer:
    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        names: List[str] = []
        names += _STRUCTURAL
        names += _PIXEL
        names += _cap_pair_names(self.config.lengths)
        names += [f"VAL_{i}" for i in range(self.config.n_bins)]
        names += [f"INDVAL_{i}" for i in range(max(self.config.n_inductor_values, 1))]
        names += _STATE
        self._names = names
        self.tok2id: Dict[str, int] = {n: i for i, n in enumerate(names)}
        self.id2tok: Dict[int, str] = {i: n for i, n in enumerate(names)}

        self._cap_pair_names = set(_cap_pair_names(self.config.lengths))

    # ---- vocabulary ------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._names)

    def t(self, name: str) -> int:
        return self.tok2id[name]

    def get_token_type(self, token_id: int) -> str:
        n = self.id2tok[token_id]
        if n in _STRUCTURAL:
            return "structural"
        if n in _PIXEL:
            return "pixel"
        if n in self._cap_pair_names:
            return "pixel_cap_pair"
        if n.startswith("VAL_"):
            return "value"
        if n.startswith("INDVAL_"):
            return "indval"
        if n in _STATE:
            return "state"
        return "unknown"

    # ---- position helpers ------------------------------------------------

    def rc_to_pos(self, r: int, c: int) -> int:
        return r * self.config.cols + c

    def pos_to_rc(self, pos: int) -> Tuple[int, int]:
        return divmod(pos, self.config.cols)

    # ---- value (de)quantization -----------------------------------------

    def _bin(self, v: float, lo: float, hi: float) -> int:
        if hi <= lo:
            return 0
        frac = (v - lo) / (hi - lo)
        b = int(frac * self.config.n_bins)
        return max(0, min(self.config.n_bins - 1, b))

    def _unbin(self, b: int, lo: float, hi: float) -> float:
        return lo + (b + 0.5) / self.config.n_bins * (hi - lo)

    def value_range(self, family: str) -> Tuple[float, float]:
        c = self.config
        if family == "switch":
            return c.cap_min, c.cap_max
        if family == "inductor_cap":
            return c.inductor_cap_min, c.inductor_cap_max
        if family == "cap_pair":
            return c.cap_pair_min, c.cap_pair_max
        raise ValueError(f"unknown value family {family!r}")

    # ---- cap-pair helpers -----------------------------------------------

    @staticmethod
    def _pair_path(hr: int, hc: int, orient: str, length: int
                   ) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        """Return (destination, interior_cells) for head + orient + length."""
        if orient == "H":
            dest = (hr, hc + length)
            interior = [(hr, hc + k) for k in range(1, length)]
        else:
            dest = (hr + length, hc)
            interior = [(hr + k, hc) for k in range(1, length)]
        return dest, interior

    # ---- encode ----------------------------------------------------------

    def _pixel_token(self, code: int, r: int, c: int,
                     head_lookup: Dict[Tuple[int, int], Tuple[str, int]]
                     ) -> int:
        if (r, c) in head_lookup:
            orient, length = head_lookup[(r, c)]
            return self.t(f"CAP_PAIR_{orient}{length}")
        if code == P_METAL and (c == 0 or c == self.config.cols - 1):
            return self.t("PORT")
        return self.t(_PIXEL_CODE_TO_NAME[code])

    def encode(self, record: Dict[str, Any]) -> List[int]:
        """record: same shape as Design.to_record() / encode.py input."""
        cfg = self.config
        matrix = record["matrix"]
        cap_values = record.get("cap_values", [])
        inductor_values = record.get("inductor_values", [])
        inductor_cap_values = record.get("inductor_cap_values", [])
        cap_pixel_pairs = record.get("cap_pixel_pairs", [])
        cap_pair_values = record.get("cap_pair_values", [])
        switch_states = record.get("switch_states", [])
        inductor_switch_states = record.get("inductor_switch_states", [])

        # validate + build head -> (orient, length) and head -> value lookups
        head_lookup: Dict[Tuple[int, int], Tuple[str, int]] = {}
        head_value: Dict[Tuple[int, int], float] = {}
        rows, cols = cfg.rows, cfg.cols
        for idx, pair in enumerate(cap_pixel_pairs):
            (r1, c1), (r2, c2) = pair[0], pair[1]
            if r1 != r2 and c1 != c2:
                raise ValueError(
                    f"cap pair {pair} is not axis-aligned; this tokenizer only "
                    f"supports H/V pairs (TODO(user): diagonal support).")
            if self.rc_to_pos(r1, c1) <= self.rc_to_pos(r2, c2):
                (hr, hc), (dr, dc) = (r1, c1), (r2, c2)
            else:
                (hr, hc), (dr, dc) = (r2, c2), (r1, c1)
            length = max(abs(dr - hr), abs(dc - hc))
            if length not in cfg.lengths:
                raise ValueError(
                    f"cap pair length {length} outside "
                    f"[{cfg.cap_pair_min_dist},{cfg.cap_pair_max_dist}]")
            orient = "H" if hr == dr else "V"
            # path-validity check (matches generation constraint; interior
            # is NOT enforced here because real training data may have non-
            # ground interior cells.  The constraint mask enforces the stricter
            # interior-all-ground rule only at generation time.)
            dest, _interior = self._pair_path(hr, hc, orient, length)
            if not (0 <= dest[0] < rows and 0 <= dest[1] < cols):
                raise ValueError(f"cap pair {pair} dest out of bounds")
            if matrix[hr][hc] != P_METAL or matrix[dest[0]][dest[1]] != P_METAL:
                raise ValueError(
                    f"cap pair {pair} endpoints must be METAL (got "
                    f"head={matrix[hr][hc]}, dest={matrix[dest[0]][dest[1]]})")
            if (hr, hc) in head_lookup:
                raise ValueError(f"duplicate cap-pair head at ({hr},{hc})")
            head_lookup[(hr, hc)] = (orient, length)
            val = (cap_pair_values[idx] if idx < len(cap_pair_values)
                   else cfg.cap_pair_min)
            head_value[(hr, hc)] = float(val)

        out: List[int] = [self.t("BOS"), self.t("SEP_spec")]

        # raster walk: pixel + inline VAL/INDVAL
        sw_i = ind_i = 0
        for r in range(rows):
            for c in range(cols):
                code = matrix[r][c]
                tok_id = self._pixel_token(code, r, c, head_lookup)
                out.append(tok_id)
                name = self.id2tok[tok_id]
                if name == "SWITCH":
                    v = cap_values[sw_i] if sw_i < len(cap_values) else cfg.cap_min
                    out.append(self.t(f"VAL_{self._bin(v, cfg.cap_min, cfg.cap_max)}"))
                    sw_i += 1
                elif name == "INDUCTOR":
                    iv = (inductor_values[ind_i]
                          if ind_i < len(inductor_values) else 0)
                    iv = max(0, min(max(cfg.n_inductor_values, 1) - 1, int(iv)))
                    out.append(self.t(f"INDVAL_{iv}"))
                    mc = (inductor_cap_values[ind_i]
                          if ind_i < len(inductor_cap_values)
                          else cfg.inductor_cap_min)
                    out.append(self.t(
                        f"VAL_{self._bin(mc, cfg.inductor_cap_min, cfg.inductor_cap_max)}"))
                    ind_i += 1
                elif name in self._cap_pair_names:
                    val = head_value[(r, c)]
                    out.append(self.t(
                        f"VAL_{self._bin(val, cfg.cap_pair_min, cfg.cap_pair_max)}"))

        out.append(self.t("SEP_grid"))

        # states: K * (switches raster, then inductors raster)
        n_sw = sum(v == P_SWITCH for row in matrix for v in row)
        n_ind = sum(v == P_INDUCTOR for row in matrix for v in row)
        for s in range(cfg.n_states):
            row_sw = switch_states[s] if s < len(switch_states) else [0] * n_sw
            for k in range(n_sw):
                bit = row_sw[k] if k < len(row_sw) else 0
                out.append(self.t("STATE_ON" if bit else "STATE_OFF"))
            row_ind = (inductor_switch_states[s]
                       if s < len(inductor_switch_states) else [1] * n_ind)
            for j in range(n_ind):
                bit = row_ind[j] if j < len(row_ind) else 1
                out.append(self.t("STATE_ON" if bit else "STATE_OFF"))
        out.append(self.t("SEP_states"))
        out.append(self.t("EOS"))
        return out

    # ---- decode ----------------------------------------------------------

    def decode(self, tokens: List[int]) -> Dict[str, Any]:
        cfg = self.config
        names = [self.id2tok[t] for t in tokens]
        i = names.index("SEP_spec") + 1

        rows, cols = cfg.rows, cfg.cols
        matrix = [[P_GROUND] * cols for _ in range(rows)]
        cap_values: List[float] = []
        inductor_values: List[int] = []
        inductor_cap_values: List[float] = []
        cap_pixel_pairs: List[List[List[int]]] = []
        cap_pair_values: List[float] = []

        # walk rows*cols cells in raster order; one pixel-family token per cell.
        for cell in range(rows * cols):
            r, c = divmod(cell, cols)
            n = names[i]; i += 1
            if n == "PORT":
                matrix[r][c] = P_METAL
            elif n == "GROUND":
                matrix[r][c] = P_GROUND
            elif n == "METAL":
                matrix[r][c] = P_METAL
            elif n == "INDUCTOR":
                matrix[r][c] = P_INDUCTOR
                iv = int(names[i].split("_")[1]); i += 1
                inductor_values.append(iv)
                b = int(names[i].split("_")[1]); i += 1
                inductor_cap_values.append(
                    self._unbin(b, cfg.inductor_cap_min, cfg.inductor_cap_max))
            elif n == "SWITCH":
                matrix[r][c] = P_SWITCH
                b = int(names[i].split("_")[1]); i += 1
                cap_values.append(self._unbin(b, cfg.cap_min, cfg.cap_max))
            elif n == "VIA":
                matrix[r][c] = P_VIA
            elif n in self._cap_pair_names:
                # CAP_PAIR_{H|V}{L}
                orient = n[len("CAP_PAIR_")]
                length = int(n[len("CAP_PAIR_") + 1:])
                matrix[r][c] = P_METAL
                dest, interior = self._pair_path(r, c, orient, length)
                # interior cells will be set to GROUND, dest to METAL when
                # their pixel tokens (GROUND/METAL/PORT) are processed later
                # in the raster walk -- we just record the pair geometry.
                cap_pixel_pairs.append([[r, c], [dest[0], dest[1]]])
                b = int(names[i].split("_")[1]); i += 1
                cap_pair_values.append(
                    self._unbin(b, cfg.cap_pair_min, cfg.cap_pair_max))
            else:
                raise ValueError(f"expected pixel token at cell ({r},{c}), got {n}")

        assert names[i] == "SEP_grid", names[i]
        i += 1

        # states
        n_sw = sum(v == P_SWITCH for row in matrix for v in row)
        n_ind = sum(v == P_INDUCTOR for row in matrix for v in row)
        switch_states: List[List[int]] = []
        inductor_switch_states: List[List[int]] = []
        for _s in range(cfg.n_states):
            sw_row = []
            for _k in range(n_sw):
                sw_row.append(1 if names[i] == "STATE_ON" else 0); i += 1
            ind_row = []
            for _j in range(n_ind):
                ind_row.append(1 if names[i] == "STATE_ON" else 0); i += 1
            switch_states.append(sw_row)
            inductor_switch_states.append(ind_row)
        assert names[i] == "SEP_states", names[i]

        return {
            "matrix": matrix,
            "cap_values": cap_values,
            "inductor_values": inductor_values,
            "inductor_cap_values": inductor_cap_values,
            "cap_pixel_pairs": cap_pixel_pairs,
            "cap_pair_values": cap_pair_values,
            "switch_states": switch_states,
            "inductor_switch_states": inductor_switch_states,
        }

    # ---- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"config": asdict(self.config), "tokens": self._names},
                      f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path) as f:
            blob = json.load(f)
        tok = cls(TokenizerConfig(**blob["config"]))
        assert tok._names == blob["tokens"], "vocab drift vs saved tokenizer"
        return tok
