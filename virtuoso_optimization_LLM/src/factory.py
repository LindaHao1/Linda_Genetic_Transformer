"""Build tokenizer / model / surrogate from a config dict."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import yaml

from src.model import FilterLM, ModelConfig
from src.surrogate import Surrogate
from src.tokenizer import Tokenizer, TokenizerConfig


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def freq_points(cfg: Dict[str, Any]) -> np.ndarray:
    fq = cfg["freq"]
    return np.linspace(float(fq["min"]), float(fq["max"]), int(fq["n"]))


def build_tokenizer(cfg: Dict[str, Any]) -> Tokenizer:
    g = cfg["grid"]
    return Tokenizer(TokenizerConfig(
        rows=g["rows"], cols=g["cols"], n_states=g["n_states"],
        n_bins=cfg["tokenizer"]["n_bins"],
        n_inductor_values=g.get("n_inductor_values", 1),
        cap_min=float(g["cap_min"]), cap_max=float(g["cap_max"]),
        inductor_cap_min=float(g["inductor_cap_min"]),
        inductor_cap_max=float(g["inductor_cap_max"]),
        cap_pair_min=float(g["cap_pair_min"]),
        cap_pair_max=float(g["cap_pair_max"]),
        cap_pair_min_dist=g["cap_pair_min_dist"],
        cap_pair_max_dist=g["cap_pair_max_dist"],
        max_cap_pairs=g["max_cap_pairs"],
    ))


def build_model(cfg: Dict[str, Any], tok: Tokenizer) -> FilterLM:
    m = cfg["model"]
    scalars = cfg["spec"]["score_scalars"]
    return FilterLM(tok, ModelConfig(
        model_dim=m["model_dim"], n_layers=m["n_layers"],
        n_heads=m["n_heads"], ffn_mult=m["ffn_mult"],
        dropout=m["dropout"], max_seq_len=m["max_seq_len"],
        spec_dropout=m["spec_dropout"], spec_score_dim=len(scalars),
        n_freq=int(cfg["freq"]["n"]),
    ))


def build_specs(cfg: Dict[str, Any]):
    from src.scoring import load_specs
    return load_specs(cfg["scoring"]["filter_specs_config"])


def build_score_norm(cfg: Dict[str, Any]):
    from src.scoring import ScoreNorm
    return ScoreNorm.from_config(cfg["spec"].get("normalization", {}),
                                 cfg["spec"]["score_scalars"])


def build_surrogate(cfg: Dict[str, Any]) -> Surrogate:
    s = cfg["surrogate"]
    return Surrogate(
        encode_config=cfg,                       # has cfg["grid"]
        checkpoint_path=s.get("checkpoint_path"),
        ensemble_size=s.get("ensemble_size", 5),
        n_freq=int(cfg["freq"]["n"]),
        device=s.get("device", "cpu"),
    )
