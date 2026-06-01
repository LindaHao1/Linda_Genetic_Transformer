"""
Visualization: best LLM-generated design per Phase-2 iteration (with
world-model-predicted S21 + PV scores), and best simulated design during
calibration.  Reuses virtuoso_engine.plotter._draw_design_matrix for the
pixel grid; falls back to a minimal renderer if that import is unavailable.
Never raises into the training loop.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.scoring import SCORE_SCALARS  # noqa: E402
from src.types import Design  # noqa: E402

_RFIC_ROOT = "/disk/mcao53/RFIC_RL"
if _RFIC_ROOT not in sys.path:
    sys.path.insert(0, _RFIC_ROOT)


def _draw_grid(ax, design: Design) -> None:
    try:
        from virtuoso_engine.plotter import _draw_design_matrix
        _draw_design_matrix(design.pixel_grid, ax,
                            cap_pixel_pairs=design.cap_pixel_pairs or None)
    except Exception:
        ax.imshow(np.asarray(design.pixel_grid), origin="upper",
                  aspect="equal", interpolation="nearest", cmap="viridis")
        ax.set_title("Design Matrix")


def _draw_s21(ax, s21_K: np.ndarray, freqs: np.ndarray, specs) -> None:
    fg = np.asarray(freqs) / 1e9
    spec_by_idx = {s.state_index: s for s in specs}
    for k in range(s21_K.shape[0]):
        ax.plot(fg, s21_K[k], label=f"state {k}")
    # shade state-0 passband (green) / stopbands+cross (red) for reference
    sp = spec_by_idx.get(0, specs[0])
    ax.axvspan(sp.passband[0] / 1e9, sp.passband[1] / 1e9,
               color="green", alpha=0.10)
    for lo, hi in list(sp.stopband_regions) + list(sp.cross_stopband_regions):
        ax.axvspan(lo / 1e9, hi / 1e9, color="red", alpha=0.06)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower center", ncol=min(4, s21_K.shape[0]))


def _annotate_scores(fig, achieved: Optional[np.ndarray],
                     target: Optional[np.ndarray],
                     scalars: Sequence[str]) -> None:
    if achieved is None:
        return
    lines = []
    for k in range(achieved.shape[0]):
        a = ", ".join(f"{n.split('pv_')[-1]}={achieved[k, j]:.2f}"
                      for j, n in enumerate(scalars))
        if target is not None:
            t = ", ".join(f"{target[k, j]:.2f}" for j in range(len(scalars)))
            lines.append(f"S{k}: {a}  | target: {t}")
        else:
            lines.append(f"S{k}: {a}")
    fig.text(0.5, 0.005, "   ".join(lines), ha="center", fontsize=8)


def _plot(design: Design, s21_K: np.ndarray, specs, freqs: np.ndarray,
          title: str, save_path: str,
          achieved: Optional[np.ndarray] = None,
          target: Optional[np.ndarray] = None,
          scalars: Sequence[str] = SCORE_SCALARS) -> Optional[str]:
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5))
        _draw_grid(a0, design)
        _draw_s21(a1, np.asarray(s21_K), np.asarray(freqs), specs)
        a1.set_title(title)
        _annotate_scores(fig, achieved, target, scalars)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path
    except Exception as exc:  # never break training
        print(f"  [viz] skipped ({exc})")
        plt.close("all")
        return None


def plot_best_generated(design: Design, s21_K_pred: np.ndarray, specs,
                        freqs: np.ndarray, iteration: int, out_dir: str,
                        achieved: Optional[np.ndarray] = None,
                        target: Optional[np.ndarray] = None) -> Optional[str]:
    p = os.path.join(out_dir, "plots", f"iter_{iteration:05d}_best_gen.png")
    return _plot(design, s21_K_pred, specs, freqs,
                 f"iter {iteration} — best generated (WM-predicted S21)",
                 p, achieved, target)


def plot_best_simulated(design: Design, s21_K_sim: np.ndarray, specs,
                        freqs: np.ndarray, iteration: int, out_dir: str,
                        achieved: Optional[np.ndarray] = None,
                        target: Optional[np.ndarray] = None) -> Optional[str]:
    p = os.path.join(out_dir, "plots", f"iter_{iteration:05d}_best_sim.png")
    return _plot(design, s21_K_sim, specs, freqs,
                 f"iter {iteration} — best simulated (calibration)",
                 p, achieved, target)


def _draw_s21_overlay(ax, s21_pred: np.ndarray,
                      s21_sim: Optional[np.ndarray],
                      freqs: np.ndarray, specs) -> None:
    fg = np.asarray(freqs) / 1e9
    spec_by_idx = {s.state_index: s for s in specs}
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for k in range(s21_pred.shape[0]):
        c = colors[k % len(colors)]
        ax.plot(fg, s21_pred[k], color=c, linestyle="-",
                label=f"state {k} pred")
        if s21_sim is not None and k < s21_sim.shape[0]:
            ax.plot(fg, s21_sim[k], color=c, linestyle="--",
                    label=f"state {k} sim")
    sp = spec_by_idx.get(0, specs[0])
    ax.axvspan(sp.passband[0] / 1e9, sp.passband[1] / 1e9,
               color="green", alpha=0.10)
    for lo, hi in list(sp.stopband_regions) + list(sp.cross_stopband_regions):
        ax.axvspan(lo / 1e9, hi / 1e9, color="red", alpha=0.06)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower center",
              ncol=min(6, 2 * s21_pred.shape[0]))


def _fmt_score_row(scalars, vec):
    return " ".join(f"{n.split('pv_')[-1]}={vec[j]:+.2f}"
                    for j, n in enumerate(scalars))


def plot_test_generation(
    design: Design,
    s21_pred: np.ndarray,                   # (K, n_freq)
    s21_sim: Optional[np.ndarray],          # (K, n_freq) or None
    specs,
    freqs: np.ndarray,
    scores_requested: np.ndarray,           # (K, n_scalars)
    scores_pred: np.ndarray,                # (K, n_scalars)
    scores_sim: Optional[np.ndarray],       # (K, n_scalars) or None
    save_path: str,
    scalars: Sequence[str] = SCORE_SCALARS,
) -> Optional[str]:
    """End-to-end test-generation plot: grid + predicted vs simulated S21 +
    per-state requested/predicted/simulated PV-score readouts."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 6))
        _draw_grid(a0, design)
        _draw_s21_overlay(a1, np.asarray(s21_pred), s21_sim,
                          np.asarray(freqs), specs)
        a1.set_title("LM-generated design — S21 (solid: world model, dashed: sim)")
        lines = []
        K = s21_pred.shape[0]
        for k in range(K):
            req = _fmt_score_row(scalars, scores_requested[k])
            pred = _fmt_score_row(scalars, scores_pred[k])
            row = f"S{k}: req[{req}] | pred[{pred}]"
            if scores_sim is not None:
                sim = _fmt_score_row(scalars, scores_sim[k])
                row += f" | sim[{sim}]"
            lines.append(row)
        fig.text(0.5, 0.005, "\n".join(lines), ha="center", fontsize=8,
                 family="monospace")
        fig.tight_layout(rect=(0, 0.04 + 0.018 * K, 1, 1))
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path
    except Exception as exc:
        print(f"  [viz] plot_test_generation skipped ({exc})")
        plt.close("all")
        return None
