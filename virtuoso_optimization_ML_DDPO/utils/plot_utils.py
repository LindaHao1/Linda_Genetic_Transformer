"""
plot_utils.py — Plotting helpers for the online ML training loop.

Functions
---------
plot_iter_wm        : Save an S-param superposition plot for either the best
                      or worst world-model-predicted design of one iteration.
plot_iter_best      : Convenience wrapper — calls plot_iter_wm(mode="best").
plot_iter_worst     : Convenience wrapper — calls plot_iter_wm(mode="worst").
plot_iter_best_sim  : Save an S-param plot for the best simulated design
                      in this iteration's simulation batch.
plot_wm_vs_sim      : Compare WM predictions vs real simulation results for
                      entries processed during the correction step.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal shared WM plot helper
# ---------------------------------------------------------------------------

def plot_iter_wm(
    feasible_designs: List,
    mean_pred: torch.Tensor,
    wm_costs: List[float],
    current_iter: int,
    freq_points: np.ndarray,
    target_s: torch.Tensor,
    output_dir: Path,
    mode: Literal["best", "worst"] = "best",
    tb_writer=None,
    debug: bool = False,
) -> None:
    """
    Plot WM-predicted S21 curves (all states superimposed) for either the
    design with the lowest (mode="best") or highest (mode="worst") proxy cost
    in this iteration.

    Saved to ``output_dir/plots/iter_{N:05d}_{best|worst}_wm.png``.

    Parameters
    ----------
    feasible_designs : list of (DesignState, image_tensor) from Step 1.
    mean_pred        : (n, n_states, n_freq) WM prediction tensor.
    wm_costs         : proxy cost for each design (same length as feasible_designs).
    current_iter     : training iteration index (used in file name and title).
    freq_points      : (n_freq,) frequency grid in Hz.
    target_s         : (n_states, n_freq) target S-param conditioning tensor.
    output_dir       : base output directory; a ``plots/`` sub-dir is created.
    mode             : "best" (argmin of wm_costs) or "worst" (argmax).
    tb_writer        : optional TensorBoard SummaryWriter.
    debug            : if True, print the saved file path.
    """
    if not feasible_designs:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if mode == "best":
        idx  = int(np.argmin(wm_costs))
        label = "best"
    else:
        idx  = int(np.argmax(wm_costs))
        label = "worst"

    cost    = wm_costs[idx]
    pred_np = mean_pred[idx].detach().cpu().numpy()  # (n_states, n_freq)
    tgt_np  = target_s.detach().cpu().numpy()         # (n_states, n_freq)
    freq_ghz = freq_points / 1e9
    n_states = pred_np.shape[0]
    colors   = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 4))
    for s in range(n_states):
        col = colors[s % len(colors)]
        ax.plot(freq_ghz, pred_np[s], color=col, lw=1.8, label=f"State {s} (WM pred)")
        ax.plot(freq_ghz, tgt_np[s],  color=col, lw=1.0,
                linestyle="--", alpha=0.5, label=f"State {s} (target)")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 (dB)")
    ax.set_title(
        f"Iter {current_iter} — {label} WM design  "
        f"proxy_cost={cost:.4f}  "
        f"(design #{idx} of {len(feasible_designs)} feasible)"
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-45, 5)

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    out_path = plot_dir / f"iter_{current_iter:05d}_{label}_wm.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")

    if tb_writer is not None:
        try:
            tb_writer.add_figure(f"iter/{label}_wm_design", fig, current_iter)
        except Exception:
            pass

    plt.close(fig)

    if debug:
        print(
            f"  [DBG] iter {current_iter} {label} WM: design #{idx}  "
            f"proxy_cost={cost:.4f}  → plots/{out_path.name}"
        )


# ---------------------------------------------------------------------------
# Convenience wrappers (preserve existing call sites)
# ---------------------------------------------------------------------------

def plot_iter_best(
    feasible_designs: List,
    mean_pred: torch.Tensor,
    wm_costs: List[float],
    current_iter: int,
    freq_points: np.ndarray,
    target_s: torch.Tensor,
    output_dir: Path,
    tb_writer=None,
    debug: bool = False,
) -> None:
    """Plot best WM design. Thin wrapper around plot_iter_wm(mode='best')."""
    plot_iter_wm(
        feasible_designs = feasible_designs,
        mean_pred        = mean_pred,
        wm_costs         = wm_costs,
        current_iter     = current_iter,
        freq_points      = freq_points,
        target_s         = target_s,
        output_dir       = output_dir,
        mode             = "best",
        tb_writer        = tb_writer,
        debug            = debug,
    )


def plot_iter_worst(
    feasible_designs: List,
    mean_pred: torch.Tensor,
    wm_costs: List[float],
    current_iter: int,
    freq_points: np.ndarray,
    target_s: torch.Tensor,
    output_dir: Path,
    tb_writer=None,
    debug: bool = False,
) -> None:
    """Plot worst WM design. Thin wrapper around plot_iter_wm(mode='worst')."""
    plot_iter_wm(
        feasible_designs = feasible_designs,
        mean_pred        = mean_pred,
        wm_costs         = wm_costs,
        current_iter     = current_iter,
        freq_points      = freq_points,
        target_s         = target_s,
        output_dir       = output_dir,
        mode             = "worst",
        tb_writer        = tb_writer,
        debug            = debug,
    )


# ---------------------------------------------------------------------------
# Best simulated design per iteration
# ---------------------------------------------------------------------------

def plot_iter_best_sim(
    s_params: torch.Tensor,
    current_iter: int,
    freq_points: np.ndarray,
    target_s: torch.Tensor,
    output_dir: Path,
    cost: Optional[float] = None,
    tb_writer=None,
    debug: bool = False,
) -> None:
    """
    Plot simulated S-params vs target for the best design in this iteration's
    simulation batch. Save to ``output_dir/plots/iter_{N:05d}_best_sim.png``.

    Parameters
    ----------
    s_params     : (n_states, n_freq) real simulated S21 tensor for the best design.
    current_iter : training iteration index.
    freq_points  : (n_freq,) frequency grid in Hz.
    target_s     : (n_states, n_freq) target S-param conditioning tensor.
    output_dir   : base output directory; a ``plots/`` sub-dir is created.
    cost         : optional real cost value for the design (shown in title).
    tb_writer    : optional TensorBoard SummaryWriter.
    debug        : if True, print the saved file path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    s_np      = s_params.detach().cpu().numpy()  # (n_states, n_freq)
    tgt_np    = target_s.detach().cpu().numpy()  # (n_states, n_freq)
    freq_ghz  = freq_points / 1e9
    n_states  = s_np.shape[0]
    colors    = plt.cm.tab10.colors

    cost_str = f"  cost={cost:.4f}" if cost is not None else ""

    fig, ax = plt.subplots(figsize=(9, 4))
    for s in range(n_states):
        col = colors[s % len(colors)]
        ax.plot(freq_ghz, s_np[s],   color=col, lw=1.8, label=f"State {s} (sim)")
        ax.plot(freq_ghz, tgt_np[s], color=col, lw=1.0,
                linestyle="--", alpha=0.5, label=f"State {s} (target)")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 (dB)")
    ax.set_title(f"Iter {current_iter} — best simulated design{cost_str}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-45, 5)

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    out_path = plot_dir / f"iter_{current_iter:05d}_best_sim.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")

    if tb_writer is not None:
        try:
            tb_writer.add_figure("iter/best_sim_design", fig, current_iter)
        except Exception:
            pass

    plt.close(fig)

    if debug:
        print(
            f"  [DBG] iter {current_iter} best sim: "
            f"cost={cost:.4f}  → plots/{out_path.name}"
        )


# ---------------------------------------------------------------------------
# WM vs simulation comparison for correction step
# ---------------------------------------------------------------------------

def plot_wm_vs_sim(
    wm_preds: List[torch.Tensor],
    sim_preds: List[torch.Tensor],
    entry_ids: List[int],
    current_iter: int,
    freq_points: np.ndarray,
    target_s: torch.Tensor,
    output_dir: Path,
    tb_writer=None,
    debug: bool = False,
) -> None:
    """
    Compare world-model predictions (before correction) against real simulation
    results for entries processed in the correction step.

    Plots up to 4 entries in a grid (up to 2x2), each subplot showing WM pred
    (solid lines) vs sim result (dashed lines) for each filter state.

    Save to ``output_dir/plots/iter_{N:05d}_correction_compare.png``.

    Parameters
    ----------
    wm_preds     : list of (n_states, n_freq) WM prediction tensors.
    sim_preds    : list of (n_states, n_freq) real simulation result tensors.
    entry_ids    : list of buffer entry IDs (same order as wm_preds/sim_preds).
    current_iter : training iteration index.
    freq_points  : (n_freq,) frequency grid in Hz.
    target_s     : (n_states, n_freq) target S-param conditioning tensor.
    output_dir   : base output directory; a ``plots/`` sub-dir is created.
    tb_writer    : optional TensorBoard SummaryWriter.
    debug        : if True, print the saved file path.
    """
    if not wm_preds:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Limit to at most 4 entries
    n_entries = min(len(wm_preds), 4)
    wm_preds  = wm_preds[:n_entries]
    sim_preds = sim_preds[:n_entries]
    entry_ids = entry_ids[:n_entries]

    freq_ghz = freq_points / 1e9
    colors   = plt.cm.tab10.colors

    # Layout: 1x1, 1x2, 2x2 depending on count
    if n_entries == 1:
        n_rows, n_cols = 1, 1
    elif n_entries == 2:
        n_rows, n_cols = 1, 2
    else:
        n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for plot_idx in range(n_entries):
        ax       = axes_flat[plot_idx]
        wm_np    = wm_preds[plot_idx].detach().cpu().numpy()   # (n_states, n_freq)
        sim_np   = sim_preds[plot_idx].detach().cpu().numpy()  # (n_states, n_freq)
        n_states = wm_np.shape[0]
        eid      = entry_ids[plot_idx]

        for s in range(n_states):
            col = colors[s % len(colors)]
            ax.plot(freq_ghz, wm_np[s],  color=col, lw=1.8, linestyle="-",
                    label=f"S{s} WM")
            ax.plot(freq_ghz, sim_np[s], color=col, lw=1.2, linestyle="--",
                    label=f"S{s} sim")

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("S21 (dB)")
        ax.set_title(f"Entry {eid} — WM (solid) vs sim (dashed)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-45, 5)

    # Hide unused subplots
    for plot_idx in range(n_entries, len(axes_flat)):
        axes_flat[plot_idx].set_visible(False)

    fig.suptitle(f"Iter {current_iter} — correction: WM predictions vs simulation", y=1.01)
    fig.tight_layout()

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    out_path = plot_dir / f"iter_{current_iter:05d}_correction_compare.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")

    if tb_writer is not None:
        try:
            tb_writer.add_figure("correction/wm_vs_sim", fig, current_iter)
        except Exception:
            pass

    plt.close(fig)

    if debug:
        print(
            f"  [DBG] iter {current_iter} correction compare: "
            f"{n_entries} entries  → plots/{out_path.name}"
        )
