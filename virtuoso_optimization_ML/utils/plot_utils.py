"""
plot_utils.py — Plotting helpers for the online ML training loop.

Functions
---------
plot_iter_wm          : Save an S-param superposition plot for either the best
                        or worst world-model-predicted design of one iteration.
plot_iter_best        : Convenience wrapper — calls plot_iter_wm(mode="best").
plot_iter_worst       : Convenience wrapper — calls plot_iter_wm(mode="worst").
plot_iter_best_sim    : Save an S-param plot for the best simulated design
                        in this iteration's simulation batch.
plot_wm_vs_sim        : Compare WM predictions vs real simulation results for
                        entries processed during the correction step.
plot_uncertainty_summary : Two-panel plot of predictive variance and RND OOD
                        scores per design, with simulation-trigger markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
from collections import deque


# ---------------------------------------------------------------------------
# Design pixel visualisation helper
# ---------------------------------------------------------------------------

def _render_design_pixels(image_tensor: torch.Tensor, n_states: int) -> np.ndarray:
    """
    Convert a (1, rows, cols+n_states) encoded design image to a uint8 RGB array.

    Pixel-type colour mapping (based on [0,1] float encoding regions):
      ground   (val < 0.125)           → dark gray   [60,  60,  60]
      metal    (0.125 ≤ val < 0.275)   → light gray  [200, 200, 200]
      inductor (0.275 ≤ val < 0.55)    → green       [50,  180, 80]
      switch   (val ≥ 0.55)            → orange      [230, 100, 30]

    The last n_states columns (binary switch-state flags) use switch-index
    order: row i encodes the ON/OFF state of switch i (sorted in row-first
    scan order).  Rows beyond the number of switches are left at 0.
      0 → open  : white [240, 240, 240]
      1 → closed: blue  [50,  100, 200]

    A thin vertical separator (black column) is inserted between the pixel
    grid and the switch-state columns when n_states > 0.

    Returns
    -------
    rgb : np.ndarray of shape (rows, cols + 1 + n_states, 3) uint8
    """
    img = image_tensor.detach().cpu().squeeze(0).numpy()  # (rows, total_cols)
    rows, total_cols = img.shape
    cols = total_cols - n_states

    # Build output with an extra separator column
    out_cols = cols + 1 + n_states
    rgb = np.zeros((rows, out_cols, 3), dtype=np.uint8)

    pix = img[:, :cols]
    rgb[:, :cols][pix <  0.125]                          = [60,  60,  60]
    rgb[:, :cols][(pix >= 0.125) & (pix < 0.275)]        = [200, 200, 200]
    rgb[:, :cols][(pix >= 0.275) & (pix < 0.55)]         = [50,  180, 80]
    rgb[:, :cols][pix >= 0.55]                           = [230, 100, 30]

    # Separator
    rgb[:, cols] = [0, 0, 0]

    # Switch-state columns
    for s in range(n_states):
        state_col = img[:, cols + s]
        col_pixels = rgb[:, cols + 1 + s]
        col_pixels[state_col <  0.5] = [240, 240, 240]
        col_pixels[state_col >= 0.5] = [50,  100, 200]

    return rgb


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

    # Retrieve the design image for this index (feasible_designs[idx][1])
    design_img = None
    try:
        design_img = feasible_designs[idx][1]
    except Exception:
        pass

    if design_img is not None:
        fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(14, 4),
                                          gridspec_kw={"width_ratios": [1, 2]})
        rgb = _render_design_pixels(design_img, n_states)
        ax_img.imshow(rgb, aspect="equal", interpolation="nearest")
        ax_img.set_title("Design pixels", fontsize=9)
        ax_img.axis("off")
    else:
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
    design_image: Optional[torch.Tensor] = None,
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
    design_image : optional (1, rows, cols+n_states) encoded design tensor; if given,
                   a coloured pixel-grid panel is shown to the left of the S-param plot.
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

    if design_image is not None:
        fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(14, 4),
                                          gridspec_kw={"width_ratios": [1, 2]})
        rgb = _render_design_pixels(design_image, n_states)
        ax_img.imshow(rgb, aspect="equal", interpolation="nearest")
        ax_img.set_title("Design pixels", fontsize=9)
        ax_img.axis("off")
    else:
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


# ---------------------------------------------------------------------------
# Uncertainty summary: variance + RND OOD scores per design
# ---------------------------------------------------------------------------

# Rolling history for the RND running mean overlay
_rnd_history: deque = deque(maxlen=200)


def plot_uncertainty_summary(
    variance_per_design:  List[float],
    rnd_score_per_design: List[float],
    sim_mask:             List[bool],
    var_threshold:        float,
    rnd_threshold:        float,
    iteration:            int,
    output_dir:           Path,
    writer=None,
    debug:                bool = False,
) -> None:
    """
    Two-panel figure showing predictive variance and RND OOD score per sampled
    feasible design for one iteration.  Designs that triggered real simulation
    are highlighted with a red ×.

    Top panel    — Predictive Variance (log scale)
      · One scatter point per design (y = mean variance over states×freq).
      · Dashed horizontal line at var_threshold.
      · Red × markers for simulation-triggered designs.

    Bottom panel — RND OOD Score (log scale)
      · Same per-design layout.
      · Dashed horizontal line at rnd_threshold.
      · Thin line showing running mean of RND scores across recent iterations
        (tracks how in-distribution the generator has become over time).

    Saved to ``output_dir/plots/uncertainty_iter{N:05d}.png``.
    TensorBoard scalars are logged when ``writer`` is provided.

    Parameters
    ----------
    variance_per_design  : mean predictive variance per feasible design this iter.
    rnd_score_per_design : RND OOD score per feasible design this iter.
    sim_mask             : True for designs sent to real simulation.
    var_threshold        : variance trigger threshold (shown as dashed line).
    rnd_threshold        : RND trigger threshold (shown as dashed line).
    iteration            : current training iteration.
    output_dir           : base output directory.
    writer               : optional TensorBoard SummaryWriter.
    debug                : if True, print saved file path.
    """
    if not variance_per_design:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(variance_per_design)
    x = list(range(n))

    # Separate sim-triggered vs WM-labeled indices
    sim_idx = [i for i, s in enumerate(sim_mask) if s]
    wm_idx  = [i for i, s in enumerate(sim_mask) if not s]

    var_arr = np.array(variance_per_design, dtype=np.float64)
    rnd_arr = np.array(rnd_score_per_design, dtype=np.float64)

    # Update running RND history with the mean RND score this iteration
    mean_rnd = float(rnd_arr.mean()) if len(rnd_arr) > 0 else 0.0
    _rnd_history.append((iteration, mean_rnd))

    fig, axes = plt.subplots(2, 1, figsize=(max(6, n * 0.4 + 2), 7), sharex=False)

    # --- Top panel: Predictive Variance ---
    ax = axes[0]
    if wm_idx:
        ax.scatter(wm_idx, var_arr[wm_idx], color="steelblue", s=50,
                   zorder=3, label="WM-labeled")
    if sim_idx:
        ax.scatter(sim_idx, var_arr[sim_idx], color="red", marker="x", s=80,
                   linewidths=2, zorder=4, label="Sent to sim")
    ax.axhline(var_threshold, color="orange", linestyle="--", linewidth=1.2,
               label=f"var_threshold={var_threshold:.4f}")
    ax.set_yscale("log" if var_arr.max() > 0 else "linear")
    ax.set_ylabel("Mean Predictive Variance")
    ax.set_title(f"Iter {iteration} — Predictive Variance per Design")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, max(n - 0.5, 0.5))

    # --- Bottom panel: RND OOD Score ---
    ax = axes[1]
    if wm_idx:
        ax.scatter(wm_idx, rnd_arr[wm_idx], color="steelblue", s=50,
                   zorder=3, label="WM-labeled")
    if sim_idx:
        ax.scatter(sim_idx, rnd_arr[sim_idx], color="red", marker="x", s=80,
                   linewidths=2, zorder=4, label="Sent to sim")
    ax.axhline(rnd_threshold, color="orange", linestyle="--", linewidth=1.2,
               label=f"rnd_threshold={rnd_threshold:.4f}")

    # Running mean overlay
    if len(_rnd_history) >= 2:
        hist_iters = [h[0] for h in _rnd_history]
        hist_means = [h[1] for h in _rnd_history]
        # Plot as a text annotation (single point per iter) on a secondary x-axis
        ax2 = ax.twinx()
        ax2.plot(range(len(hist_iters)), hist_means, color="grey", lw=1,
                 alpha=0.6, label="Running mean RND")
        ax2.set_ylabel("Running mean RND", color="grey", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=7)

    ax.set_yscale("log" if rnd_arr.max() > 0 else "linear")
    ax.set_xlabel("Design index")
    ax.set_ylabel("RND OOD Score")
    ax.set_title(f"Iter {iteration} — RND OOD Score per Design")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, max(n - 0.5, 0.5))

    fig.tight_layout()

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    out_path = plot_dir / f"uncertainty_iter{iteration:05d}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")

    if writer is not None:
        try:
            writer.add_figure("uncertainty/per_design", fig, iteration)
        except Exception:
            pass

    plt.close(fig)

    if debug:
        print(f"  [DBG] iter {iteration} uncertainty plot  → plots/{out_path.name}")
