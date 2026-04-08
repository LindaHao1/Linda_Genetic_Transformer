"""
S-parameter plotting using scikit-rf.

plot_s11_results  (layout_pixel, simplified_pixel)
    Reads .s1p Touchstone files (one per control voltage/state) and produces:
      1. S11 magnitude vs frequency (dB)        — all states on one axes
      2. S11 on a Smith chart (with labels)     — all states on one axes
      3. |Z(V1)| / |Z(V2)| ratio vs frequency  — when exactly 2 states given

plot_s2p_results  (pixelated_design_simple)
    Reads .s2p Touchstone files (one per switch state) and produces:
      1. S21 magnitude vs frequency (dB)  — filter transmission, all states
      2. S11 magnitude vs frequency (dB)  — input match, all states

Outputs are saved as PNG files in the run's plots/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for subprocess contexts
import matplotlib.pyplot as plt

try:
    import skrf as rf
except ImportError as exc:
    raise ImportError(
        "scikit-rf is required for plotting.  Install it with:\n"
        "    pip install scikit-rf"
    ) from exc



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_s2p_results(
    s2p_files: List[Path],
    state_labels: List[str],
    output_dir: Path,
    pixel_matrix: List[List[int]] | None = None,
    cap_pixel_pairs: List[List[List[int]]] | None = None,
) -> dict:
    """
    Plot S21 and S11 for each switch state and save PNGs to *output_dir*.

    Args:
        s2p_files:       Ordered list of .s2p paths, one per state.
        state_labels:    Corresponding state labels (e.g. ["state0", "state1"]).
        output_dir:      Directory where plots are saved.
        pixel_matrix:    Optional 2-D list of pixel types (0-3).  When provided,
                         a design-matrix visualisation is added as a third panel
                         in both the S21 and S11 figure files.
        cap_pixel_pairs: Optional list of [[row1,col1],[row2,col2]] pairs.  When
                         provided alongside pixel_matrix, each pair of pixels is
                         annotated with a shared index number on the design matrix.

    Returns:
        dict with keys 's21_magnitude', 's11_magnitude', and optionally
        'design_matrix', each pointing to the saved PNG path (or None).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    networks = _load_networks(s2p_files, state_labels)
    if not networks:
        print("  [plotter] No valid .s2p files to plot.")
        return {"s21_magnitude": None, "s11_magnitude": None, "design_matrix": None}

    s21_path = output_dir / "s21_magnitude.png"
    s11_path = output_dir / "s11_magnitude.png"

    _plot_s2p_magnitude(networks, s21_path, row=1, col=0, ylabel="S21 (dB)",
                        title="S21 Magnitude vs Frequency (Filter Transmission)",
                        pixel_matrix=pixel_matrix, cap_pixel_pairs=cap_pixel_pairs)
    _plot_s2p_magnitude(networks, s11_path, row=0, col=0, ylabel="S11 (dB)",
                        title="S11 Magnitude vs Frequency (Input Match)",
                        pixel_matrix=pixel_matrix, cap_pixel_pairs=cap_pixel_pairs)

    print(f"  [plotter] S21 magnitude → {s21_path}")
    print(f"  [plotter] S11 magnitude → {s11_path}")

    result: dict = {"s21_magnitude": str(s21_path), "s11_magnitude": str(s11_path),
                    "design_matrix": None}

    if pixel_matrix is not None:
        dm_path = output_dir / "design_matrix.png"
        _plot_design_matrix(pixel_matrix, dm_path, cap_pixel_pairs=cap_pixel_pairs)
        print(f"  [plotter] Design matrix → {dm_path}")
        result["design_matrix"] = str(dm_path)

    return result


def _plot_s2p_magnitude(
    networks: List[Tuple[Any, str]],
    save_path: Path,
    row: int,
    col: int,
    ylabel: str,
    title: str,
    pixel_matrix: List[List[int]] | None = None,
    cap_pixel_pairs: List[List[List[int]]] | None = None,
) -> None:
    """Plot S[row][col] magnitude in dB vs frequency (GHz) for 2-port networks.

    When *pixel_matrix* is supplied a design-matrix panel is appended to the
    right of the S-parameter axes.
    """
    if pixel_matrix is not None:
        fig, (ax, ax_dm) = plt.subplots(
            1, 2,
            figsize=(14, 6),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        _draw_design_matrix(pixel_matrix, ax_dm, cap_pixel_pairs=cap_pixel_pairs)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    for ntwk, label in networks:
        freq_ghz = ntwk.f / 1e9
        s_db     = ntwk.s_db[:, row, col]
        ax.plot(freq_ghz, s_db, label=label, linewidth=1.8)

    ax.set_xlabel("Frequency (GHz)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_s11_results(
    s1p_files: List[Path],
    voltage_labels: List[float],
    output_dir: Path,
) -> dict:
    """
    Plot S11 for each control voltage and save PNGs to *output_dir*.

    Args:
        s1p_files:      Ordered list of .s1p paths, one per voltage.
        voltage_labels: Corresponding V_ctrl values (for legend labels).
        output_dir:     Directory where plots are saved.

    Returns:
        dict with keys 's11_magnitude' and 's11_smith', each pointing to
        the saved PNG path (or None if that plot could not be generated).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    networks = _load_networks(s1p_files, voltage_labels)
    if not networks:
        print("  [plotter] No valid .s1p files to plot.")
        return {"s11_magnitude": None, "s11_smith": None}

    mag_path   = output_dir / "s11_magnitude.png"
    smith_path = output_dir / "s11_smith.png"
    ratio_path = output_dir / "impedance_ratio.png"

    _plot_magnitude(networks, mag_path)
    _plot_smith(networks, smith_path)

    print(f"  [plotter] S11 magnitude → {mag_path}")
    print(f"  [plotter] S11 Smith     → {smith_path}")

    result = {"s11_magnitude": str(mag_path), "s11_smith": str(smith_path), "impedance_ratio": None}
    if _plot_impedance_ratio(networks, ratio_path):
        print(f"  [plotter] |Z| ratio     → {ratio_path}")
        result["impedance_ratio"] = str(ratio_path)

    return result


# ---------------------------------------------------------------------------
# Design-matrix visualisation helpers
# ---------------------------------------------------------------------------

# Pixel type → (display label, colour)
_PIXEL_STYLE = {
    0: ("GND",  "#4a4a4a"),   # dark grey  — ground layer only
    1: ("MTL",  "#5ba8d4"),   # steel blue — top + ground metal
    2: ("IND",  "#e8a838"),   # amber      — inductor port
    3: ("SW",   "#d94f3d"),   # red        — switch port
}


def _draw_design_matrix(
    matrix: List[List[int]],
    ax,
    cap_pixel_pairs: List[List[List[int]]] | None = None,
) -> None:
    """Render the pixel-type matrix as a colour-coded grid on *ax*.

    When *cap_pixel_pairs* is provided, each pair of pixels is annotated with a
    shared index number drawn in the cell centre.  Both pixels in a pair show the
    same number, making the connection visually obvious.  A thin line is also
    drawn between the two pixels to reinforce the pairing.
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows else 0

    # Build an RGB image (row 0 of the matrix = top row visually)
    img = np.zeros((n_rows, n_cols, 3))
    for r in range(n_rows):
        for c in range(n_cols):
            ptype = int(matrix[r][c])
            hex_color = _PIXEL_STYLE.get(ptype, ("?", "#ffffff"))[1]
            img[r, c] = [int(hex_color.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([str(c) for c in range(n_cols)], fontsize=8)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(r) for r in range(n_rows)], fontsize=8)
    ax.set_title("Design Matrix", fontsize=11)

    # --- Capacitor pair annotations ---
    # Each pair gets a shared integer label drawn on both pixels, plus a
    # connecting line between their centres.
    if cap_pixel_pairs:
        # Cycle through a set of high-contrast colours for multiple pairs.
        pair_colors = ["#ffffff", "#ffe066", "#ff9ff3", "#54a0ff", "#00d2d3", "#ff6b6b"]
        for pair_idx, pair in enumerate(cap_pixel_pairs):
            (r1, c1), (r2, c2) = pair
            color = pair_colors[pair_idx % len(pair_colors)]
            label = str(pair_idx)

            # Number on each pixel
            for r, c in [(r1, c1), (r2, c2)]:
                ax.text(
                    c, r, label,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=color,
                    bbox=dict(boxstyle="circle,pad=0.15", facecolor="black",
                              edgecolor=color, linewidth=1.2, alpha=0.75),
                )

            # Connecting line between the two pixel centres
            ax.plot(
                [c1, c2], [r1, r2],
                color=color, linewidth=1.5, linestyle="--", alpha=0.8,
                zorder=3,
            )

    # Legend patches
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=style[1], label=f"{ptype}: {style[0]}")
        for ptype, style in _PIXEL_STYLE.items()
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right",
              bbox_to_anchor=(1.0, -0.08), ncol=2)


def _plot_design_matrix(
    matrix: List[List[int]],
    save_path: Path,
    cap_pixel_pairs: List[List[List[int]]] | None = None,
) -> None:
    """Save a standalone design-matrix visualisation to *save_path*."""
    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_design_matrix(matrix, ax, cap_pixel_pairs=cap_pixel_pairs)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_networks(
    s1p_files: List[Path],
    voltage_labels: List[float],
) -> List[Tuple[Any, str]]:
    """Load each .s1p and pair it with a human-readable label."""

    networks = []
    for path, v in zip(s1p_files, voltage_labels):
        path = Path(path)
        if not path.exists():
            print(f"  [plotter] WARNING: {path} not found — skipping.")
            continue
        try:
            ntwk = rf.Network(str(path))
            label = f"V_ctrl = {v:g} V" if isinstance(v, (int, float)) else str(v)
            networks.append((ntwk, label))
        except Exception as exc:
            print(f"  [plotter] WARNING: Could not read {path}: {exc}")
    return networks


def _plot_magnitude(
    networks: List[Tuple[rf.Network, str]],
    save_path: Path,
) -> None:
    """S11 magnitude in dB vs frequency (GHz)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ntwk, label in networks:
        freq_ghz = ntwk.f / 1e9
        s11_db = ntwk.s_db[:, 0, 0]
        ax.plot(freq_ghz, s11_db, label=label, linewidth=1.8)

    ax.set_xlabel("Frequency (GHz)", fontsize=12)
    ax.set_ylabel("S11 (dB)", fontsize=12)
    ax.set_title("S11 Magnitude vs Frequency", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _plot_smith(
    networks: List[Tuple[rf.Network, str]],
    save_path: Path,
) -> None:
    """S11 impedance Smith chart with evenly spaced frequency markers."""
    fig, ax = plt.subplots(figsize=(8, 8))
    num_markers = 4

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (ntwk, label) in enumerate(networks):
        color = colors[i % len(colors)]
        gamma = ntwk.s[:, 0, 0]
        # draw_labels=True annotates the Smith chart grid with R/X values.
        # We draw them only on the first pass to avoid clutter.
        ntwk.plot_s_smith(
            m=0, n=0,
            ax=ax,
            label=label,
            color=color,
            linewidth=1.8,
            draw_labels=(i == 0),
        )

        # Mark a few evenly spaced frequencies along each trace.
        if len(gamma) > 0:
            marker_count = min(num_markers, len(gamma))
            marker_idx = np.unique(np.linspace(0, len(gamma) - 1, marker_count, dtype=int))
            ax.plot(
                np.real(gamma[marker_idx]),
                np.imag(gamma[marker_idx]),
                linestyle="None",
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
            )

    ax.set_title("S11 Smith Chart (Impedance)", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _plot_impedance_ratio(
    networks: List[Tuple[rf.Network, str]],
    save_path: Path,
    z0: float = 50.0,
) -> bool:
    """
    Plot |Z(V1, f)| / |Z(V2, f)| vs frequency for the first two networks.

    Impedance is derived from S11 via  Z = Z0 * (1 + S11) / (1 - S11).

    Returns True if the plot was saved, False if skipped (< 2 networks).
    """
    if len(networks) < 2:
        print("  [plotter] Impedance ratio requires 2 networks — skipping.")
        return False

    (ntwk1, label1) = networks[0]
    (ntwk2, label2) = networks[1]

    s11_1 = ntwk1.s[:, 0, 0]
    s11_2 = ntwk2.s[:, 0, 0]

    # Guard against S11 = 1 (open circuit singularity)
    denom1 = 1.0 - s11_1
    denom2 = 1.0 - s11_2
    z1_mag = np.where(np.abs(denom1) > 1e-12, np.abs(z0 * (1 + s11_1) / denom1), np.nan)
    z2_mag = np.where(np.abs(denom2) > 1e-12, np.abs(z0 * (1 + s11_2) / denom2), np.nan)

    ratio     = np.where(z1_mag > 1e-12, z2_mag / z1_mag, np.nan)
    freq_ghz  = ntwk1.f / 1e9

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_ghz, ratio, linewidth=1.8, color="tab:purple",
            label=f"|Z({label2})| / |Z({label1})|")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5, label="ratio = 1")

    ax.set_xlabel("Frequency (GHz)", fontsize=12)
    ax.set_ylabel("Impedance Magnitude Ratio", fontsize=12)
    ax.set_title(f"|Z| Ratio: ({label2}) / ({label1})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(left=freq_ghz[0] if len(freq_ghz) else 0)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return True
