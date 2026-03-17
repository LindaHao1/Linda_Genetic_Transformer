"""
S-parameter plotting using scikit-rf.

Reads .s1p Touchstone files (one per control voltage) and produces:
  1. S11 magnitude vs frequency (dB)        — all voltages on one axes
  2. S11 on a Smith chart (with labels)     — all voltages on one axes
  3. |Z(V1)| / |Z(V2)| ratio vs frequency  — when exactly 2 voltages given

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
