"""
compare_snp.py — Compare two Touchstone SNP files and report statistics.

Usage:
    python compare_snp.py <file1.snp> <file2.snp>
    python compare_snp.py <file1.snp> <file2.snp> --plot
"""

import sys
import argparse
import numpy as np
import skrf

def load_network(path: str) -> skrf.Network:
    return skrf.Network(path)

def compare_snp(path1: str, path2: str, plot: bool = False):
    net1 = load_network(path1)
    net2 = load_network(path2)

    print(f"\n{'='*60}")
    print(f"  File A: {path1}")
    print(f"  File B: {path2}")
    print(f"{'='*60}")

    # --- Basic info ---
    print(f"\n[Info]")
    print(f"  Ports  : A={net1.nports}   B={net2.nports}")
    print(f"  Freqs  : A={len(net1.f)}   B={len(net2.f)}")
    print(f"  F range: A={net1.f[0]/1e9:.1f}–{net1.f[-1]/1e9:.1f} GHz   "
          f"B={net2.f[0]/1e9:.1f}–{net2.f[-1]/1e9:.1f} GHz")

    if net1.nports != net2.nports:
        print("\n  WARNING: port counts differ — interpolating onto common frequency axis only.")

    # --- Interpolate onto common frequency axis ---
    f_common = np.intersect1d(
        np.round(net1.f, 3),
        np.round(net2.f, 3),
    )
    if len(f_common) == 0:
        print("\n  ERROR: no overlapping frequency points.")
        return

    net1_i = net1.interpolate(f_common)
    net2_i = net2.interpolate(f_common)

    n_ports = min(net1.nports, net2.nports)
    S1 = net1_i.s[:, :n_ports, :n_ports]   # (nfreq, n, n)
    S2 = net2_i.s[:, :n_ports, :n_ports]

    diff = S1 - S2
    mag1 = np.abs(S1)
    mag2 = np.abs(S2)
    mag_diff = np.abs(diff)
    phase_diff_deg = np.angle(diff, deg=True)

    print(f"\n  Compared {len(f_common)} overlapping frequency points, "
          f"{n_ports}x{n_ports} S-matrix")

    # --- Aggregate statistics ---
    print(f"\n[Aggregate |S| difference over all ports and frequencies]")
    print(f"  Mean  : {mag_diff.mean():.4f}")
    print(f"  Median: {np.median(mag_diff):.4f}")
    print(f"  Max   : {mag_diff.max():.4f}")
    print(f"  RMS   : {np.sqrt(np.mean(mag_diff**2)):.4f}")

    # dB difference for entries that are non-trivially small
    db1 = 20 * np.log10(np.clip(mag1, 1e-12, None))
    db2 = 20 * np.log10(np.clip(mag2, 1e-12, None))
    db_diff = db1 - db2
    print(f"\n[Aggregate dB difference  (20·log10|S_A| − 20·log10|S_B|)]")
    print(f"  Mean  : {db_diff.mean():.3f} dB")
    print(f"  Std   : {db_diff.std():.3f} dB")
    print(f"  Max   : {db_diff.max():.3f} dB")
    print(f"  Min   : {db_diff.min():.3f} dB")

    # --- Per-entry statistics ---
    print(f"\n[Per S-parameter summary]")
    print(f"  {'Entry':<10} {'MeanMagA':>10} {'MeanMagB':>10} {'MaxDiff':>10} {'RMSDiff':>10} {'MaxdBdiff':>10}")
    print(f"  {'-'*62}")

    per_entry = []
    for i in range(n_ports):
        for j in range(n_ports):
            m1 = mag1[:, i, j]
            m2 = mag2[:, i, j]
            d  = mag_diff[:, i, j]
            rms = np.sqrt(np.mean(d**2))
            db_d = np.abs(db1[:, i, j] - db2[:, i, j])
            per_entry.append((i, j, m1.mean(), m2.mean(), d.max(), rms, db_d.max()))
            label = f"S{i+1}{j+1}"
            print(f"  {label:<10} {m1.mean():>10.4f} {m2.mean():>10.4f} "
                  f"{d.max():>10.4f} {rms:>10.4f} {db_d.max():>10.2f} dB")

    # --- Worst offenders ---
    per_entry.sort(key=lambda x: x[5], reverse=True)  # sort by RMS diff
    print(f"\n[Top 5 entries by RMS |S| difference]")
    for i, j, m1, m2, mx, rms, dbmx in per_entry[:5]:
        print(f"  S{i+1}{j+1}  RMS={rms:.4f}  MaxDiff={mx:.4f}  MaxdBdiff={dbmx:.2f} dB")

    # --- Frequency of max deviation ---
    flat_diff = mag_diff.reshape(len(f_common), -1)
    worst_freq_idx = np.argmax(flat_diff.max(axis=1))
    print(f"\n[Frequency with largest deviation]")
    print(f"  {f_common[worst_freq_idx]/1e9:.1f} GHz  (max |ΔS|={flat_diff[worst_freq_idx].max():.4f})")

    # --- Optional plot ---
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(n_ports, n_ports, figsize=(4 * n_ports, 3 * n_ports), squeeze=False)
            freqs_ghz = f_common / 1e9
            for i in range(n_ports):
                for j in range(n_ports):
                    ax = axes[i][j]
                    ax.plot(freqs_ghz, 20 * np.log10(np.clip(mag1[:, i, j], 1e-12, None)),
                            label="A", linewidth=1.2)
                    ax.plot(freqs_ghz, 20 * np.log10(np.clip(mag2[:, i, j], 1e-12, None)),
                            label="B", linewidth=1.2, linestyle="--")
                    ax.set_title(f"S{i+1}{j+1}")
                    ax.set_xlabel("Freq (GHz)")
                    ax.set_ylabel("|S| (dB)")
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.4)
            fig.tight_layout()
            out = "compare_snp_plot.png"
            plt.savefig(out, dpi=120)
            print(f"\n[Plot saved to {out}]")
        except Exception as e:
            print(f"\n  Plot failed: {e}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two SNP/Touchstone files.")
    parser.add_argument("file1", help="First SNP file (reference)")
    parser.add_argument("file2", help="Second SNP file (comparison)")
    parser.add_argument("--plot", action="store_true", help="Save a magnitude plot")
    args = parser.parse_args()

    compare_snp(args.file1, args.file2, plot=args.plot)
