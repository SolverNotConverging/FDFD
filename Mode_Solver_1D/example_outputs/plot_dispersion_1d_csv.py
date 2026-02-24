"""Plot 1D modal dispersion CSV outputs.

Usage:
  python plot_dispersion_1d_csv.py /path/to/1D_modes_dispersion.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def plot_1d_dispersion(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    freq = df["Frequency (GHz)"]
    te_beta_cols = [c for c in df.columns if c.startswith("Beta_TE_")]
    tm_beta_cols = [c for c in df.columns if c.startswith("Beta_TM_")]
    te_alpha_cols = [c for c in df.columns if c.startswith("Alpha_TE_")]
    tm_alpha_cols = [c for c in df.columns if c.startswith("Alpha_TM_")]

    for col in te_beta_cols:
        mode = col.split("_")[-1]
        axes[0].plot(freq, df[col], "--", label=f"TE Mode {mode}")
    for col in tm_beta_cols:
        mode = col.split("_")[-1]
        axes[0].plot(freq, df[col], "-", label=f"TM Mode {mode}")

    axes[0].set_ylabel(r"Normalized Beta $\beta/k_0$")
    axes[0].set_title("Propagation Constants (β) for TE and TM Modes")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].grid(True)
    axes[0].set_ylim(0)

    for col in te_alpha_cols:
        mode = col.split("_")[-1]
        axes[1].plot(freq, df[col], "--", label=f"TE Mode {mode}")
    for col in tm_alpha_cols:
        mode = col.split("_")[-1]
        axes[1].plot(freq, df[col], "-", label=f"TM Mode {mode}")

    axes[1].set_ylabel(r"Normalized Alpha $\alpha/k_0$")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_title("Attenuation Constants (α) for TE and TM Modes")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].grid(True)

    plt.tight_layout()
    out_path = csv_path.with_suffix(".png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Plot saved to {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_dispersion_1d_csv.py /path/to/1D_modes_dispersion.csv")
    csv_path = Path(sys.argv[1]).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    plot_1d_dispersion(csv_path)


if __name__ == "__main__":
    main()
