"""Plot 1D modal dispersion CSV outputs.

Run this file directly in PyCharm. Adjust `csv_path` if needed.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent / "1D_modes_dispersion.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    plot_1d_dispersion(csv_path)
