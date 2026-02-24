"""Plot periodic 2D dispersion CSV outputs.

Run this file directly in PyCharm. Adjust `csv_path` if needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def plot_periodic_2d(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    freq_ghz = df["Frequency (Hz)"] / 1e9

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for col in [c for c in df.columns if c.startswith("Alpha_Mode_")]:
        mode = col.split("_")[-1]
        axs[0].scatter(freq_ghz, df[col], label=f"Mode {mode}", s=15)
    axs[0].set_ylabel(r"$\alpha / k_0$")
    axs[0].grid(True)
    axs[0].legend()

    for col in [c for c in df.columns if c.startswith("Beta_Mode_")]:
        mode = col.split("_")[-1]
        axs[1].scatter(freq_ghz, df[col], label=f"Mode {mode}", s=15)
    axs[1].set_xlabel("Frequency (GHz)")
    axs[1].set_ylabel(r"$\beta / k_0$")
    axs[1].grid(True)

    plt.tight_layout()
    out_path = csv_path.with_suffix(".png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent / "mode_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    plot_periodic_2d(csv_path)
