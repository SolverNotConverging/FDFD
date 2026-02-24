"""Plot 2D modal dispersion CSV outputs.

Run this file directly in PyCharm. Adjust `csv_path` if needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def plot_2d_dispersion(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    freq = df["frequency_GHz"]

    for col in [c for c in df.columns if c.endswith(" beta")]:
        mode = col.replace(" beta", "")
        ax[0].plot(freq, df[col], label=mode)
    ax[0].set_title("Propagation Constant vs Frequency")
    ax[0].set_xlabel("Frequency (GHz)")
    ax[0].set_ylabel(r"$\hat{\beta}$")
    ax[0].grid(True)
    ax[0].legend(ncol=2, fontsize=8)

    for col in [c for c in df.columns if c.endswith(" alpha")]:
        mode = col.replace(" alpha", "")
        ax[1].plot(freq, df[col], label=mode)
    ax[1].set_title("Attenuation Constant vs Frequency")
    ax[1].set_xlabel("Frequency (GHz)")
    ax[1].set_ylabel(r"$\alpha$")
    ax[1].grid(True)
    ax[1].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    out_path = csv_path.with_suffix(".png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent / "2D_modes_dispersion.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    plot_2d_dispersion(csv_path)
