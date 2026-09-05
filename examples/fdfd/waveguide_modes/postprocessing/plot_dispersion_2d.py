"""Plot the CSV produced by dielectric_waveguide_2d_dispersion."""
import argparse
import csv
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure

DEFAULT_INPUT = Path(__file__).resolve().parents[4] / "outputs/examples/fdfd/waveguide_modes/dielectric_waveguide_2d_dispersion/dispersion.csv"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",nargs="?",type=Path,default=DEFAULT_INPUT)
    args=parser.parse_args()
    with args.path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    figure=Figure(figsize=(7,5))
    axes=figure.subplots(2,1,sharex=True)
    for mode in sorted({int(row["mode"]) for row in rows}):
        selected=sorted((r for r in rows if int(r["mode"])==mode),key=lambda r:float(r["frequency_hz"]))
        f=np.array([float(r["frequency_hz"]) for r in selected])
        axes[0].plot(f/1e9,[float(r["neff_real"]) for r in selected],"o-",label=f"mode {mode}")
        axes[1].plot(f/1e9,[-float(r["neff_imag"]) for r in selected],"o-")
    axes[0].set(ylabel="Re(neff)")
    axes[0].legend()
    axes[1].set(xlabel="Frequency (GHz)",ylabel="-Im(neff)")
    figure.tight_layout()
    output=args.path.with_suffix(".png")
    figure.savefig(output,dpi=160)
    print(output)
    return figure


if __name__ == "__main__":
    main()
