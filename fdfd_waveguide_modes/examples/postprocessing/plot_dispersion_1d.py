"""Plot the CSV produced by layered_waveguide_1d_dispersion."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import argparse
import csv
from fdfd_waveguide_modes import plot_dispersion

DEFAULT_INPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/layered_waveguide_1d_dispersion/dispersion.csv"

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("path",nargs="?",type=Path,default=DEFAULT_INPUT)
args=parser.parse_args()
with args.path.open(newline="") as stream:
    rows=list(csv.DictReader(stream))
frequencies = sorted({float(row["frequency_hz"]) for row in rows})
modes = sorted({int(row["mode"]) for row in rows})
values = {(float(row["frequency_hz"]), int(row["mode"])):
          complex(float(row["neff_real"]), float(row["neff_imag"])) for row in rows}
neff = [[values[frequency, mode] for mode in modes] for frequency in frequencies]
figure = plot_dispersion(frequencies, neff, show=False)
output=args.path.with_suffix(".png")
figure.savefig(output,dpi=160)
print(output)
