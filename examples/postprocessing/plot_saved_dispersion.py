"""Plot the dispersion of a saved frequency sweep (one ModeResult file per frequency).

Usage, from the repository root:

    python examples/postprocessing/plot_saved_dispersion.py <sweep folder> [--unconnected]

e.g. ``examples/waveguide/outputs/1d_layered_waveguide_dispersion``. Use
``--unconnected`` for periodic sweeps, whose mode order can change.
"""

import argparse
from pathlib import Path

import fdfd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("folder", type=Path, help="folder of .h5 results written by a dispersion example")
parser.add_argument("--unconnected", action="store_true", help="do not join samples (mode order may change)")
args = parser.parse_args()

results = sorted((fdfd.load(path) for path in args.folder.glob("*.h5")), key=lambda result: result.frequency)
if not results:
    raise SystemExit(f"No .h5 results in {args.folder}")
figure = fdfd.plot_dispersion(
    [r.frequency for r in results], [r.neff for r in results], connect=not args.unconnected, show=False
)
output = args.folder / "dispersion.png"
figure.savefig(output, dpi=160)
print(output)
