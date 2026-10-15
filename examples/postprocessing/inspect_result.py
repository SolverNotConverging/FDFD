"""Open any saved fdfd result in its interactive viewer, without re-solving.

python examples/postprocessing/inspect_result.py path/to/result.h5
"""

import argparse
from pathlib import Path

import numpy as np

import fdfd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("path", type=Path, help="an .h5 file written by Result.save()")
args = parser.parse_args()

result = fdfd.load(args.path)
print(f"{type(result).__name__} from the {result.family!r} solver")
print(f"Grid: {result.grid.resolution} cells over {result.grid.bounds} m")
print(f"Solve: {result.info.method} on {result.info.device}")
print(f"Largest backward error: {np.max(result.info.residuals):.2e}")
if isinstance(result, fdfd.ModeResult):
    print("Effective indices:", result.neff)
result.show()
