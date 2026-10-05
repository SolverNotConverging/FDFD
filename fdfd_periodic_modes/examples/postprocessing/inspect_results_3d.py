"""Inspect a saved periodic result without reconstructing or running a solver."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import argparse
from fdfd_periodic_modes import load_result

DEFAULT_INPUT = _ROOT / "fdfd_periodic_modes/outputs/examples/image_guide_leaky_wave_antenna_3d/modes.h5"

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("path",nargs="?",type=Path,default=DEFAULT_INPUT)
args=parser.parse_args()
result=load_result(args.path)
print("Effective indices:",result.neff)
print("Grid:",result.mesh_data.resolution)
result.show()
