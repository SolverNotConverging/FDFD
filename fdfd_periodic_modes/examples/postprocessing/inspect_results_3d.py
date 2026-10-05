"""Inspect a saved periodic result without reconstructing or running a solver."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fdfd_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path
from fdfd_periodic_modes import load_result

DEFAULT_INPUT = _ROOT / "outputs/fdfd_periodic_modes/examples/image_guide_leaky_wave_antenna_3d/modes.h5"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",nargs="?",type=Path,default=DEFAULT_INPUT)
    args=parser.parse_args()
    result=load_result(args.path)
    print("Effective indices:",result.neff)
    print("Grid:",result.mesh_data.resolution)
    result.show()
    return result


if __name__ == "__main__":
    main()
