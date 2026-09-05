"""Inspect a saved periodic result without reconstructing or running a solver."""
import argparse
from pathlib import Path
from fdfd_periodic_modes import load_result

DEFAULT_INPUT = Path(__file__).resolve().parents[4] / "outputs/examples/fdfd/periodic_modes/image_guide_leaky_wave_antenna_3d/modes.h5"


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
