"""Install the complete FDFD source distribution using the active interpreter."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "libraries/cem_common", "libraries/fem_adaptivity", "libraries/periodic_eigensolver",
    "solvers/fdfd/waveguide_modes", "solvers/fdfd/periodic_modes",
    "solvers/fdfd/band_structure", "solvers/fdfd/scattering",
    "solvers/fem/waveguide_modes", "solvers/fem/periodic_modes",
    "solvers/fem/waveguide_scattering", "solvers/fem/electrostatics",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editable", action="store_true", help="Install source checkouts for development.")
    parser.add_argument(
        "--no-build-isolation", action="store_true",
        help="Use build dependencies already installed in the active environment.",
    )
    args = parser.parse_args()
    command = [sys.executable, "-m", "pip", "install"]
    if args.no_build_isolation:
        command.append("--no-build-isolation")
    if args.editable:
        command.append("--editable")
    command.append(str(ROOT))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
