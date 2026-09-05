"""Install the maintained packages using the active Python environment."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "libraries/fem_common", "libraries/fem_adaptivity", "libraries/periodic_eigensolver",
    "solvers/fdfd/waveguide_modes", "solvers/fdfd/periodic_modes",
    "solvers/fdfd/band_structure", "solvers/fdfd/scattering",
    "solvers/fem/waveguide_modes", "solvers/fem/periodic_modes",
    "solvers/fem/waveguide_scattering", "solvers/fem/electrostatics",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editable", action="store_true", help="Install source checkouts for development.")
    args = parser.parse_args()
    command = [sys.executable, "-m", "pip", "install"]
    for package in PACKAGES:
        if args.editable:
            command.append("--editable")
        command.append(str(ROOT / package))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
