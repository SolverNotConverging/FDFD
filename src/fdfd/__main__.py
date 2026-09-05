"""Small command-line entry point for installation checks and native GUIs."""
import argparse

from . import __version__
from ._apps import launch
from cem_common._native import bundled_executable


def main():
    parser = argparse.ArgumentParser(description="FDFD solvers and native applications")
    parser.add_argument("--version", action="version", version=f"FDFD {__version__}")
    parser.add_argument("command", choices=("info", "calculator", "calculator-cli", "periodic-viewer", "scattering-viewer"))
    args, remaining = parser.parse_known_args()
    if args.command == "info":
        if remaining:
            parser.error(f"Unrecognized arguments: {' '.join(remaining)}")
        from periodic_eigensolver import native_backend_available
        print(f"FDFD {__version__}")
        print(f"Compiled periodic eigensolver: {native_backend_available()}")
        for name in ("transmission-line-calculator", "fem-periodic-mode-viewer", "fem-waveguide-scattering-viewer"):
            print(f"{name}: {bundled_executable(name) or 'not bundled'}")
        return 0
    names = {"calculator": "transmission-line-calculator", "calculator-cli": "transmission-line-calculator-cli",
             "periodic-viewer": "fem-periodic-mode-viewer", "scattering-viewer": "fem-waveguide-scattering-viewer"}
    return launch(names[args.command], remaining)


if __name__ == "__main__":
    raise SystemExit(main())
