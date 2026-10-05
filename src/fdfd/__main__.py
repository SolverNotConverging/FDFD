"""Report the FDFD installation and periodic eigensolver backend."""
import argparse
from . import __version__

def main():
    parser = argparse.ArgumentParser(description="FDFD electromagnetic solvers")
    parser.add_argument("--version", action="version", version=f"FDFD {__version__}")
    parser.add_argument("command", choices=("info",))
    parser.parse_args()
    from periodic_eigensolver import native_backend_available
    print(f"FDFD {__version__}")
    print(f"Compiled periodic eigensolver: {native_backend_available()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
