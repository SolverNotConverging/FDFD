"""Build and verify a wheel containing the compiled periodic eigensolver."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "cem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from periodic_eigensolver.scripts.verify_native_wheel import verify_native_wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dist")
    args = parser.parse_args()
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(args.output), str(ROOT)], check=True)
    for wheel in args.output.glob("*.whl"):
        verify_native_wheel(wheel)


if __name__ == "__main__":
    main()
