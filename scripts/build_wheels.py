"""Build and verify a wheel containing the compiled periodic eigensolver."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libraries.periodic_eigensolver.scripts.verify_native_wheel import verify_native_wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dist")
    args = parser.parse_args()
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(args.output), str(ROOT)], check=True)
    for wheel in args.output.glob("*.whl"):
        verify_native_wheel(wheel)


if __name__ == "__main__":
    main()
