"""Build a portable Python wheel; opt in explicitly to a local Cython wheel."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_python_wheel(path: Path) -> None:
    """Require a universal wheel with the NumPy fallback and no Cython kernel."""
    if not path.name.endswith("-py3-none-any.whl"):
        raise RuntimeError("The Python release must be tagged py3-none-any.")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        forbidden = [name for name in names if "_cython_kernels" in name
                     or Path(name).suffix.lower() in (".pyd", ".so", ".dll", ".dylib", ".exe", ".pyx")]
        if forbidden:
            raise RuntimeError(f"Python release contains native/Cython files: {forbidden}")
        for name in ("fdfd", "fdfd_common", "fdfd_band_structure", "fdfd_mode_tracking",
                     "fdfd_periodic_modes", "fdfd_scattering", "fdfd_waveguide_modes", "periodic_eigensolver"):
            if f"{name}/__init__.py" not in names:
                raise RuntimeError(f"Missing solver package: {name}")
        if "periodic_eigensolver/_numpy_kernels.py" not in names:
            raise RuntimeError("Missing NumPy periodic eigensolver fallback.")
        metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1:
            raise RuntimeError("Expected one wheel metadata record.")
        wheel = archive.read(metadata[0]).decode("utf-8")
        if "Root-Is-Purelib: true" not in wheel or "Tag: py3-none-any" not in wheel:
            raise RuntimeError("Wheel metadata is not portable pure Python.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--cython", action="store_true", help="Build a platform-specific kernel wheel for local use.")
    args = parser.parse_args()
    output = args.output.resolve()
    if list(output.glob("*.whl")):
        parser.error("Use an output directory with no existing wheels.")
    env = dict(os.environ, FDFD_BUILD_CYTHON="1" if args.cython else "0")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output), str(ROOT)], check=True, env=env)
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one wheel, found {wheels}")
    if args.cython:
        from periodic_eigensolver.scripts.verify_native_wheel import verify_native_wheel
        verify_native_wheel(wheels[0])
    else:
        verify_python_wheel(wheels[0])
    print(f"Verified wheel: {wheels[0]}")


if __name__ == "__main__":
    main()
