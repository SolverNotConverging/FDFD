"""Build the single complete Windows CPython 3.12 release wheel."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

from install_python import PACKAGES, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dist")
    parser.add_argument("--native-bundle", type=Path, default=ROOT / "outputs/native-release-1.0.0/FDFD-1.0.0-windows-x64")
    parser.add_argument("--no-build-isolation", action="store_true")
    args = parser.parse_args()
    if sys.platform != "win32" or sys.version_info[:2] != (3, 12):
        parser.error("The complete 1.0.0 release wheel targets Windows x64 / CPython 3.12.")
    if sys.maxsize <= 2**32:
        parser.error("A 64-bit interpreter is required.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.whl")):
        parser.error("Use an output directory with no existing wheels.")
    bundle = args.native_bundle.resolve()
    if not (bundle / "SOURCE_INDEX.md").is_file():
        parser.error("Run package_native_windows.py --phase stage, then --phase finish first.")
    environment = dict(os.environ, FDFD_NATIVE_BUNDLE=str(bundle))
    with TemporaryDirectory(prefix="fdfd-release-build-") as temporary:
        staging = Path(temporary)
        for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE"):
            shutil.copy2(ROOT / name, staging / name)
        for source in [ROOT / "src", *(ROOT / package / "src" for package in PACKAGES)]:
            shutil.copytree(source, staging / source.relative_to(ROOT), ignore=shutil.ignore_patterns(
                "__pycache__", "*.egg-info", "*.pyc", "*.pyd", "*.so", "_cython_kernels.c"))
        command = [sys.executable, "-m", "pip", "wheel", "--no-cache-dir", "--no-deps", "--wheel-dir", str(output)]
        if args.no_build_isolation:
            command.append("--no-build-isolation")
        subprocess.run([*command, str(staging)], env=environment, check=True)
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1 or wheels[0].name != "fdfd-1.0.0-cp312-cp312-win_amd64.whl":
        raise SystemExit(f"Unexpected release artifacts: {wheels}")
    with zipfile.ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())
        expected = {"fdfd/" + str(path.relative_to(ROOT / "src/fdfd")).replace("\\", "/")
                    for path in (ROOT / "src/fdfd").rglob("*.py")}
        for package in PACKAGES:
            source = ROOT / package / "src"
            expected.update(path.relative_to(source).as_posix() for path in source.rglob("*.py"))
        packaged = {name for name in members if name.endswith(".py")}
        if packaged != expected:
            raise SystemExit(f"Wheel source mismatch: {packaged ^ expected}")
        if not any(name.startswith("periodic_eigensolver/_cython_kernels") and name.endswith(".pyd") for name in members):
            raise SystemExit("The compiled periodic eigensolver is missing.")
        for name in ("transmission-line-calculator", "transmission-line-calculator-cli", "fem-periodic-mode-viewer",
                     "fem-periodic-mode-inspect", "fem-waveguide-scattering-viewer", "fem-waveguide-scattering-viewer-inspect"):
            if f"fdfd/native/bin/{name}.exe" not in members:
                raise SystemExit(f"Missing native application: {name}")
        for path in bundle.rglob("*"):
            if path.is_file():
                name = "fdfd/native/" + path.relative_to(bundle).as_posix()
                if name not in members or archive.read(name) != path.read_bytes():
                    raise SystemExit(f"Native runtime differs from the qualified bundle: {name}")
    print(f"Complete release wheel: {wheels[0]}")


if __name__ == "__main__":
    main()
