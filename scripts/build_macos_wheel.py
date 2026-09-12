"""Build and validate the complete macOS arm64 CPython 3.12 release wheel."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

from delocate.delocating import delocate_wheel
from delocate.wheeltools import InWheel


ROOT = Path(__file__).resolve().parents[1]
WHEEL_NAME = "fdfd-1.0.0-cp312-cp312-macosx_15_0_arm64.whl"
APPLICATIONS = (
    "transmission-line-calculator",
    "fem-periodic-mode-viewer",
    "fem-waveguide-scattering-viewer",
)
EXECUTABLES = (
    "transmission-line-calculator-cli",
    "fem-periodic-mode-inspect",
    "fem-waveguide-scattering-viewer-inspect",
)
SOURCE_ROOTS = (
    ROOT / "src",
    *(path / "src" for path in sorted((ROOT / "libraries").iterdir()) if (path / "src").is_dir()),
    *(path / "src" for path in sorted((ROOT / "solvers").glob("*/*")) if (path / "src").is_dir()),
)


def extract_preserving_modes(wheel: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            archive.extract(member, destination)
            mode = member.external_attr >> 16
            if mode:
                (destination / member.filename).chmod(stat.S_IMODE(mode))


def macho_files(root: Path) -> list[Path]:
    binaries = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        result = subprocess.run(
            ["file", "-b", str(path)], capture_output=True, text=True, check=True
        )
        if "Mach-O" in result.stdout:
            binaries.append(path)
    return binaries


def flatten_frameworks(bundle: Path) -> None:
    """Convert framework symlinks to a wheel-safe, non-versioned layout."""
    frameworks = sorted((bundle / "Contents/Frameworks").glob("*.framework"))
    for binary in macho_files(bundle):
        dependencies = subprocess.run(
            ["otool", "-L", str(binary)], capture_output=True, text=True, check=True
        ).stdout.splitlines()[1:]
        for dependency in dependencies:
            old = dependency.strip().split(" ", 1)[0]
            if ".framework/Versions/A/" not in old:
                continue
            new = old.replace(".framework/Versions/A/", ".framework/")
            subprocess.run(
                ["install_name_tool", "-change", old, new, str(binary)],
                capture_output=True,
                check=True,
            )
        install_ids = subprocess.run(
            ["otool", "-D", str(binary)], capture_output=True, text=True, check=True
        ).stdout.splitlines()[1:]
        for old in install_ids:
            if ".framework/Versions/A/" in old:
                new = old.replace(".framework/Versions/A/", ".framework/")
                subprocess.run(
                    ["install_name_tool", "-id", new, str(binary)],
                    capture_output=True,
                    check=True,
                )

    for framework in frameworks:
        name = framework.stem
        version = framework / "Versions/A"
        binary = framework / name
        resources = framework / "Resources"
        if binary.is_symlink():
            binary.unlink()
        shutil.copy2(version / name, binary)
        if resources.is_symlink():
            resources.unlink()
        elif resources.exists():
            shutil.rmtree(resources)
        shutil.copytree(version / "Resources", resources)
        shutil.rmtree(framework / "Versions")
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(bundle)], check=True)


def validate_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        expected_sources = {
            path.relative_to(source).as_posix()
            for source in SOURCE_ROOTS
            for path in source.rglob("*.py")
        }
        packaged_sources = {name for name in members if name.endswith(".py")}
        if packaged_sources != expected_sources:
            raise SystemExit(f"Wheel source mismatch: {packaged_sources ^ expected_sources}")
        if not any(
            name.startswith("periodic_eigensolver/_cython_kernels") and name.endswith(".so")
            for name in members
        ):
            raise SystemExit("The compiled periodic eigensolver is missing.")
        for application in APPLICATIONS:
            prefix = f"fdfd/native/{application}.app/Contents"
            required = (
                f"{prefix}/MacOS/{application}",
                f"{prefix}/PlugIns/platforms/libqcocoa.dylib",
                f"{prefix}/PlugIns/platforms/libqoffscreen.dylib",
            )
            for name in required:
                if name not in members:
                    raise SystemExit(f"Missing native application component: {name}")
        for executable in EXECUTABLES:
            name = f"fdfd/native/bin/{executable}"
            if name not in members:
                raise SystemExit(f"Missing native executable: {name}")

    with tempfile.TemporaryDirectory(prefix="fdfd-macos-wheel-check-") as temporary:
        extracted = Path(temporary)
        extract_preserving_modes(wheel, extracted)
        binaries: list[Path] = []
        for path in extracted.rglob("*"):
            if not path.is_file():
                continue
            result = subprocess.run(
                ["file", "-b", str(path)], capture_output=True, text=True, check=True
            )
            if "Mach-O" in result.stdout:
                binaries.append(path)
        if not binaries:
            raise SystemExit("The wheel contains no Mach-O binaries.")
        for binary in binaries:
            architectures = subprocess.run(
                ["lipo", "-archs", str(binary)], capture_output=True, text=True, check=True
            ).stdout.split()
            if architectures != ["arm64"]:
                raise SystemExit(f"Unexpected architectures for {binary}: {architectures}")
            dependencies = subprocess.run(
                ["otool", "-L", str(binary)], capture_output=True, text=True, check=True
            ).stdout.splitlines()[1:]
            install_ids = set(
                subprocess.run(
                    ["otool", "-D", str(binary)], capture_output=True, text=True, check=True
                ).stdout.splitlines()[1:]
            )
            for dependency in dependencies:
                name = dependency.strip().split(" ", 1)[0]
                if name in install_ids:
                    continue
                if name.startswith("/") and not name.startswith(("/usr/lib/", "/System/Library/")):
                    raise SystemExit(f"Unrelocated dependency in {binary}: {name}")
            subprocess.run(["codesign", "--verify", "--strict", str(binary)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dist-macos")
    args = parser.parse_args()
    if (
        sys.platform != "darwin"
        or platform.machine() != "arm64"
        or sys.version_info[:2] != (3, 12)
    ):
        parser.error("The macOS release wheel targets Apple silicon / CPython 3.12.")
    if shutil.which("delocate-wheel") is None:
        parser.error("Install the macOS development dependencies with `uv sync` first.")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.whl")):
        parser.error("Use an output directory with no existing wheels.")

    with tempfile.TemporaryDirectory(prefix="fdfd-macos-wheel-build-") as temporary:
        raw = Path(temporary)
        prepared = raw / "prepared"
        prepared.mkdir()
        environment = dict(os.environ, MACOSX_DEPLOYMENT_TARGET="15.0")
        qtpaths = shutil.which("qtpaths6") or shutil.which("qtpaths")
        if qtpaths is None:
            parser.error("Qt's qtpaths executable is required for a macOS wheel build.")
        qt_library_directory = subprocess.run(
            [qtpaths, "--query", "QT_INSTALL_LIBS"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        environment["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, (qt_library_directory, environment.get("DYLD_LIBRARY_PATH")))
        )
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(raw), str(ROOT)],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        raw_wheels = list(raw.glob("*.whl"))
        if len(raw_wheels) != 1 or raw_wheels[0].name != WHEEL_NAME:
            raise SystemExit(f"Unexpected release artifacts: {raw_wheels}")
        prepared_wheel = prepared / WHEEL_NAME
        with InWheel(str(raw_wheels[0]), str(prepared_wheel)) as wheel_root:
            wheel_root = Path(wheel_root)
            for application in APPLICATIONS:
                bundle = wheel_root / "fdfd/native" / f"{application}.app"
                subprocess.run(
                    [
                        "macdeployqt",
                        str(bundle),
                        "-no-strip",
                        "-always-overwrite",
                        f"-libpath={qt_library_directory}",
                    ],
                    check=True,
                )
                flatten_frameworks(bundle)
        delocate_wheel(
            str(prepared_wheel),
            str(output / WHEEL_NAME),
            lib_filt_func=lambda path: ".app/" not in path,
            require_archs=["arm64"],
            sanitize_rpaths=True,
        )

    wheel = output / WHEEL_NAME
    if not wheel.is_file():
        raise SystemExit(f"delocate did not produce {wheel}")
    validate_wheel(wheel)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    print(f"Complete macOS release wheel: {wheel}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
