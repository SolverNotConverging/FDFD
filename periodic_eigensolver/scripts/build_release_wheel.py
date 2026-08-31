"""Build a release wheel and enforce the native-extension contract.

Normal ``pip install`` and editable source installs keep the extension optional
so that the Python backend remains available without a compiler.  Maintainers
use this entry point for publishable wheels: it builds into a private staging
directory, validates the archive, and only then moves it into the requested
output directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

try:
    from .verify_native_wheel import verify_native_wheel
except ImportError:  # Direct ``python scripts/build_release_wheel.py`` use.
    from verify_native_wheel import verify_native_wheel


def build_release_wheel(
    output_directory: Path,
    *,
    no_build_isolation: bool = False,
) -> Path:
    """Build, verify, and publish one wheel into ``output_directory``."""
    package_root = Path(__file__).resolve().parents[1]
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="periodic-eigensolver-release-") as staging:
        staging_directory = Path(staging)
        command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            "--wheel-dir",
            os.fspath(staging_directory),
        ]
        if no_build_isolation:
            command.append("--no-build-isolation")
        command.append(os.fspath(package_root))
        subprocess.run(command, cwd=package_root, check=True)

        wheels = sorted(staging_directory.glob("periodic_eigensolver-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                "release build must produce exactly one periodic-eigensolver "
                f"wheel; found {[wheel.name for wheel in wheels]}"
            )
        wheel = wheels[0]
        native_member = verify_native_wheel(wheel)

        destination = output_directory / wheel.name
        staged_destination = output_directory / f".{wheel.name}.tmp"
        try:
            shutil.copy2(wheel, staged_destination)
            os.replace(staged_destination, destination)
        finally:
            staged_destination.unlink(missing_ok=True)
        print(f"verified native extension: {native_member}")
        print(destination)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a publishable periodic-eigensolver native wheel."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dist",
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help="Use build dependencies from the active environment.",
    )
    arguments = parser.parse_args()
    try:
        build_release_wheel(
            arguments.output_dir,
            no_build_isolation=arguments.no_build_isolation,
        )
    except (
        BadZipFile,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
