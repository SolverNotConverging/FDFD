"""Inspect or open FEM periodic HDF5 results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .persistence import launch_viewer, open_periodic_h5, validate_periodic_h5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="HDF5 file or directory (default: current directory).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="launch the native viewer; directories open its HDF5 selector",
    )
    parser.add_argument("--deep", action="store_true", help="validate all heavy datasets")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    path = arguments.path.expanduser().resolve()
    if arguments.gui:
        if arguments.deep:
            parser.error("--deep cannot be combined with --gui")
        launch_viewer(path)
        return 0
    if path.is_dir():
        files = sorted((*path.glob("*.h5"), *path.glob("*.hdf5")))
        if not files:
            raise SystemExit(f"No .h5 or .hdf5 files found in {path}")
        valid_count = 0
        for candidate in files:
            try:
                report = validate_periodic_h5(candidate, deep=arguments.deep)
            except Exception as error:
                print(f"{candidate.name}: invalid ({error})")
            else:
                valid_count += 1
                print(
                    f"{candidate.name}: schema={report.schema_major}.{report.schema_minor} "
                    f"cases={report.case_count} modes={report.mode_count}"
                )
        return 0 if valid_count else 1
    report = validate_periodic_h5(path, deep=arguments.deep)
    with open_periodic_h5(path) as archive:
        print(f"path={path}")
        print(f"schema={report.schema_major}.{report.schema_minor}")
        print(f"cases={archive.case_count}")
        print(f"modes={archive.mode_count}")
        print("frequencies_hz=" + ",".join(f"{value:.12g}" for value in archive.frequencies))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
