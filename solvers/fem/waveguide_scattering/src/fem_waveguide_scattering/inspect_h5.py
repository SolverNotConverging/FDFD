"""Command-line inspection and native-GUI launch for FEM Waveguide Scattering HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .hdf5 import load_h5
from .viewer import launch_viewer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fem-waveguide-scattering-inspect",
        description="Inspect a FEM Waveguide Scattering HDF5 result or open it in the native viewer.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help=(
            "HDF5 result to inspect. With --gui this may also be a directory; "
            "the current directory is used when omitted."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="launch the native viewer instead of loading all arrays in Python",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="with --gui, wait for the native viewer to exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inspector and return a process-style status code."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.wait and not arguments.gui:
        parser.error("--wait requires --gui")

    if arguments.gui:
        target = arguments.path if arguments.path is not None else Path.cwd()
        process = launch_viewer(target)
        print(f"native viewer opened: {Path(target).expanduser().resolve()}")
        return process.wait() if arguments.wait else 0

    path = arguments.path if arguments.path is not None else Path("cem_scattering_result.h5")
    saved = load_h5(path)
    print("kind =", saved.kind)
    print("frequencies (Hz) =", saved.frequencies_hz)
    for index, result in enumerate(saved.results):
        print(f"point {index}: frequency={result.frequency_hz} Hz, ky={result.ky}")
        print("  S =", dict(result.s_parameters))
        print("  E/H samples =", result.E_total.shape, result.H_total.shape)
        print("  modes =", len(result.modes))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the entry point
    raise SystemExit(main())
