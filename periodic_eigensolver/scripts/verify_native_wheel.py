"""Fail a release job when a binary wheel omitted the Cython extension."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def native_extension_members(wheel: Path) -> list[str]:
    """Return native Arnoldi extension members from one wheel archive."""
    with ZipFile(wheel) as archive:
        members = archive.namelist()
    return [
        name
        for name in members
        if name.startswith("periodic_eigensolver/_cython_kernels.")
        and name.lower().endswith((".pyd", ".so"))
    ]


def verify_native_wheel(wheel: Path) -> str:
    """Return the extension member or raise when the release contract fails."""
    wheel = Path(wheel)
    native = native_extension_members(wheel)
    if len(native) != 1:
        raise RuntimeError(
            "release wheel must contain exactly one native Arnoldi extension; "
            f"found {native}"
        )
    return native[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    argument = parser.parse_args()
    try:
        native = verify_native_wheel(argument.wheel)
    except (BadZipFile, OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(native)


if __name__ == "__main__":
    main()
