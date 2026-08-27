"""Inspect a saved WaveFEM result without rerunning the FEM solver."""

from __future__ import annotations

import argparse
from pathlib import Path

import wavefem as wf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=Path("wavefem_result.h5"))
    arguments = parser.parse_args()

    saved = wf.load_h5(arguments.path)
    print("kind =", saved.kind)
    print("frequencies (Hz) =", saved.frequencies_hz)
    for index, result in enumerate(saved.results):
        print(f"point {index}: frequency={result.frequency_hz} Hz, ky={result.ky}")
        print("  S =", dict(result.s_parameters))
        print("  E/H samples =", result.E_total.shape, result.H_total.shape)
        print("  modes =", len(result.modes))


if __name__ == "__main__":
    main()
