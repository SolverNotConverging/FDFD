"""Launch the native applications installed inside the FDFD distribution."""
from pathlib import Path
import subprocess
import sys

from cem_common._native import bundled_executable, bundled_environment


def launch(name, arguments=None):
    executable = bundled_executable(name)
    if executable is None:
        raise SystemExit("Native applications are absent from this source installation. "
                         "Install the complete Windows release wheel or build the native apps "
                         "as described in README.md.")
    try:
        return subprocess.call([str(executable), *(sys.argv[1:] if arguments is None else arguments)],
                               env=bundled_environment(executable))
    except OSError as exc:
        raise SystemExit(f"Cannot launch {executable}: {exc}") from exc


def calculator():
    return launch("transmission-line-calculator")


def calculator_cli():
    return launch("transmission-line-calculator-cli")


def periodic_viewer():
    return launch("fem-periodic-mode-viewer")


def periodic_inspector():
    return launch("fem-periodic-mode-inspect")


def scattering_viewer():
    return launch("fem-waveguide-scattering-viewer")


def scattering_inspector():
    return launch("fem-waveguide-scattering-viewer-inspect")
