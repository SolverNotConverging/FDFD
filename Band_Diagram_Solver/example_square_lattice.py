"""Example usage of :class:`BandDiagramSolver2D` for a square lattice."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

MODULE_NAME = "band_diagram_solver_2d"
MODULE_PATH = Path(__file__).with_name("2D_Band_Diagram.py")
_spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load 2D band diagram solver from {MODULE_PATH}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[MODULE_NAME] = _module
_spec.loader.exec_module(_module)
BandDiagramSolver2D = _module.BandDiagramSolver2D


def main() -> None:
    """Build, solve and plot the band diagram of a dielectric rod lattice."""

    solver = BandDiagramSolver2D(a=1.0, Nx=40, background_er=10.2)
    solver.add_circular_inclusion(radius=0.4, er=1.0)

    symmetry_points, labels = solver.default_high_symmetry_path()
    beta_path, tick_positions = solver.generate_bloch_path(symmetry_points, total_points=200)
    solver.set_tick_labels(labels, tick_positions)

    result = solver.compute_band_structure(beta_path, num_bands=5)

    illustration = os.path.join(os.path.dirname(__file__), "2D_Band_Diagram_Illustration.png")
    solver.plot_band_diagram(result, wnmax=0.6, illustration_path=illustration)
    plt.show()


if __name__ == "__main__":
    main()
