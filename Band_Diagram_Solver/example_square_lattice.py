
"""Example usage of :class:`BandDiagramSolver2D` with multiple Bloch paths."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


MODULE_NAME = "band_diagram_solver_2d"
MODULE_PATH = Path(__file__).with_name("2D_Band_Diagram.py")
_spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load 2D band diagram solver from {MODULE_PATH}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[MODULE_NAME] = _module
_spec.loader.exec_module(_module)
BandDiagramSolver2D = _module.BandDiagramSolver2D



def build_solver() -> "BandDiagramSolver2D":
    """Construct the dielectric rod lattice used in both examples."""

    solver = BandDiagramSolver2D(a=1.0, Nx=40, background_er=10.2)
    solver.add_circular_inclusion(radius=0.4, er=1.0)
    return solver


def _solve_with_solver(
    solver: "BandDiagramSolver2D",
    symmetry_points: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    total_points: int,
    num_bands: int,
    wnmax: float,
    path_title: str,
):
    """Drive the solver along ``symmetry_points`` and plot the band diagram."""

    beta_path, tick_positions = solver.generate_bloch_path(
        symmetry_points, total_points=total_points
    )
    solver.set_tick_labels(labels, tick_positions)

    result = solver.compute_band_structure(beta_path, num_bands=num_bands)
    fig, (ax_structure, ax_path, ax_bands) = solver.plot_band_diagram(
        result, wnmax=wnmax
    )
    ax_path.set_title(f"Bloch Path ({path_title})")
    return fig


def solve_default_path_example(*, total_points: int, num_bands: int, wnmax: float):
    """Run the canonical Γ–X–M–Γ sweep for a square lattice."""

    solver = build_solver()
    points, labels = solver.default_high_symmetry_path()
    return _solve_with_solver(
        solver,
        points,
        labels,
        total_points=total_points,
        num_bands=num_bands,
        wnmax=wnmax,
        path_title="Γ–X–M–Γ",
    )


def solve_custom_path_example(*, total_points: int, num_bands: int, wnmax: float):
    """Demonstrate a user-defined Bloch path through the first Brillouin zone."""

    solver = build_solver()
    g = 2 * np.pi / solver.a
    gamma = np.array([0.0, 0.0])
    q_point = np.array([0.45 * g, 0.10 * g])
    r_point = np.array([0.60 * g, 0.65 * g])
    s_point = np.array([0.10 * g, 0.90 * g])
    points = [gamma, q_point, r_point, s_point, gamma]
    labels = ["Γ", "Q", "R", "S", "Γ"]

    return _solve_with_solver(
        solver,
        points,
        labels,
        total_points=total_points,
        num_bands=num_bands,
        wnmax=wnmax,
        path_title="Γ–Q–R–S–Γ",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the example script."""

    parser = argparse.ArgumentParser(
        description=(
            "Solve and plot photonic band diagrams for a dielectric rod lattice "
            "using the BandDiagramSolver2D class. Choose between the default "
            "Γ–X–M–Γ path or a custom user-defined sweep."
        )
    )
    parser.add_argument(
        "--path",
        choices=("default", "custom", "both"),
        default="default",
        help="Which Bloch path example to run (default: %(default)s).",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=200,
        help=(
            "Number of β samples along each sweep when generating the Bloch path "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=5,
        help="How many bands to compute for each polarisation (default: %(default)s).",
    )
    parser.add_argument(
        "--wnmax",
        type=float,
        default=0.6,
        help="Upper limit for the normalised frequency axis (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the requested band diagram example(s)."""

    args = parse_args(argv)

    figures = []
    if args.path in {"default", "both"}:
        figures.append(
            solve_default_path_example(
                total_points=args.points,
                num_bands=args.num_bands,
                wnmax=args.wnmax,
            )
        )

    if args.path in {"custom", "both"}:
        figures.append(
            solve_custom_path_example(
                total_points=args.points,
                num_bands=args.num_bands,
                wnmax=args.wnmax,
            )
        )

    if figures:
        plt.show()



if __name__ == "__main__":
    main()

