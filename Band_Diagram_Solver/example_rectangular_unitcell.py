"""Example showing how to work with a rectangular FDFD unit cell."""

import numpy as np

from Band_Diagram import BandDiagramSolver2D


solver = BandDiagramSolver2D(a=1.0, b=1.4, Nx=48, Ny=64, background_er=11.9)

# Add a dielectric bar that breaks the four-fold symmetry of the square case.
def dielectric_bar(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return (np.abs(X) < 0.15) & (Y > -0.2) & (Y < 0.45)


solver.add_object(dielectric_bar, er=4.2)
solver.add_circular_inclusion(radius=0.28, center=(-0.15, 0.0), er=1.0)

points, labels = solver.default_rectangular_lattice_path()
beta_path, tick_positions = solver.generate_bloch_path(points, total_points=240)
solver.set_tick_labels(labels, tick_positions)

result = solver.compute_band_structure(beta_path, num_bands=6)
solver.plot_band_diagram(result, wnmax=0.75)
