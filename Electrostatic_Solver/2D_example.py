"""Anisotropic dielectric and internal electrode on a refined 2D FEM mesh."""

from pathlib import Path
import sys


# Support both ``python -m Electrostatic_Solver.2D_example`` and direct IDE/
# file execution.  Without the repository root first on sys.path, the legacy
# Electrostatic_Solver.py file shadows the package directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Electrostatic_Solver import Circle, ElectrostaticSolver, Rectangle


solver = ElectrostaticSolver(
    dim=2,
    domain=((0.0, 20e-3), (0.0, 10e-3)),
    outer_potential=0.0,
)
solver.add_object(
    Rectangle((6e-3, 14e-3), (2e-3, 8e-3)),
    erxx=9.0,
    eryy=6.0,
    erxy=0.4,
    name="high_dk_substrate",
)
solver.set_potential(Circle((10e-3, 5e-3), 1.0e-3), 100.0, name="electrode")

mesh = solver.discretize(
    max_element_size=1.0e-3,
    interface_refinement=0.65,
    boundary_refinement=0.4,
)
result = solver.solve()
print(f"nodes={mesh.info.nodes}, triangles={mesh.info.elements}")
print(f"electrode charge={result.conductor_charge('electrode'):.6e} C/m")
solver.visualize()
