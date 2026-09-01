"""Geometry-first 1D/2D finite-element electrostatics."""

from .exceptions import ElectrostaticSolverError, GeometryError, MeshError, NotDiscretizedError, SolverError
from .geometry import ChargeRegion, Circle, Interval, MaterialRegion, Permittivity, Polygon, PotentialRegion, Rectangle
from .meshing import FEMMesh, MeshInfo
from .results import ElectrostaticResult
from .solver import EPSILON_0, ElectrostaticSolver

__version__ = "1.0.0"

__all__ = [
    "ChargeRegion", "Circle", "EPSILON_0", "ElectrostaticResult", "ElectrostaticSolver",
    "ElectrostaticSolverError", "FEMMesh", "GeometryError", "Interval", "MaterialRegion",
    "MeshError", "MeshInfo", "NotDiscretizedError", "Permittivity", "Polygon",
    "PotentialRegion", "Rectangle", "SolverError",
]
