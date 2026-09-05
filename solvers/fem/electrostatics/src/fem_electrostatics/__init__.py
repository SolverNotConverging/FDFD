"""Fem Electrostatics. Public user API for version 1.0.0."""
from .solver import ElectrostaticSolver
from .results import ElectrostaticResult
from .geometry import Interval
from .geometry import Rectangle
from .geometry import Circle
from .geometry import Polygon
from .exceptions import ElectrostaticSolverError
from .exceptions import GeometryError
from .exceptions import MeshError
from .exceptions import SolverError
from .result_api import load_result
from fem_common import NoResultError
from fem_common import PersistenceError

__version__ = "1.0.0"
__all__ = ['ElectrostaticSolver', 'ElectrostaticResult', 'Interval', 'Rectangle', 'Circle', 'Polygon', 'ElectrostaticSolverError', 'GeometryError', 'MeshError', 'SolverError', 'load_result', 'NoResultError', 'PersistenceError']
