"""Fem Electrostatics. Public user API for version 1.0.0."""
from .solver import ElectrostaticSolver
from .results import ElectrostaticResult
from .exceptions import ElectrostaticSolverError
from cem_common.errors import GeometryError
from cem_common.errors import MeshError
from cem_common.errors import SolverError
from .result_api import load_result
from cem_common import NoResultError
from cem_common import PersistenceError

__version__ = "1.0.0"
__all__ = ['ElectrostaticSolver', 'ElectrostaticResult', 'ElectrostaticSolverError', 'GeometryError', 'MeshError', 'SolverError', 'load_result', 'NoResultError', 'PersistenceError']
