"""Fem Waveguide Modes. Public user API for version 1.0.0."""
from .solver_1d import ModeSolver1D
from .solver_2d import ModeSolver2D
from .results import Mode
from .results import ModeSet
from .results import SampledFields
from cem_common.errors import BackendCapabilityError
from cem_common.errors import ConfigurationError
from .exceptions import FEMModeSolverError
from cem_common.errors import GeometryError
from cem_common.errors import MeshError
from cem_common.errors import SolverError
from .result_api import load_result
from cem_common import NoResultError
from cem_common import PersistenceError

__version__ = "1.0.0"
__all__ = ['ModeSolver1D', 'ModeSolver2D', 'Mode', 'ModeSet', 'SampledFields', 'BackendCapabilityError', 'ConfigurationError', 'FEMModeSolverError', 'GeometryError', 'MeshError', 'SolverError', 'load_result', 'NoResultError', 'PersistenceError']
