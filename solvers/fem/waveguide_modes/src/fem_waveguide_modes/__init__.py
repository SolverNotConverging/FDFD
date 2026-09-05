"""Fem Waveguide Modes. Public user API for version 1.0.0."""
from .solver_1d import ModeSolver1D
from .solver_2d import ModeSolver2D
from .results import Mode
from .results import ModeSet
from .results import SampledFields
from .materials import Material
from .geometry import Interval
from .geometry import Rectangle
from .geometry import Circle
from .geometry import Polygon
from .boundaries import good_conductor_surface_impedance
from .exceptions import BackendCapabilityError
from .exceptions import ConfigurationError
from .exceptions import FEMModeSolverError
from .exceptions import GeometryError
from .exceptions import MeshError
from .exceptions import SolverError
from .result_api import load_result
from fem_common import NoResultError
from fem_common import PersistenceError

__version__ = "1.0.0"
__all__ = ['ModeSolver1D', 'ModeSolver2D', 'Mode', 'ModeSet', 'SampledFields', 'Material', 'Interval', 'Rectangle', 'Circle', 'Polygon', 'good_conductor_surface_impedance', 'BackendCapabilityError', 'ConfigurationError', 'FEMModeSolverError', 'GeometryError', 'MeshError', 'SolverError', 'load_result', 'NoResultError', 'PersistenceError']
