"""Fem Waveguide Scattering. Public user API for version 1.1.0."""
from .scattering import WaveguideScatteringSolver2D
from .results import ScatteringResult
from .farfield import ClosedContourFields, LayeredExterior, FarFieldResult
from .sweep import FrequencySweepResult
from .incident import IncidentMode
from .modes import Mode
from .modes import ModeSet
from .results import Diagnostic
from .results import DiagnosticReport
from cem_common.errors import BackendCapabilityError, ConfigurationError, GeometryError
from .exceptions import MaterialError
from cem_common.errors import MeshError
from .exceptions import ModeProjectionError
from .exceptions import ModeSolverError
from cem_common.errors import SolverError
from cem_common.errors import ViewerError
from .result_api import load_result
from cem_common import NoResultError
from cem_common import PersistenceError

__version__ = "1.1.0"
_FARFIELD_EXPORTS = ["ClosedContourFields", "LayeredExterior", "FarFieldResult"]
__all__ = ['WaveguideScatteringSolver2D', 'ScatteringResult', 'FrequencySweepResult', 'IncidentMode', 'Mode', 'ModeSet', 'Diagnostic', 'DiagnosticReport', 'BackendCapabilityError', 'ConfigurationError', 'GeometryError', 'MaterialError', 'MeshError', 'ModeProjectionError', 'ModeSolverError', 'SolverError', 'ViewerError', 'load_result', 'NoResultError', 'PersistenceError']
__all__ += _FARFIELD_EXPORTS
