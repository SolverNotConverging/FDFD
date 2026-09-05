"""Exception hierarchy for the standalone FEM mode solvers."""
from cem_common import errors as _common



class FEMModeSolverError(_common.CEMError):
    """Base class for all public FEM mode-solver errors."""


class ConfigurationError(FEMModeSolverError, _common.ConfigurationError):
    """The continuous model or solver configuration is invalid."""


class GeometryError(ConfigurationError, _common.GeometryError):
    """A geometry primitive or region operation is invalid."""


class MeshError(FEMModeSolverError, _common.MeshError):
    """The continuous model could not be discretized."""


class NotDiscretizedError(FEMModeSolverError):
    """A solve or plot was requested before discretization."""


class StaleDiscretizationError(FEMModeSolverError):
    """The continuous model changed after it was discretized."""


class SolverError(FEMModeSolverError, _common.SolverError):
    """The polynomial eigenproblem could not produce valid modes."""


class BackendCapabilityError(FEMModeSolverError, _common.BackendCapabilityError):
    """A requested physical feature is not supported by this FEM backend."""


__all__ = [
    "BackendCapabilityError",
    "ConfigurationError",
    "FEMModeSolverError",
    "GeometryError",
    "MeshError",
    "NotDiscretizedError",
    "SolverError",
    "StaleDiscretizationError",
]
