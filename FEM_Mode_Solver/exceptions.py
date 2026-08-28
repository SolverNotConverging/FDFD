"""Exception hierarchy for the standalone FEM mode solvers."""


class FEMModeSolverError(Exception):
    """Base class for all public FEM mode-solver errors."""


class ConfigurationError(FEMModeSolverError, ValueError):
    """The continuous model or solver configuration is invalid."""


class GeometryError(ConfigurationError):
    """A geometry primitive or region operation is invalid."""


class MeshError(FEMModeSolverError):
    """The continuous model could not be discretized."""


class NotDiscretizedError(FEMModeSolverError):
    """A solve or plot was requested before discretization."""


class StaleDiscretizationError(FEMModeSolverError):
    """The continuous model changed after it was discretized."""


class SolverError(FEMModeSolverError):
    """The polynomial eigenproblem could not produce valid modes."""


class BackendCapabilityError(FEMModeSolverError, NotImplementedError):
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
