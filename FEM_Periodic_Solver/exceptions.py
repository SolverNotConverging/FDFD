"""Public exception hierarchy for :mod:`FEM_Periodic_Solver`."""


class FEMPeriodicSolverError(Exception):
    """Base class for errors raised by the periodic FEM package."""


class ConfigurationError(FEMPeriodicSolverError, ValueError):
    """A solver option or material is invalid."""


class GeometryError(ConfigurationError):
    """A geometry object cannot be represented by the periodic cell."""


class MeshError(FEMPeriodicSolverError):
    """Gmsh could not create a conforming periodic mesh."""


class NotDiscretizedError(FEMPeriodicSolverError):
    """An assembled operation was requested before :meth:`discretize`."""


class StaleDiscretizationError(FEMPeriodicSolverError):
    """The continuous geometry changed after discretization."""


class SolverError(FEMPeriodicSolverError):
    """The polynomial eigenproblem did not produce the requested modes."""


class BackendCapabilityError(FEMPeriodicSolverError, NotImplementedError):
    """A requested feature is intentionally unavailable in this backend."""


class PersistenceError(FEMPeriodicSolverError):
    """A periodic-mode HDF5 archive is invalid or cannot be written."""


__all__ = [
    "BackendCapabilityError",
    "ConfigurationError",
    "FEMPeriodicSolverError",
    "GeometryError",
    "MeshError",
    "NotDiscretizedError",
    "PersistenceError",
    "SolverError",
    "StaleDiscretizationError",
]
