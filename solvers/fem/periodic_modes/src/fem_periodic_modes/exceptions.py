"""Public exception hierarchy for :mod:`fem_periodic_modes`."""
from fem_common import errors as _common



class FEMPeriodicSolverError(_common.FEMError):
    """Base class for errors raised by the periodic FEM package."""


class ConfigurationError(FEMPeriodicSolverError, _common.ConfigurationError):
    """A solver option or material is invalid."""


class GeometryError(ConfigurationError, _common.GeometryError):
    """A geometry object cannot be represented by the periodic cell."""


class MeshError(FEMPeriodicSolverError, _common.MeshError):
    """Gmsh could not create a conforming periodic mesh."""


class NotDiscretizedError(FEMPeriodicSolverError):
    """An assembled operation was requested before :meth:`discretize`."""


class StaleDiscretizationError(FEMPeriodicSolverError):
    """The continuous geometry changed after discretization."""


class SolverError(FEMPeriodicSolverError, _common.SolverError):
    """The polynomial eigenproblem did not produce the requested modes."""


class BackendCapabilityError(FEMPeriodicSolverError, NotImplementedError):
    """A requested feature is intentionally unavailable in this backend."""


from fem_common.errors import PersistenceError as _PersistenceError


class PersistenceError(FEMPeriodicSolverError, _PersistenceError):
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
