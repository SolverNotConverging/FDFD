"""Exceptions shared by every solver family."""


class FDFDError(Exception):
    """Base class for all errors raised by fdfd."""


class ConfigurationError(FDFDError, ValueError):
    """An argument or solver configuration is invalid."""


class GeometryError(ConfigurationError):
    """A shape, material assignment or geometry edit is invalid."""


class BackendCapabilityError(FDFDError, NotImplementedError):
    """The solver does not support the requested material, geometry or option."""


class SolverError(FDFDError):
    """A numerical solve failed or did not meet its accuracy check."""


class NoResultError(SolverError):
    """The operation needs a result, but solve() has not been called."""


class PersistenceError(FDFDError, ValueError):
    """A result file is invalid, incompatible or could not be written."""


__all__ = [
    "BackendCapabilityError",
    "ConfigurationError",
    "FDFDError",
    "GeometryError",
    "NoResultError",
    "PersistenceError",
    "SolverError",
]
