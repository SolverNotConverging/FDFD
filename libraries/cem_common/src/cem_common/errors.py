"""Errors applications can handle consistently across FEM solver families."""
class CEMError(Exception):
    """Base class for user-facing FEM errors."""

class BackendCapabilityError(CEMError, NotImplementedError):
    """A material, geometry, or operation is unsupported by this solver."""


class ConfigurationError(CEMError, ValueError):
    """Invalid or unsupported configuration."""

class GeometryError(ConfigurationError):
    """Invalid physical geometry or material configuration."""

class MeshError(CEMError):
    """A mesh could not be generated or used."""

class SolverError(CEMError):
    """A numerical solve failed."""

class NoResultError(SolverError):
    """The requested operation needs a successful, current result."""

class PersistenceError(CEMError, ValueError):
    """An archive is invalid, incompatible, or cannot be written."""

class ViewerError(CEMError):
    """An interactive viewer could not be opened."""
