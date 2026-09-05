"""Errors applications can handle consistently across FEM solver families."""
class FEMError(Exception):
    """Base class for user-facing FEM errors."""

class ConfigurationError(FEMError, ValueError):
    """Invalid or unsupported configuration."""

class GeometryError(ConfigurationError):
    """Invalid physical geometry or material configuration."""

class MeshError(FEMError):
    """A mesh could not be generated or used."""

class SolverError(FEMError):
    """A numerical solve failed."""

class NoResultError(SolverError):
    """The requested operation needs a successful, current result."""

class PersistenceError(FEMError, ValueError):
    """An archive is invalid, incompatible, or cannot be written."""

class ViewerError(FEMError):
    """An interactive viewer could not be opened."""
