"""Shared user-facing FEM contracts; numerical assembly stays in each solver."""
from .contracts import FEMSolverMixin, ResultMixin, MeshSnapshot, mesh_snapshot
from .errors import FEMError, ConfigurationError, GeometryError, MeshError, SolverError, NoResultError, PersistenceError, ViewerError

TIME_CONVENTION = "exp(+i*omega*t)"
__version__ = "1.0.0"
__all__ = ["FEMError", "ConfigurationError", "GeometryError", "MeshError", "SolverError", "NoResultError", "PersistenceError", "ViewerError", "MeshSnapshot"]
