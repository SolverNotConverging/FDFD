"""Shared materials, shapes, result metadata, and actionable errors."""
from .contracts import MeshSnapshot
from .errors import CEMError, ConfigurationError, GeometryError, MeshError, SolverError, NoResultError, PersistenceError, ViewerError
from .errors import BackendCapabilityError
from .materials import Material, GoodConductor, SurfaceImpedance
from . import materials, shapes

TIME_CONVENTION = "exp(+i*omega*t)"
__version__ = "1.0.0"
__all__ = ["Material", "GoodConductor", "SurfaceImpedance", "materials", "shapes", "CEMError", "BackendCapabilityError", "ConfigurationError", "GeometryError", "MeshError", "SolverError", "NoResultError", "PersistenceError", "ViewerError", "MeshSnapshot"]
