"""Shared building blocks used by every solver family.

Users normally import from :mod:`fdfd` directly; this package holds the
materials, shapes, Yee-grid operators, PML, eigensolvers, and result types
that the solver families share.
"""

from . import materials, shapes
from .errors import (
    BackendCapabilityError,
    ConfigurationError,
    FDFDError,
    GeometryError,
    NoResultError,
    PersistenceError,
    SolverError,
)
from .materials import GoodConductor, Material, SurfaceImpedance

__all__ = [
    "BackendCapabilityError",
    "ConfigurationError",
    "FDFDError",
    "GeometryError",
    "GoodConductor",
    "Material",
    "NoResultError",
    "PersistenceError",
    "SolverError",
    "SurfaceImpedance",
    "materials",
    "shapes",
]
