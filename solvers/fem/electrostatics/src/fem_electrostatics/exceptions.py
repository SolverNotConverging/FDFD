"""Exceptions raised by the finite-element electrostatic solver."""
from fem_common import errors as _common



class ElectrostaticSolverError(_common.FEMError, RuntimeError):
    """Base exception for this package."""


class GeometryError(ElectrostaticSolverError, _common.GeometryError):
    """The continuous geometry or material definition is invalid."""


class MeshError(ElectrostaticSolverError, _common.MeshError):
    """Gmsh could not produce a valid conforming mesh."""


class NotDiscretizedError(ElectrostaticSolverError):
    """An operation requires a current finite-element mesh."""


class SolverError(ElectrostaticSolverError, _common.SolverError):
    """The assembled electrostatic boundary-value problem cannot be solved."""
