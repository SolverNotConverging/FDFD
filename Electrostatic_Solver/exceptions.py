"""Exceptions raised by the finite-element electrostatic solver."""


class ElectrostaticSolverError(RuntimeError):
    """Base exception for this package."""


class GeometryError(ElectrostaticSolverError, ValueError):
    """The continuous geometry or material definition is invalid."""


class MeshError(ElectrostaticSolverError):
    """Gmsh could not produce a valid conforming mesh."""


class NotDiscretizedError(ElectrostaticSolverError):
    """An operation requires a current finite-element mesh."""


class SolverError(ElectrostaticSolverError):
    """The assembled electrostatic boundary-value problem cannot be solved."""
