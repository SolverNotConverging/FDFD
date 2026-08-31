"""Nedelec finite-element QEP assembly for a cell periodic in ``z``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import bmat, csc_matrix, csr_matrix, eye
from scipy.sparse.linalg import norm as sparse_norm
from skfem import Basis, BilinearForm, MeshTet, asm
from skfem.element import ElementTetN1, ElementTetP1
from skfem.helpers import dot

from .exceptions import ConfigurationError
from .meshing_3d import PeriodicMesh3D
from .periodic import (
    PeriodicProlongation,
    build_node_prolongation,
    build_signed_edge_prolongation,
)


ComplexArray: TypeAlias = NDArray[np.complex128]
IntArray: TypeAlias = NDArray[np.int64]
MaterialEvaluator3D: TypeAlias = Callable[
    [NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
    tuple[ArrayLike, ArrayLike],
]


def _as_diagonal_field(values: ArrayLike, shape: tuple[int, ...], name: str) -> ComplexArray:
    array = np.asarray(values, dtype=np.complex128)
    target = (3, *shape)
    if array.ndim == 0:
        result = np.broadcast_to(array, target)
    elif array.shape == (3,):
        result = np.broadcast_to(array.reshape(3, *([1] * len(shape))), target)
    elif array.shape == shape:
        result = np.broadcast_to(array.reshape(1, *shape), target)
    elif array.shape == target:
        result = array
    elif array.shape == (*shape, 3):
        result = np.moveaxis(array, -1, 0)
    else:
        raise ConfigurationError(
            f"{name} must be scalar, length three, shape {shape}, "
            f"(3, *shape), or (*shape, 3); received {array.shape}."
        )
    result = np.asarray(result, dtype=np.complex128)
    if not np.isfinite(result).all():
        raise ConfigurationError(f"{name} contains a non-finite value.")
    return result


def evaluate_material_3d(
    material_at: MaterialEvaluator3D,
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
) -> tuple[ComplexArray, ComplexArray]:
    epsilon, mu = material_at(x, y, z)
    eps = _as_diagonal_field(epsilon, x.shape, "epsilon")
    permeability = _as_diagonal_field(mu, x.shape, "mu")
    if np.any(np.abs(eps) == 0.0) or np.any(np.abs(permeability) == 0.0):
        raise ConfigurationError("epsilon and mu entries must be nonzero.")
    return eps, permeability


def _z_cross(field: object) -> object:
    return np.stack((-field[1], field[0], np.zeros_like(field[0])))


@BilinearForm(dtype=np.complex128)
def _a0_form(u: object, v: object, w: object) -> object:
    return dot(w.inv_mu * u.curl, np.conj(v.curl)) - dot(
        w.epsilon * u, np.conj(v)
    )


@BilinearForm(dtype=np.complex128)
def _a1_form(u: object, v: object, w: object) -> object:
    trial_cross = _z_cross(u)
    test_cross = _z_cross(v)
    return 1j * (
        dot(w.inv_mu * u.curl, np.conj(test_cross))
        - dot(w.inv_mu * trial_cross, np.conj(v.curl))
    )


@BilinearForm(dtype=np.complex128)
def _a2_form(u: object, v: object, w: object) -> object:
    return dot(w.inv_mu * _z_cross(u), np.conj(_z_cross(v)))


@BilinearForm(dtype=np.complex128)
def _gauss0_form(u: object, phi: object, w: object) -> object:
    return -dot(w.epsilon * u, np.conj(phi.grad))


@BilinearForm(dtype=np.complex128)
def _gauss1_form(u: object, phi: object, w: object) -> object:
    return -1j * w.epsilon[2] * u[2] * np.conj(phi)


@dataclass(frozen=True, slots=True)
class PeriodicFEMSystem3D:
    basis: Basis
    scalar_basis: Basis
    physical_mesh: MeshTet
    computational_mesh: MeshTet
    mesh_data: PeriodicMesh3D
    A0: csr_matrix
    A1: csr_matrix
    A2: csr_matrix
    prolongation: PeriodicProlongation
    scalar_prolongation: PeriodicProlongation
    gauss0: csr_matrix
    gauss1: csr_matrix
    frequency: float
    k0: float
    material_at: MaterialEvaluator3D
    quadrature_order: int

    @property
    def ndofs(self) -> int:
        return int(self.A0.shape[0])

    @property
    def full_size(self) -> int:
        return int(self.prolongation.full_size)

    def polynomial(self, neff: complex) -> csr_matrix:
        value = complex(neff)
        return self.A0 + value * self.A1 + value**2 * self.A2

    def expand(self, vector: ArrayLike) -> ComplexArray:
        return self.prolongation.expand(vector)

    def relative_residual(self, vector: ArrayLike, neff: complex) -> float:
        reduced = np.asarray(vector, dtype=np.complex128)
        if reduced.shape != (self.ndofs,):
            raise ValueError(f"vector must have shape ({self.ndofs},).")
        value = complex(neff)
        terms = (
            self.A0 @ reduced,
            value * (self.A1 @ reduced),
            value**2 * (self.A2 @ reduced),
        )
        denominator = sum(float(np.linalg.norm(term)) for term in terms)
        residual = float(np.linalg.norm(terms[0] + terms[1] + terms[2]))
        return residual if denominator == 0.0 else residual / denominator

    def divergence_residual(self, vector: ArrayLike, neff: complex) -> float:
        """Return the dimensionless weak Gauss-defect energy.

        Both numerator and operator scale are squared 2-norms.  This energy
        convention is insensitive to eigenvector scaling and is more useful
        for mesh-convergence filtering than reporting the amplitude ratio.
        Explicitly, the returned value is
        ``(||(G0 + neff G1)x|| / ((||G0|| + |neff| ||G1||)||x||))**2``.
        """

        reduced = np.asarray(vector, dtype=np.complex128)
        value = complex(neff)
        first = self.gauss0 @ reduced
        second = value * (self.gauss1 @ reduced)
        denominator = (
            float(sparse_norm(self.gauss0)) * float(np.linalg.norm(reduced))
            + abs(value) * float(sparse_norm(self.gauss1)) * float(np.linalg.norm(reduced))
        )
        residual = float(np.linalg.norm(first + second))
        return residual**2 if denominator == 0.0 else (residual / denominator) ** 2

    def relative_hermiticity_errors(self) -> tuple[float, float, float]:
        result: list[float] = []
        for matrix in (self.A0, self.A1, self.A2):
            denominator = float(sparse_norm(matrix))
            numerator = float(sparse_norm(matrix - matrix.getH()))
            result.append(numerator if denominator == 0.0 else numerator / denominator)
        return tuple(result)  # type: ignore[return-value]


def assemble_periodic_system_3d(
    mesh_data: PeriodicMesh3D,
    *,
    frequency: float,
    k0: float,
    material_at: MaterialEvaluator3D,
    quadrature_order: int = 3,
) -> PeriodicFEMSystem3D:
    """Assemble and periodically reduce the electric-field QEP."""

    if not isinstance(mesh_data, PeriodicMesh3D):
        raise TypeError("mesh_data must be a PeriodicMesh3D instance.")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ConfigurationError("frequency must be finite and positive.")
    if not np.isfinite(k0) or k0 <= 0.0:
        raise ConfigurationError("k0 must be finite and positive.")
    if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order or quadrature_order < 2:
        raise ConfigurationError("quadrature_order must be an integer of at least two.")

    physical_mesh = mesh_data.mesh
    computational_mesh = physical_mesh.scaled(float(k0))
    basis = Basis(
        computational_mesh, ElementTetN1(), intorder=int(quadrature_order)
    )
    coordinates = basis.global_coordinates()
    epsilon, mu = evaluate_material_3d(
        material_at,
        coordinates[0] / k0,
        coordinates[1] / k0,
        coordinates[2] / k0,
    )
    full_matrices = tuple(
        asm(form, basis, epsilon=epsilon, inv_mu=1.0 / mu).astype(
            np.complex128, copy=False
        )
        for form in (_a0_form, _a1_form, _a2_form)
    )

    pec_facets = np.unique(
        np.concatenate(
            [
                np.asarray(mesh_data.boundary_facets.get(name, ()), dtype=np.int64)
                for name in ("outer_pec", "pec")
            ]
        )
    )
    constrained_edges = (
        np.asarray(basis.get_dofs(facets=pec_facets).all(), dtype=np.int64)
        if pec_facets.size
        else np.empty(0, dtype=np.int64)
    )
    node_pairs = np.asarray(mesh_data.periodic_node_pairs, dtype=np.int64)
    prolongation = build_signed_edge_prolongation(
        physical_mesh.edges,
        node_pairs[:, 0],
        node_pairs[:, 1],
        node_count=physical_mesh.nvertices,
        constrained_edges=constrained_edges,
    )
    reduced = tuple(
        prolongation.reduce_matrix(matrix).tocsr() for matrix in full_matrices
    )
    if reduced[0].shape[0] < 3:
        raise ConfigurationError("Periodic/PEC constraints leave too few edge DOFs.")

    scalar_basis = Basis(
        computational_mesh, ElementTetP1(), intorder=int(quadrature_order)
    )
    scalar_coordinates = scalar_basis.global_coordinates()
    gauss_epsilon, _ = evaluate_material_3d(
        material_at,
        scalar_coordinates[0] / k0,
        scalar_coordinates[1] / k0,
        scalar_coordinates[2] / k0,
    )
    gauss0_full = asm(
        _gauss0_form, basis, scalar_basis, epsilon=gauss_epsilon
    ).astype(np.complex128, copy=False)
    gauss1_full = asm(
        _gauss1_form, basis, scalar_basis, epsilon=gauss_epsilon
    ).astype(np.complex128, copy=False)
    # The weak divergence identity is tested with H1 functions that vanish on
    # every non-periodic boundary.  Keeping boundary test rows would measure
    # legitimate surface charge/flux terms at PEC/PMC walls instead of the
    # volume Gauss-law defect used for spurious-mode rejection.
    nonperiodic_facets = np.unique(
        np.concatenate(
            [
                np.asarray(facets, dtype=np.int64)
                for name, facets in mesh_data.boundary_facets.items()
                if name not in ("periodic_master", "periodic_slave")
                and len(facets)
            ]
        )
    ) if any(
        name not in ("periodic_master", "periodic_slave") and len(facets)
        for name, facets in mesh_data.boundary_facets.items()
    ) else np.empty(0, dtype=np.int64)
    constrained_scalar_nodes = (
        np.unique(physical_mesh.facets[:, nonperiodic_facets])
        if nonperiodic_facets.size
        else np.empty(0, dtype=np.int64)
    )
    try:
        scalar_prolongation = build_node_prolongation(
            physical_mesh.nvertices,
            node_pairs[:, 0],
            node_pairs[:, 1],
            constrained_nodes=constrained_scalar_nodes,
        )
    except ConfigurationError:
        # On a one-element-thick diagnostic mesh there may be no admissible
        # scalar test DOF.  Retain the periodic space; convergence fixtures use
        # meshes with interior nodes and therefore exercise the strict test.
        scalar_prolongation = build_node_prolongation(
            physical_mesh.nvertices,
            node_pairs[:, 0],
            node_pairs[:, 1],
        )
    gauss0 = (
        scalar_prolongation.matrix.getH()
        @ gauss0_full
        @ prolongation.matrix
    ).tocsr()
    gauss1 = (
        scalar_prolongation.matrix.getH()
        @ gauss1_full
        @ prolongation.matrix
    ).tocsr()

    return PeriodicFEMSystem3D(
        basis=basis,
        scalar_basis=scalar_basis,
        physical_mesh=physical_mesh,
        computational_mesh=computational_mesh,
        mesh_data=mesh_data,
        A0=reduced[0],
        A1=reduced[1],
        A2=reduced[2],
        prolongation=prolongation,
        scalar_prolongation=scalar_prolongation,
        gauss0=gauss0,
        gauss1=gauss1,
        frequency=float(frequency),
        k0=float(k0),
        material_at=material_at,
        quadrature_order=int(quadrature_order),
    )


def linearized_pencil_3d(
    system: PeriodicFEMSystem3D,
) -> tuple[csc_matrix, csc_matrix]:
    n = system.ndofs
    identity = eye(n, format="csc", dtype=np.complex128)
    zero = csr_matrix((n, n), dtype=np.complex128)
    left = bmat(
        ((zero, identity), (-system.A0, -system.A1)),
        format="csc",
        dtype=np.complex128,
    )
    right = bmat(
        ((identity, zero), (zero, system.A2)),
        format="csc",
        dtype=np.complex128,
    )
    return left, right


__all__ = [
    "MaterialEvaluator3D",
    "PeriodicFEMSystem3D",
    "assemble_periodic_system_3d",
    "evaluate_material_3d",
    "linearized_pencil_3d",
]
