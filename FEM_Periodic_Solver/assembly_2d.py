"""Scalar TE/TM weak forms for a two-dimensional periodic cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.sparse import bmat, csc_matrix, csr_matrix, eye
from scipy.sparse.linalg import norm as sparse_norm
from skfem import Basis, BilinearForm, ElementTriP1, asm

from .exceptions import ConfigurationError, SolverError
from .meshing_2d import FEMPeriodicMesh2D
from .periodic import PeriodicProlongation, build_node_prolongation

ComplexArray: TypeAlias = NDArray[np.complex128]
MaterialEvaluator: TypeAlias = Callable[
    [NDArray[np.floating], NDArray[np.floating]],
    tuple[ComplexArray, ComplexArray],
]
Polarization = Literal["TE", "TM"]


def _diagonal_field(value: ArrayLike, shape: tuple[int, ...], name: str) -> ComplexArray:
    array = np.asarray(value, dtype=np.complex128)
    if array.shape == (3,):
        array = np.broadcast_to(array.reshape((3, *([1] * len(shape)))), (3, *shape))
    if array.shape != (3, *shape):
        raise ConfigurationError(
            f"{name} must have shape (3, *quadrature_shape); received {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ConfigurationError(f"{name} contains a non-finite value.")
    return np.asarray(array, dtype=np.complex128)


def evaluate_material(
    material_at: MaterialEvaluator,
    x: NDArray[np.floating],
    z: NDArray[np.floating],
) -> tuple[ComplexArray, ComplexArray]:
    epsilon, mu = material_at(x, z)
    eps = _diagonal_field(epsilon, x.shape, "epsilon")
    permeability = _diagonal_field(mu, x.shape, "mu")
    if np.any(np.abs(eps) == 0.0) or np.any(np.abs(permeability) == 0.0):
        raise ConfigurationError("epsilon and mu must be nonzero at every quadrature point.")
    return eps, permeability


@BilinearForm(dtype=np.complex128)
def _a0(u: object, v: object, w: object) -> object:
    return (
        w.c_x * u.grad[0] * np.conj(v.grad[0])
        + w.c_z * u.grad[1] * np.conj(v.grad[1])
        - w.mass * u * np.conj(v)
    )


@BilinearForm(dtype=np.complex128)
def _a1(u: object, v: object, w: object) -> object:
    return 1j * w.c_z * (
        np.conj(v) * u.grad[1] - np.conj(v.grad[1]) * u
    )


@BilinearForm(dtype=np.complex128)
def _a2(u: object, v: object, w: object) -> object:
    return w.c_z * u * np.conj(v)


@dataclass(frozen=True, slots=True)
class PeriodicFEMSystem2D:
    """One reduced scalar QEP and the data needed for field reconstruction."""

    polarization: Polarization
    basis: Basis
    mesh_data: FEMPeriodicMesh2D
    prolongation: PeriodicProlongation
    A0: csr_matrix
    A1: csr_matrix
    A2: csr_matrix
    frequency: float
    k0: float
    material_at: MaterialEvaluator
    quadrature_order: int

    @property
    def ndofs(self) -> int:
        return int(self.A0.shape[0])

    @property
    def full_size(self) -> int:
        return self.prolongation.full_size

    def polynomial(self, neff: complex) -> csr_matrix:
        value = complex(neff)
        return (self.A0 + value * self.A1 + value**2 * self.A2).tocsr()

    def expand(self, vector: ArrayLike) -> ComplexArray:
        return self.prolongation.expand(vector)

    def relative_residual(self, vector: ArrayLike, neff: complex) -> float:
        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.ndofs,):
            raise ValueError(f"vector must have shape ({self.ndofs},).")
        terms = (
            np.asarray(self.A0 @ values),
            complex(neff) * np.asarray(self.A1 @ values),
            complex(neff) ** 2 * np.asarray(self.A2 @ values),
        )
        denominator = sum(float(np.linalg.norm(term)) for term in terms)
        residual = float(np.linalg.norm(terms[0] + terms[1] + terms[2]))
        return residual if denominator == 0.0 else residual / denominator

    def relative_hermiticity_errors(self) -> tuple[float, float, float]:
        result: list[float] = []
        for matrix in (self.A0, self.A1, self.A2):
            denominator = float(sparse_norm(matrix))
            numerator = float(sparse_norm(matrix - matrix.getH()))
            result.append(numerator if denominator == 0.0 else numerator / denominator)
        return tuple(result)  # type: ignore[return-value]


def _dirichlet_facets(mesh: FEMPeriodicMesh2D, polarization: Polarization) -> NDArray[np.int64]:
    kind = "pec" if polarization == "TE" else "pmc"
    parts = [
        values
        for name, values in mesh.boundary_facets.items()
        if (name == kind or name == f"outer_{kind}") and values.size
    ]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts)).astype(np.int64, copy=False)


def assemble_periodic_system_2d(
    mesh_data: FEMPeriodicMesh2D,
    *,
    polarization: Polarization,
    frequency: float,
    k0: float,
    material_at: MaterialEvaluator,
    quadrature_order: int = 4,
) -> PeriodicFEMSystem2D:
    """Assemble and periodically reduce the analytic scalar QEP."""

    normalized = str(polarization).strip().upper()
    if normalized not in ("TE", "TM"):
        raise ConfigurationError("polarization must be 'TE' or 'TM'.")
    if not np.isfinite(frequency) or frequency <= 0.0 or not np.isfinite(k0) or k0 <= 0.0:
        raise ConfigurationError("frequency and k0 must be finite and positive.")
    if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order or quadrature_order < 2:
        raise ConfigurationError("quadrature_order must be an integer of at least two.")

    computational_mesh = mesh_data.mesh.scaled(float(k0))
    basis = Basis(computational_mesh, ElementTriP1(), intorder=int(quadrature_order))
    coordinates = np.asarray(basis.global_coordinates() / k0, dtype=np.float64)
    epsilon, mu = evaluate_material(material_at, coordinates[0], coordinates[1])
    if normalized == "TE":
        c_x = 1.0 / mu[2]
        c_z = 1.0 / mu[0]
        mass = epsilon[1]
    else:
        c_x = 1.0 / epsilon[2]
        c_z = 1.0 / epsilon[0]
        mass = mu[1]
    full_matrices = tuple(
        asm(form, basis, c_x=c_x, c_z=c_z, mass=mass).astype(np.complex128, copy=False).tocsr()
        for form in (_a0, _a1, _a2)
    )
    facets = _dirichlet_facets(mesh_data, normalized)  # type: ignore[arg-type]
    constrained = (
        np.asarray(basis.get_dofs(facets=facets).all(), dtype=np.int64)
        if facets.size
        else np.empty(0, dtype=np.int64)
    )
    prolongation = build_node_prolongation(
        basis.N,
        mesh_data.slave_nodes,
        mesh_data.master_nodes,
        constrained_nodes=constrained,
    )
    reduced = tuple(prolongation.reduce_matrix(matrix) for matrix in full_matrices)
    if reduced[0].shape[0] < 2:
        raise ConfigurationError("Periodic and boundary constraints leave too few scalar DOFs.")
    return PeriodicFEMSystem2D(
        polarization=normalized,  # type: ignore[arg-type]
        basis=basis,
        mesh_data=mesh_data,
        prolongation=prolongation,
        A0=reduced[0],
        A1=reduced[1],
        A2=reduced[2],
        frequency=float(frequency),
        k0=float(k0),
        material_at=material_at,
        quadrature_order=int(quadrature_order),
    )


def linearized_pencil(system: PeriodicFEMSystem2D) -> tuple[csc_matrix, csc_matrix]:
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


def _dense_candidates(
    system: PeriodicFEMSystem2D,
    target: complex,
    candidate_count: int,
) -> tuple[ComplexArray, ComplexArray, NDArray[np.float64], str]:
    left, right = linearized_pencil(system)
    try:
        homogeneous, eigenvectors = linalg.eig(
            left.toarray(),
            right.toarray(),
            right=True,
            check_finite=False,
            homogeneous_eigvals=True,
        )
    except linalg.LinAlgError as exc:
        raise SolverError("Dense homogeneous QZ failed to converge.") from exc
    alpha, denominator = homogeneous
    scale = np.maximum(np.abs(alpha), np.abs(denominator))
    finite = np.abs(denominator) > 256.0 * np.finfo(float).eps * np.maximum(scale, 1.0)
    values = np.asarray(alpha[finite] / denominator[finite], dtype=np.complex128)
    vectors = np.asarray(eigenvectors[: system.ndofs, finite], dtype=np.complex128)
    order = np.argsort(np.abs(values - target))[: min(values.size, candidate_count)]
    values = values[order]
    vectors = vectors[:, order]
    residuals = np.asarray(
        [system.relative_residual(vectors[:, index], values[index]) for index in range(values.size)],
        dtype=np.float64,
    )
    return values, vectors, residuals, "dense-qz"


def _refined_candidates(
    system: PeriodicFEMSystem2D,
    target: complex,
    candidate_count: int,
    tolerance: float,
    arnoldi_backend: str,
) -> tuple[ComplexArray, ComplexArray, NDArray[np.float64], str]:
    left, right = linearized_pencil(system)
    try:
        import periodic_eigensolver
    except ImportError as exc:
        raise SolverError(
            "The refined eigensolver requires the periodic-eigensolver distribution."
        ) from exc
    public_backend = "python" if arnoldi_backend == "numpy" else arnoldi_backend
    if hasattr(periodic_eigensolver, "solve_generalized"):
        result = periodic_eigensolver.solve_generalized(
            left,
            right,
            sigma=complex(target),
            num_modes=candidate_count,
            tol=float(tolerance),
            backend=public_backend,
        )
        if not result.converged:
            raise SolverError(
                "Refined Arnoldi exhausted its restart budget; best original-pencil "
                f"residual is {float(np.max(result.physical_residuals)):.3e}."
            )
        values = np.asarray(result.eigenvalues, dtype=np.complex128)
        full_vectors = np.asarray(result.eigenvectors, dtype=np.complex128)
        residuals = np.asarray(result.physical_residuals, dtype=np.float64)
        backend_name = str(result.backend)
    else:  # compatibility with the first periodic-eigensolver package release
        values, full_vectors, residuals, _ = periodic_eigensolver.refined_shift_invert_arnoldi(
            left,
            right,
            complex(target),
            candidate_count,
            float(tolerance),
            kernel_backend=arnoldi_backend,
        )
        values = np.asarray(values, dtype=np.complex128)
        full_vectors = np.asarray(full_vectors, dtype=np.complex128)
        residuals = np.asarray(residuals, dtype=np.float64)
        backend_name = str(arnoldi_backend)
    vectors = np.asarray(full_vectors[: system.ndofs], dtype=np.complex128)
    return values, vectors, residuals, f"refined-{backend_name}"


def solve_qep_candidates(
    system: PeriodicFEMSystem2D,
    *,
    target: complex,
    candidate_count: int,
    tolerance: float = 1e-10,
    eigensolver: str = "auto",
    arnoldi_backend: str = "auto",
    dense_linearization_limit: int = 700,
) -> tuple[ComplexArray, ComplexArray, NDArray[np.float64], str]:
    """Return roots and scalar coefficient vectors nearest ``target``."""

    if candidate_count < 1:
        raise ConfigurationError("candidate_count must be positive.")
    method = str(eigensolver).strip().lower()
    if method not in ("auto", "dense", "refined"):
        raise ConfigurationError("eigensolver must be 'auto', 'dense', or 'refined'.")
    backend = str(arnoldi_backend).strip().lower()
    if backend not in ("auto", "cython", "python", "numpy"):
        raise ConfigurationError("arnoldi_backend must be 'auto', 'cython', or 'python'.")
    size = 2 * system.ndofs
    if method == "auto":
        method = "dense" if size <= int(dense_linearization_limit) else "refined"
    if method == "dense":
        return _dense_candidates(system, complex(target), int(candidate_count))
    return _refined_candidates(
        system,
        complex(target),
        int(candidate_count),
        float(tolerance),
        backend,
    )


__all__ = [
    "MaterialEvaluator",
    "PeriodicFEMSystem2D",
    "assemble_periodic_system_2d",
    "evaluate_material",
    "linearized_pencil",
    "solve_qep_candidates",
]
