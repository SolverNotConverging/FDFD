"""Mixed Nedelec--Lagrange assembly for two-dimensional waveguide modes.

The public convention is ``exp(+i*omega*t - i*beta*z)``.  After scaling the
transverse coordinates by ``k0`` and writing ``lambda = beta/k0``, the electric
field pencil is

``(A0 + lambda*A1 + lambda**2*A2) e = 0``.

The transverse field ``(Ex, Ey)`` uses an H(curl) Nedelec space and
the longitudinal field ``Ez`` uses continuous scalars (N1/P1 or N2/P2).
These compatible pairings preserve the
curl conformity needed at material interfaces and avoids representing the
longitudinal component with edge degrees of freedom.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.sparse import bmat, csc_matrix, csr_matrix, eye
from scipy.sparse.linalg import (
    ArpackNoConvergence,
    LinearOperator,
    eigs,
    norm as sparse_norm,
    splu,
)
from skfem import Basis, BilinearForm, FacetBasis, MeshTri, asm
from skfem.element import ElementComposite, ElementTriN1, ElementTriN2, ElementTriP1, ElementTriP2

from .boundaries import validate_surface_impedance
from .constants import ETA_0
from .exceptions import ConfigurationError, SolverError


ComplexArray: TypeAlias = NDArray[np.complex128]
IntArray: TypeAlias = NDArray[np.int64]
MaterialEvaluator: TypeAlias = Callable[
    [NDArray[np.floating], NDArray[np.floating]],
    tuple[ArrayLike, ArrayLike],
]
ImpedanceBoundaryInput: TypeAlias = tuple[ArrayLike, complex | float]


def _as_diagonal_field(
    values: ArrayLike,
    shape: tuple[int, ...],
    name: str,
) -> ComplexArray:
    """Broadcast scalar/diagonal material data to ``(3, *shape)``."""

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


def evaluate_material(
    material_at: MaterialEvaluator,
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> tuple[ComplexArray, ComplexArray]:
    """Evaluate and validate relative diagonal ``(epsilon, mu)`` fields."""

    epsilon, mu = material_at(x, y)
    eps = _as_diagonal_field(epsilon, x.shape, "epsilon")
    permeability = _as_diagonal_field(mu, x.shape, "mu")
    if np.any(np.abs(permeability) == 0.0):
        raise ConfigurationError("mu entries must be nonzero at every quadrature point.")
    return eps, permeability


def _facet_indices(mesh: MeshTri, values: ArrayLike, name: str) -> IntArray:
    """Return unique, validated boundary-facet indices."""

    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ConfigurationError(f"{name} must be a one-dimensional facet array.")
    if raw.dtype.kind == "b":
        raise ConfigurationError(f"{name} must contain integer facet indices, not booleans.")
    try:
        numeric = np.asarray(raw, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must contain integer facet indices.") from exc
    if not np.isfinite(numeric).all() or np.any(numeric.imag != 0.0):
        raise ConfigurationError(f"{name} must contain finite real facet indices.")
    real = numeric.real
    if np.any(real != np.floor(real)):
        raise ConfigurationError(f"{name} must contain integer facet indices.")
    facets = np.unique(np.asarray(real, dtype=np.int64))
    if facets.size and (facets[0] < 0 or facets[-1] >= mesh.nfacets):
        raise ConfigurationError(
            f"{name} contains an index outside the mesh facet range 0..{mesh.nfacets - 1}."
        )
    boundary_facets = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    if facets.size and not np.all(np.isin(facets, boundary_facets)):
        raise ConfigurationError(f"{name} may contain boundary facets only.")
    facets.setflags(write=False)
    return facets


def _impedance_boundary_data(
    mesh: MeshTri,
    values: Sequence[ImpedanceBoundaryInput] | None,
) -> tuple[tuple[IntArray, complex], ...]:
    """Validate ``(facets, Zs)`` pairs and reject ambiguous overlaps."""

    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise ConfigurationError(
            "impedance_boundaries must be a sequence of (facets, Zs) pairs."
        )
    try:
        entries = tuple(values)
    except TypeError as exc:
        raise ConfigurationError(
            "impedance_boundaries must be a sequence of (facets, Zs) pairs."
        ) from exc

    result: list[tuple[IntArray, complex]] = []
    occupied = np.empty(0, dtype=np.int64)
    for index, entry in enumerate(entries):
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) != 2:
            raise ConfigurationError(
                "Each impedance boundary must be a (facets, Zs) pair."
            )
        facets = _facet_indices(mesh, entry[0], f"impedance_boundaries[{index}].facets")
        impedance = validate_surface_impedance(entry[1])
        if np.intersect1d(occupied, facets, assume_unique=True).size:
            raise ConfigurationError(
                "A mesh facet cannot belong to more than one impedance boundary."
            )
        if facets.size:
            occupied = np.union1d(occupied, facets).astype(np.int64, copy=False)
            result.append((facets, impedance))
    return tuple(result)


@BilinearForm(dtype=np.complex128)
def _a0_form(et: object, ez: object, vt: object, vz: object, w: object) -> object:
    eps = w.epsilon
    q = w.inv_mu
    curl = q[2] * et.curl * np.conj(vt.curl)
    longitudinal_gradient = (
        q[1] * ez.grad[0] * np.conj(vz.grad[0])
        + q[0] * ez.grad[1] * np.conj(vz.grad[1])
    )
    electric_mass = (
        eps[0] * et[0] * np.conj(vt[0])
        + eps[1] * et[1] * np.conj(vt[1])
        + eps[2] * ez * np.conj(vz)
    )
    return curl + longitudinal_gradient - electric_mass


@BilinearForm(dtype=np.complex128)
def _a1_form(et: object, ez: object, vt: object, vz: object, w: object) -> object:
    q = w.inv_mu
    trial_longitudinal = -1j * (
        q[1] * ez.grad[0] * np.conj(vt[0])
        + q[0] * ez.grad[1] * np.conj(vt[1])
    )
    test_longitudinal = 1j * (
        q[1] * et[0] * np.conj(vz.grad[0])
        + q[0] * et[1] * np.conj(vz.grad[1])
    )
    return trial_longitudinal + test_longitudinal


@BilinearForm(dtype=np.complex128)
def _a2_form(et: object, ez: object, vt: object, vz: object, w: object) -> object:
    del ez, vz
    q = w.inv_mu
    return q[1] * et[0] * np.conj(vt[0]) + q[0] * et[1] * np.conj(vt[1])


@BilinearForm(dtype=np.complex128)
def _surface_impedance_form(
    et: object,
    ez: object,
    vt: object,
    vz: object,
    w: object,
) -> object:
    r"""Dimensionless Leontovich contribution on one impedance boundary.

    The outward normal points from the dielectric solve domain into the
    conductor and ``E_t = Zs (H x n)``.  Under
    ``exp(+i*omega*t - i*beta*z)`` integration by parts therefore contributes

    ``+i*eta0/Zs int[(E_xy.t)(v_xy.t)* + Ez*vz*] ds``

    to ``A0``.  The sign makes the boundary quadratic form dissipative for a
    passive positive-resistance surface and produces ``Im(neff) < 0`` for a
    forward attenuating mode.
    """

    # Either tangent orientation is valid because it appears in both the
    # trial and conjugated test trace.  This one is a +90 degree rotation of
    # the outward in-plane normal.
    trial_tangent = -w.n[1] * et[0] + w.n[0] * et[1]
    test_tangent = -w.n[1] * vt[0] + w.n[0] * vt[1]
    return w.surface_coefficient * (
        trial_tangent * np.conj(test_tangent) + ez * np.conj(vz)
    )


@BilinearForm(dtype=np.complex128)
def _gauss_transverse(et: object, phi: object, w: object) -> object:
    eps = w.epsilon
    return -(
        eps[0] * et[0] * np.conj(phi.grad[0])
        + eps[1] * et[1] * np.conj(phi.grad[1])
    )


@BilinearForm(dtype=np.complex128)
def _gauss_longitudinal(ez: object, phi: object, w: object) -> object:
    return -1j * w.epsilon[2] * ez * np.conj(phi)


@dataclass(frozen=True, slots=True)
class ModeFEMSystem2D:
    """Reduced quadratic pencil plus data needed to reconstruct FEM fields."""

    basis: Basis
    physical_mesh: MeshTri
    computational_mesh: MeshTri
    A0: csr_matrix
    A1: csr_matrix
    A2: csr_matrix
    free_dofs: IntArray
    full_size: int
    transverse_indices: IntArray
    longitudinal_indices: IntArray
    gauss_transverse: csr_matrix
    gauss_longitudinal: csr_matrix
    gauss_test_dofs: IntArray
    frequency: float
    k0: float
    boundary: str
    material_at: MaterialEvaluator
    quadrature_order: int
    impedance_boundaries: tuple[tuple[IntArray, complex], ...]

    @property
    def ndofs(self) -> int:
        return int(self.A0.shape[0])

    def polynomial(self, neff: complex) -> csr_matrix:
        value = complex(neff)
        return self.A0 + value * self.A1 + value**2 * self.A2

    def expand(self, vector: ArrayLike) -> ComplexArray:
        reduced = np.asarray(vector, dtype=np.complex128)
        if reduced.shape != (self.ndofs,):
            raise ValueError(
                f"mode vector must have shape ({self.ndofs},); received {reduced.shape}."
            )
        full = np.zeros(self.full_size, dtype=np.complex128)
        full[self.free_dofs] = reduced
        return full

    def split_full(self, vector: ArrayLike) -> tuple[ComplexArray, ComplexArray]:
        full = np.asarray(vector, dtype=np.complex128)
        if full.shape != (self.full_size,):
            raise ValueError(
                f"full mode vector must have shape ({self.full_size},); received {full.shape}."
            )
        return full[self.transverse_indices], full[self.longitudinal_indices]

    def relative_residual(self, vector: ArrayLike, neff: complex) -> float:
        reduced = np.asarray(vector, dtype=np.complex128)
        terms = (
            self.A0 @ reduced,
            complex(neff) * (self.A1 @ reduced),
            complex(neff) ** 2 * (self.A2 @ reduced),
        )
        denominator = sum(float(np.linalg.norm(term)) for term in terms)
        residual = float(np.linalg.norm(sum(terms)))
        return residual if denominator == 0.0 else residual / denominator

    def divergence_residual(self, full_vector: ArrayLike, neff: complex) -> float:
        transverse, longitudinal = self.split_full(full_vector)
        rows = self.gauss_test_dofs
        transverse_term = self.gauss_transverse[rows] @ transverse
        longitudinal_term = complex(neff) * (
            self.gauss_longitudinal[rows] @ longitudinal
        )
        denominator = (
            float(sparse_norm(self.gauss_transverse[rows]))
            * float(np.linalg.norm(transverse))
            + abs(complex(neff))
            * float(sparse_norm(self.gauss_longitudinal[rows]))
            * float(np.linalg.norm(longitudinal))
        )
        residual = float(np.linalg.norm(transverse_term + longitudinal_term))
        return residual if denominator == 0.0 else residual / denominator

    def relative_hermiticity_errors(self) -> tuple[float, float, float]:
        errors: list[float] = []
        for matrix in (self.A0, self.A1, self.A2):
            denominator = float(sparse_norm(matrix))
            numerator = float(sparse_norm(matrix - matrix.getH()))
            errors.append(numerator if denominator == 0.0 else numerator / denominator)
        return tuple(errors)  # type: ignore[return-value]


def assemble_mode_system_2d(
    mesh: MeshTri,
    *,
    frequency: float,
    k0: float,
    material_at: MaterialEvaluator,
    boundary: str = "pec",
    quadrature_order: int = 4,
    element_order: int = 1,
    pec_facets: ArrayLike | None = None,
    impedance_boundaries: Sequence[ImpedanceBoundaryInput] | None = None,
) -> ModeFEMSystem2D:
    """Assemble the dimensionless full-vector propagation-constant pencil.

    ``impedance_boundaries`` is a sequence of ``(facet_indices, Zs)`` pairs,
    where each impedance is in ohms.  Supplied facets must be boundary facets,
    must not overlap each other, and must not also be constrained as PEC.
    When ``boundary='pec'`` and ``pec_facets`` is omitted, impedance facets
    automatically replace the default PEC condition on those exterior facets.
    """

    if not isinstance(mesh, MeshTri):
        raise TypeError("mesh must be a skfem.MeshTri instance.")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ConfigurationError("frequency must be finite and positive.")
    if not np.isfinite(k0) or k0 <= 0.0:
        raise ConfigurationError("k0 must be finite and positive.")
    if boundary not in ("pec", "pmc"):
        raise ConfigurationError("boundary must be 'pec' or 'pmc'.")
    if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order or quadrature_order < 2:
        raise ConfigurationError("quadrature_order must be an integer of at least two.")
    if isinstance(element_order, (bool, np.bool_)) or element_order not in (1, 2):
        raise ConfigurationError("element_order must be 1 or 2.")
    quadrature_order = max(int(quadrature_order), 2 * int(element_order))
    transverse_element = ElementTriN1() if element_order == 1 else ElementTriN2()
    scalar_element = ElementTriP1() if element_order == 1 else ElementTriP2()

    normalized_impedance = _impedance_boundary_data(mesh, impedance_boundaries)
    impedance_facets = (
        np.unique(
            np.concatenate([facets for facets, _ in normalized_impedance])
        ).astype(np.int64, copy=False)
        if normalized_impedance
        else np.empty(0, dtype=np.int64)
    )

    computational_mesh = mesh.scaled(float(k0))
    basis = Basis(
        computational_mesh,
        transverse_element * scalar_element,
        intorder=int(quadrature_order),
    )
    if not isinstance(basis.elem, ElementComposite):  # pragma: no cover - defensive
        raise RuntimeError("Expected a composite Nedelec/P1 basis.")

    coordinates = basis.global_coordinates()
    epsilon, mu = evaluate_material(
        material_at,
        coordinates[0] / k0,
        coordinates[1] / k0,
    )
    inv_mu = 1.0 / mu
    matrices = list(
        asm(form, basis, epsilon=epsilon, inv_mu=inv_mu).astype(
            np.complex128, copy=False
        )
        for form in (_a0_form, _a1_form, _a2_form)
    )

    for facets, impedance in normalized_impedance:
        facet_basis = FacetBasis(
            computational_mesh,
            basis.elem,
            facets=facets,
            intorder=int(quadrature_order),
        )
        coefficient = 1j * ETA_0 / impedance
        if not np.isfinite(coefficient):
            raise ConfigurationError(
                "surface impedance is too small to form a finite FEM boundary coefficient; "
                "use PEC for the zero-impedance limit."
            )
        matrices[0] = matrices[0] + asm(
            _surface_impedance_form,
            facet_basis,
            surface_coefficient=coefficient,
        ).astype(np.complex128, copy=False)

    if pec_facets is not None:
        normalized_pec = _facet_indices(mesh, pec_facets, "pec_facets")
        if np.intersect1d(normalized_pec, impedance_facets, assume_unique=True).size:
            raise ConfigurationError(
                "PEC and impedance boundary facet sets must be disjoint."
            )
        constrained = np.asarray(
            basis.get_dofs(facets=normalized_pec).all(),
            dtype=np.int64,
        )
    elif boundary == "pec":
        default_pec = np.setdiff1d(
            np.asarray(mesh.boundary_facets(), dtype=np.int64),
            impedance_facets,
            assume_unique=False,
        )
        constrained = np.asarray(
            basis.get_dofs(facets=default_pec).all(), dtype=np.int64
        )
    else:
        constrained = np.empty(0, dtype=np.int64)
    free = np.setdiff1d(
        np.arange(basis.N, dtype=np.int64),
        constrained,
        assume_unique=False,
    )
    if free.size < 3:
        raise ConfigurationError("The boundary constraints leave too few FEM degrees of freedom.")
    reduced = tuple(matrix[free][:, free].tocsr() for matrix in matrices)

    split_indices = basis.split_indices()
    transverse_indices = np.asarray(split_indices[0], dtype=np.int64)
    longitudinal_indices = np.asarray(split_indices[1], dtype=np.int64)

    transverse_basis = Basis(
        computational_mesh, transverse_element, intorder=int(quadrature_order)
    )
    scalar_basis = Basis(
        computational_mesh, scalar_element, intorder=int(quadrature_order)
    )
    scalar_coordinates = scalar_basis.global_coordinates()
    gauss_epsilon, _ = evaluate_material(
        material_at,
        scalar_coordinates[0] / k0,
        scalar_coordinates[1] / k0,
    )
    divergence_transverse = asm(
        _gauss_transverse,
        transverse_basis,
        scalar_basis,
        epsilon=gauss_epsilon,
    ).astype(np.complex128, copy=False)
    divergence_longitudinal = asm(
        _gauss_longitudinal,
        scalar_basis,
        scalar_basis,
        epsilon=gauss_epsilon,
    ).astype(np.complex128, copy=False)
    scalar_boundary = np.asarray(scalar_basis.get_dofs().all(), dtype=np.int64)
    gauss_test_dofs = np.setdiff1d(
        np.arange(scalar_basis.N, dtype=np.int64),
        scalar_boundary,
        assume_unique=False,
    )

    return ModeFEMSystem2D(
        basis=basis,
        physical_mesh=mesh,
        computational_mesh=computational_mesh,
        A0=reduced[0],
        A1=reduced[1],
        A2=reduced[2],
        free_dofs=free,
        full_size=int(basis.N),
        transverse_indices=transverse_indices,
        longitudinal_indices=longitudinal_indices,
        gauss_transverse=divergence_transverse.tocsr(),
        gauss_longitudinal=divergence_longitudinal.tocsr(),
        gauss_test_dofs=gauss_test_dofs,
        frequency=float(frequency),
        k0=float(k0),
        boundary=boundary,
        material_at=material_at,
        quadrature_order=int(quadrature_order),
        impedance_boundaries=normalized_impedance,
    )


def linearized_pencil(system: ModeFEMSystem2D) -> tuple[csc_matrix, csc_matrix]:
    """Return a first companion linearization of ``system``."""

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


def solve_qep_candidates(
    system: ModeFEMSystem2D,
    *,
    target: complex,
    candidate_count: int,
    tolerance: float,
    dense_linearization_limit: int = 700,
) -> tuple[ComplexArray, ComplexArray, str]:
    """Return finite candidate roots/vectors nearest ``target``."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive.")
    n = system.ndofs
    size = 2 * n

    if size <= dense_linearization_limit or size <= 3:
        left, right = linearized_pencil(system)
        try:
            homogeneous, eigenvectors = linalg.eig(
                left.toarray(),
                right.toarray(),
                right=True,
                check_finite=False,
                homogeneous_eigvals=True,
            )
        except np.linalg.LinAlgError:
            # Singular A2 blocks create infinite generalized roots.  LAPACK's
            # homogeneous QZ normally separates them, but some lossy/PML
            # pencils can still fail to converge.  The explicit shifted
            # operator below is the numerically appropriate fallback because
            # it never attempts to resolve roots at infinity.
            pass
        else:
            alpha, denominator = homogeneous
            scale = np.maximum(np.abs(alpha), np.abs(denominator))
            finite = np.abs(denominator) > 256.0 * np.finfo(float).eps * np.maximum(
                scale, 1.0
            )
            values = np.asarray(alpha[finite] / denominator[finite], dtype=np.complex128)
            vectors = np.asarray(eigenvectors[:n, finite], dtype=np.complex128)
            order = np.argsort(np.abs(values - target))
            keep = order[: min(order.size, max(candidate_count, 3 * candidate_count))]
            return values[keep], vectors[:, keep], "dense-qz"

    requested = min(max(2 * candidate_count, 8), size - 2)
    shift = complex(target)
    perturbation = 1e-6j * max(1.0, abs(shift))
    shifted = shift + perturbation
    try:
        factor = splu(csc_matrix(system.polynomial(shifted)))
    except RuntimeError as exc:
        raise SolverError(
            f"The linearized FEM pencil could not be factorized near neff={target!r}."
        ) from exc

    # Block elimination applies (L - s R)^-1 R using a factorization of
    # Q(s), with n rows instead of the 2n-row companion pencil.  It also
    # works when A2 is singular; no inverse of the mass block is needed.
    coupling = system.A1 + shifted * system.A2

    def apply_shift_invert(vector: ComplexArray) -> ComplexArray:
        first = factor.solve(-(coupling @ vector[:n] + system.A2 @ vector[n:]))
        return np.concatenate((first, vector[:n] + shifted * first))

    operator = LinearOperator(
        (size, size), matvec=apply_shift_invert, dtype=np.complex128
    )
    rng = np.random.default_rng(20260828)
    initial = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    try:
        theta, eigenvectors = eigs(
            operator,
            k=requested,
            which="LM",
            v0=initial,
            tol=tolerance,
            maxiter=max(3000, 20 * size),
        )
    except ArpackNoConvergence as exc:
        if exc.eigenvalues is None or exc.eigenvectors is None:
            raise SolverError("Sparse FEM mode iteration did not converge.") from exc
        theta = exc.eigenvalues
        eigenvectors = exc.eigenvectors
    valid = np.abs(theta) > np.finfo(float).eps
    values = np.asarray(shifted + 1.0 / theta[valid], dtype=np.complex128)
    vectors = np.asarray(eigenvectors[:n, valid], dtype=np.complex128)
    order = np.argsort(np.abs(values - target))
    return values[order], vectors[:, order], "sparse-shift-invert"


__all__ = [
    "ImpedanceBoundaryInput",
    "MaterialEvaluator",
    "ModeFEMSystem2D",
    "assemble_mode_system_2d",
    "evaluate_material",
    "linearized_pencil",
    "solve_qep_candidates",
]
