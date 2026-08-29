"""Conforming mixed finite-element assembly for 2.5D Maxwell problems.

This module implements the Stage-A discretization

``(E_x, E_z) in H(curl; Omega)`` and ``E_y in H1(Omega)``

using the lowest-order triangular Nedelec and continuous P1 elements.  The
phasor convention is ``exp(-i omega t)`` and the prescribed invariant-direction
dependence is ``exp(i k_y y)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, TypeAlias
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, norm as sparse_norm
from skfem import (
    Basis,
    BilinearForm,
    LinearForm,
    MeshTri,
    asm,
    condense,
    solve,
)
from skfem.element import ElementComposite, ElementTriN1, ElementTriP1

from .operators import electric_field_vector, modified_curl
from .exceptions import SolverError

if TYPE_CHECKING:
    from .materials import Material


RealOrComplex: TypeAlias = float | complex | np.number
ConstitutiveCallback: TypeAlias = Callable[
    [NDArray[np.floating], NDArray[np.floating]], object
]
ConstitutiveCoefficient: TypeAlias = (
    RealOrComplex
    | tuple[RealOrComplex, RealOrComplex, RealOrComplex]
    | NDArray[np.generic]
    | ConstitutiveCallback
)
VectorSourceCallback: TypeAlias = Callable[
    [NDArray[np.floating], NDArray[np.floating]], object
]
VectorSource: TypeAlias = NDArray[np.generic] | VectorSourceCallback


@dataclass(frozen=True, slots=True)
class MaxwellParameters:
    """Physical parameters for one relative-unit Maxwell assembly.

    ``eps_r`` and ``mu_r`` may each be a scalar, a three-entry diagonal in
    physical ``(x, y, z)`` order, a quadrature-compatible array, or a callback
    ``coefficient(x, z)`` returning any of those forms.  Full off-diagonal
    tensors are intentionally outside this Stage-A implementation.
    """

    k0: float
    ky: complex = 0.0
    eps_r: ConstitutiveCoefficient = 1.0
    mu_r: ConstitutiveCoefficient = 1.0

    def __post_init__(self) -> None:
        k0 = float(self.k0)
        ky = complex(self.ky)
        if not np.isfinite(k0) or k0 <= 0.0:
            raise ValueError("k0 must be a positive finite real number.")
        if not np.isfinite(ky.real) or not np.isfinite(ky.imag):
            raise ValueError("ky must be finite.")
        if ky.imag != 0.0:
            raise ValueError(
                "Complex ky is not yet validated by the 2.5D sesquilinear "
                "form; use a real ky."
            )
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "ky", ky)

    @classmethod
    def from_material(
        cls,
        *,
        k0: float,
        material: Material,
        ky: complex = 0.0,
    ) -> MaxwellParameters:
        """Construct parameters from :mod:`wavefem.materials` material data."""

        from .materials import as_diagonal_material

        diagonal = as_diagonal_material(material)
        return cls(
            k0=k0,
            ky=ky,
            eps_r=diagonal.eps_r.as_array(),
            mu_r=diagonal.mu_r.as_array(),
        )


@dataclass(frozen=True, slots=True)
class MixedFEMSystem:
    """Assembled sparse Maxwell system and its mixed basis."""

    basis: Basis
    matrix: csr_matrix
    parameters: MaxwellParameters
    physical_mesh: MeshTri
    length_scale: float = 1.0
    internal_pec_facets: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )

    def __post_init__(self) -> None:
        raw = np.asarray(self.internal_pec_facets)
        if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
            raise ValueError("internal_pec_facets must be a one-dimensional integer array.")
        facets = np.unique(np.asarray(raw, dtype=np.int64))
        if np.any(facets < 0) or np.any(facets >= self.basis.mesh.nfacets):
            raise ValueError("internal_pec_facets contains an out-of-range facet index.")
        if facets.size and np.any(self.basis.mesh.f2t[1, facets] < 0):
            raise ValueError("internal_pec_facets must contain interior mesh facets only.")
        object.__setattr__(self, "internal_pec_facets", facets)

    @property
    def ndofs(self) -> int:
        """Number of mixed finite-element degrees of freedom."""

        return int(self.basis.N)

    @property
    def pec_dofs(self) -> NDArray[np.integer]:
        """Outer and internal DOFs imposing zero tangential electric field."""

        outer = np.asarray(self.basis.get_dofs().all(), dtype=np.int64)
        if not self.internal_pec_facets.size:
            return outer
        internal = np.asarray(
            self.basis.get_dofs(facets=self.internal_pec_facets).all(),
            dtype=np.int64,
        )
        return np.union1d(outer, internal)

    @property
    def dimensionless_k0(self) -> float:
        """Vacuum wavenumber used on the scaled computational mesh."""

        return self.parameters.k0 * self.length_scale

    @property
    def dimensionless_ky(self) -> float:
        """Prescribed y-wavenumber used on the scaled computational mesh."""

        return float(self.parameters.ky.real * self.length_scale)

    def physical_coordinates(self) -> NDArray[np.float64]:
        """Return cell-basis quadrature coordinates in metres."""

        return np.asarray(
            self.length_scale * self.basis.global_coordinates(), dtype=float
        )


@dataclass(frozen=True, slots=True)
class MixedFieldSolution:
    """Coefficient vector together with the basis needed to interpret it."""

    basis: Basis
    coefficients: NDArray[np.complex128]
    solve_info: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.complex128)
        if coefficients.shape != (self.basis.N,) or not np.isfinite(coefficients).all():
            raise SolverError("The FEM solution contains invalid or non-finite coefficients.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "solve_info",
            MappingProxyType(dict(self.solve_info or {})),
        )

    def split_coefficients(
        self,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Return ``(E_t coefficients, E_y coefficients)`` safely.

        Composite-space DOFs are grouped by topological type and must not be
        split by assuming contiguous element blocks.
        """

        split = self.basis.split(self.coefficients)
        return split[0][0], split[1][0]

    def interpolate(self) -> tuple[object, object]:
        """Interpolate and return the quadrature fields ``(E_t, E_y)``."""

        fields = self.basis.interpolate(self.coefficients)
        if not isinstance(fields, tuple) or len(fields) != 2:
            raise RuntimeError("Expected an H(curl)-H1 composite field.")
        return fields


def create_mixed_basis(
    mesh: MeshTri,
    *,
    intorder: int = 4,
) -> Basis:
    """Create the conforming ``ElementTriN1 * ElementTriP1`` basis."""

    if intorder < 1:
        raise ValueError("intorder must be at least one.")
    return Basis(mesh, ElementTriN1() * ElementTriP1(), intorder=intorder)


def evaluate_diagonal_coefficient(
    coefficient: ConstitutiveCoefficient,
    x: NDArray[np.floating],
    z: NDArray[np.floating],
    *,
    name: str = "coefficient",
) -> NDArray[np.complex128]:
    """Evaluate a scalar/diagonal coefficient at quadrature points.

    The returned shape is always ``(3, nelements, nquadrature)`` in physical
    ``(x, y, z)`` order.  A callback may return a scalar, ``x.shape`` for an
    isotropic spatial field, ``(3,)``, ``(3, *x.shape)``, or ``(*x.shape, 3)``.
    """

    if x.shape != z.shape:
        raise ValueError("x and z quadrature arrays must have identical shapes.")
    raw = coefficient(x, z) if callable(coefficient) else coefficient
    values = np.asarray(raw, dtype=np.complex128)
    qshape = x.shape
    target = (3, *qshape)

    try:
        if values.ndim == 0:
            diagonal = np.broadcast_to(values, target)
        elif values.shape == qshape:
            diagonal = np.broadcast_to(values[np.newaxis, ...], target)
        elif values.shape == (3,):
            diagonal = np.broadcast_to(
                values.reshape((3,) + (1,) * len(qshape)), target
            )
        elif values.shape == target:
            diagonal = values
        elif values.shape == (*qshape, 3):
            diagonal = np.moveaxis(values, -1, 0)
        elif values.shape[0] in (1, 3):
            diagonal = np.broadcast_to(values, target)
        else:
            isotropic = np.broadcast_to(values, qshape)
            diagonal = np.broadcast_to(isotropic[np.newaxis, ...], target)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be scalar or diagonal in (x, y, z) order; "
            f"received shape {values.shape}, expected compatibility with {target}."
        ) from exc

    result = np.array(diagonal, dtype=np.complex128, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value.")
    return result


def _evaluate_vector_source(
    source: VectorSource,
    x: NDArray[np.floating],
    z: NDArray[np.floating],
) -> NDArray[np.complex128]:
    raw = source(x, z) if callable(source) else source
    values = np.asarray(raw, dtype=np.complex128)
    qshape = x.shape
    target = (3, *qshape)
    try:
        if values.shape == (3,):
            result = np.broadcast_to(
                values.reshape((3,) + (1,) * len(qshape)), target
            )
        elif values.shape == target:
            result = values
        elif values.shape == (*qshape, 3):
            result = np.moveaxis(values, -1, 0)
        else:
            result = np.broadcast_to(values, target)
    except ValueError as exc:
        raise ValueError(
            "source must have three physical components in (x, y, z) order; "
            f"received shape {values.shape}, expected compatibility with {target}."
        ) from exc
    result = np.array(result, dtype=np.complex128, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("source contains a non-finite value.")
    return result


def _validate_mixed_basis(basis: Basis) -> None:
    element = basis.elem
    if not isinstance(element, ElementComposite) or len(element.elems) != 2:
        raise TypeError("Expected an ElementTriN1 * ElementTriP1 composite basis.")
    if not isinstance(element.elems[0], ElementTriN1) or not isinstance(
        element.elems[1], ElementTriP1
    ):
        raise TypeError("Expected element order ElementTriN1 * ElementTriP1.")


@BilinearForm(dtype=np.complex128)
def _maxwell_form(et: object, ey: object, vt: object, vy: object, w: object) -> object:
    curl_e = modified_curl(et, ey, w.ky)
    curl_v = modified_curl(vt, vy, w.ky)
    field_e = electric_field_vector(et, ey)
    field_v = electric_field_vector(vt, vy)

    curl_term = np.sum(
        np.conj(curl_v) * w.inv_mu_r_diagonal * curl_e,
        axis=0,
    )
    mass_term = np.sum(
        np.conj(field_v) * w.eps_r_diagonal * field_e,
        axis=0,
    )
    return curl_term - w.k0_sq * mass_term


@LinearForm(dtype=np.complex128)
def _source_form(vt: object, vy: object, w: object) -> object:
    test_field = electric_field_vector(vt, vy)
    return np.sum(np.conj(test_field) * w.source_values, axis=0)


def assemble_maxwell_matrix(
    basis: Basis,
    parameters: MaxwellParameters,
) -> csr_matrix:
    """Assemble the sparse complex 2.5D Maxwell matrix on ``basis``."""

    _validate_mixed_basis(basis)
    coordinates = basis.global_coordinates()
    eps_diagonal = evaluate_diagonal_coefficient(
        parameters.eps_r,
        coordinates[0],
        coordinates[1],
        name="eps_r",
    )
    mu_diagonal = evaluate_diagonal_coefficient(
        parameters.mu_r,
        coordinates[0],
        coordinates[1],
        name="mu_r",
    )
    if np.any(np.abs(mu_diagonal) == 0.0):
        raise ValueError("mu_r must be nonzero at every quadrature point.")

    matrix = asm(
        _maxwell_form,
        basis,
        ky=parameters.ky,
        k0_sq=parameters.k0**2,
        eps_r_diagonal=eps_diagonal,
        inv_mu_r_diagonal=1.0 / mu_diagonal,
    )
    return matrix.astype(np.complex128, copy=False)


def assemble_mixed_system(
    mesh: MeshTri,
    parameters: MaxwellParameters,
    *,
    intorder: int = 4,
    length_scale: float = 1.0,
    internal_pec_facets: ArrayLike = (),
) -> MixedFEMSystem:
    """Create the mixed basis and assemble its sparse Maxwell matrix.

    ``length_scale`` is the number of physical metres represented by one
    computational coordinate unit.  Supplying (for example) ``1 / k0``
    nondimensionalizes micron-scale meshes, keeping Nedelec and H1 blocks
    comparably scaled without changing the public SI material callbacks.

    ``internal_pec_facets`` identifies conforming interior facets on which
    both mixed-space tangential electric traces are constrained.  The normal
    electric component remains free, as required for surface charge on an
    infinitesimally thin PEC sheet.
    """

    length_scale = float(length_scale)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive.")
    computational_mesh = (
        mesh if length_scale == 1.0 else mesh.scaled(float(1.0 / length_scale))
    )

    def scaled_coefficient(
        coefficient: ConstitutiveCoefficient,
    ) -> ConstitutiveCoefficient:
        if not callable(coefficient):
            return coefficient

        def evaluate(
            x: NDArray[np.floating], z: NDArray[np.floating]
        ) -> object:
            return coefficient(length_scale * x, length_scale * z)

        return evaluate

    assembly_parameters = MaxwellParameters(
        k0=parameters.k0 * length_scale,
        ky=parameters.ky * length_scale,
        eps_r=scaled_coefficient(parameters.eps_r),
        mu_r=scaled_coefficient(parameters.mu_r),
    )
    basis = create_mixed_basis(computational_mesh, intorder=intorder)
    return MixedFEMSystem(
        basis=basis,
        matrix=assemble_maxwell_matrix(basis, assembly_parameters),
        parameters=parameters,
        physical_mesh=mesh,
        length_scale=length_scale,
        internal_pec_facets=np.asarray(internal_pec_facets),
    )


def assemble_load_vector(
    basis: Basis,
    source: VectorSource,
) -> NDArray[np.complex128]:
    """Assemble ``integral(conj(V) . source)`` in physical component order."""

    _validate_mixed_basis(basis)
    coordinates = basis.global_coordinates()
    source_values = _evaluate_vector_source(
        source,
        coordinates[0],
        coordinates[1],
    )
    return np.asarray(
        asm(_source_form, basis, source_values=source_values),
        dtype=np.complex128,
    )


def solve_homogeneous_pec(
    system: MixedFEMSystem,
    load: NDArray[np.generic],
    *,
    residual_tolerance: float = 1e-7,
) -> MixedFieldSolution:
    """Solve with homogeneous PEC conditions on outer and registered facets."""

    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive.")
    rhs = np.asarray(load, dtype=np.complex128)
    if rhs.shape != (system.ndofs,):
        raise ValueError(
            f"load must have shape ({system.ndofs},), received {rhs.shape}."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            coefficients = solve(
                *condense(system.matrix, rhs, D=system.pec_dofs)
            )
    except (MatrixRankWarning, RuntimeError, ValueError) as exc:
        raise SolverError(
            "The PEC Maxwell system could not be solved; it may be singular "
            "or too close to a closed-domain resonance."
        ) from exc
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if not np.isfinite(coefficients).all():
        raise SolverError("The PEC Maxwell solve returned non-finite coefficients.")
    free = np.setdiff1d(
        np.arange(system.ndofs, dtype=np.int64),
        np.asarray(system.pec_dofs, dtype=np.int64),
        assume_unique=False,
    )
    residual = system.matrix @ coefficients - rhs
    denominator = np.linalg.norm(rhs[free])
    relative_residual = float(
        np.linalg.norm(residual[free]) / denominator
        if denominator > 0.0
        else np.linalg.norm(residual[free])
    )
    if not np.isfinite(relative_residual) or relative_residual > residual_tolerance:
        raise SolverError(
            f"The PEC Maxwell linear residual is too large ({relative_residual:.3e}; "
            f"tolerance {residual_tolerance:.3e})."
        )
    return MixedFieldSolution(
        basis=system.basis,
        coefficients=coefficients,
        solve_info={"method": "scipy-direct", "relative_residual": relative_residual},
    )


def relative_hermiticity_error(matrix: csr_matrix) -> float:
    """Return ``||A - A^H|| / ||A||`` for a sparse matrix."""

    denominator = float(sparse_norm(matrix))
    numerator = float(sparse_norm(matrix - matrix.getH()))
    if denominator == 0.0:
        return numerator
    return numerator / denominator
