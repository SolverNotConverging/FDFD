"""Scattered-field equivalent-source assembly.

For unchanged permeability, the relative-unit volume equation is

``L_actual E_sc = k0**2 (eps_actual - eps_background) E_inc``.

The material difference is evaluated directly at quadrature points so its
support is exactly the user-defined perturbation rather than a projected or
smoothed approximation.  Removing part of a background PEC sheet is a
boundary perturbation instead.  Its load is assembled from the two one-sided
incident magnetic tractions on the released mesh facets.  Inserting a finite
actual-only PEC sheet prescribes the scattered tangential trace as the
negative incident trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skfem import BilinearForm, InteriorFacetBasis, LinearForm, asm, condense, solve

from .constants import ETA_0
from .exceptions import ConfigurationError
from .fem import (
    ConstitutiveCoefficient,
    MixedFEMSystem,
    MixedFieldSolution,
    assemble_load_vector,
    evaluate_diagonal_coefficient,
    solve_prescribed_pec,
)
from .operators import electric_field_vector


IncidentField: TypeAlias = Callable[
    [NDArray[np.floating], NDArray[np.floating]], object
]


def _incident_values(
    incident: IncidentField,
    x: NDArray[np.floating],
    z: NDArray[np.floating],
) -> NDArray[np.complex128]:
    values = np.asarray(incident(x, z), dtype=np.complex128)
    target = (3, *x.shape)
    try:
        if values.shape == (3,):
            values = np.broadcast_to(
                values.reshape((3,) + (1,) * x.ndim), target
            )
        elif values.shape == (*x.shape, 3):
            values = np.moveaxis(values, -1, 0)
        else:
            values = np.broadcast_to(values, target)
    except ValueError as exc:
        raise ValueError(
            "incident field must return three components in (x, y, z) order; "
            f"received {values.shape}, expected compatibility with {target}."
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError("incident field contains a non-finite value.")
    return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class EquivalentSource:
    """Assembled RHS and quadrature support diagnostics."""

    load: NDArray[np.complex128]
    active_quadrature_fraction: float
    maximum_delta_eps: float
    released_pec_facet_count: int = 0
    inserted_pec_facet_count: int = 0

    @property
    def is_zero(self) -> bool:
        return not np.any(self.load) and self.inserted_pec_facet_count == 0


@dataclass(frozen=True, slots=True)
class ScatteredFieldSolution:
    """Low-level scattered FEM field and the source that produced it."""

    field: MixedFieldSolution
    source: EquivalentSource


@LinearForm(dtype=np.complex128)
def _released_pec_form(vt: object, vy: object, w: object) -> object:
    test_field = electric_field_vector(vt, vy)
    return np.sum(np.conj(test_field) * w.traction_values, axis=0)


@BilinearForm(dtype=np.complex128)
def _tangential_trace_mass(u: object, v: object, w: object) -> object:
    tangent_u = -w.n[1] * u[0] + w.n[0] * u[1]
    tangent_v = -w.n[1] * v[0] + w.n[0] * v[1]
    return np.conj(tangent_v) * tangent_u


@LinearForm(dtype=np.complex128)
def _tangential_trace_load(v: object, w: object) -> object:
    tangent_v = -w.n[1] * v[0] + w.n[0] * v[1]
    return np.conj(tangent_v) * w.trace_values


def _validated_interior_facets(
    system: MixedFEMSystem,
    facets: ArrayLike,
) -> NDArray[np.int64]:
    raw = np.asarray(facets)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError("released_pec_facets must be a one-dimensional integer array.")
    result = np.unique(np.asarray(raw, dtype=np.int64))
    if np.any(result < 0) or np.any(result >= system.basis.mesh.nfacets):
        raise ValueError("released_pec_facets contains an out-of-range facet index.")
    if result.size and np.any(system.basis.mesh.f2t[1, result] < 0):
        raise ValueError("released_pec_facets must contain interior mesh facets only.")
    if np.intersect1d(result, system.internal_pec_facets).size:
        raise ValueError(
            "A released PEC facet cannot also be constrained as an actual PEC facet."
        )
    return result


def _validated_inserted_pec_facets(
    system: MixedFEMSystem,
    facets: ArrayLike,
) -> NDArray[np.int64]:
    """Return inserted facets that are registered as actual PEC constraints."""

    raw = np.asarray(facets)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError("inserted_pec_facets must be a one-dimensional integer array.")
    result = np.unique(np.asarray(raw, dtype=np.int64))
    if np.any(result < 0) or np.any(result >= system.basis.mesh.nfacets):
        raise ValueError("inserted_pec_facets contains an out-of-range facet index.")
    if result.size and np.any(system.basis.mesh.f2t[1, result] < 0):
        raise ValueError("inserted_pec_facets must contain interior mesh facets only.")
    if np.setdiff1d(result, system.internal_pec_facets, assume_unique=True).size:
        raise ValueError(
            "Every inserted PEC facet must also be registered as an actual PEC constraint."
        )
    return result


def assemble_inserted_pec_boundary_values(
    system: MixedFEMSystem,
    *,
    inserted_pec_facets: ArrayLike,
    incident: IncidentField,
) -> NDArray[np.complex128]:
    """Project ``-E_inc,t`` onto finite actual-only PEC trace DOFs.

    The returned full mixed-space vector is zero away from the inserted PEC
    facets.  On those facets it supplies the scattered-field essential data
    required by ``E_total,t = E_inc,t + E_sc,t = 0``.  The Nedelec and scalar
    traces are projected separately on the facet set, so values away from the
    plate and the Nedelec normal component cannot affect the prescribed data.
    Composite-space DOF ordering is handled by scikit-fem rather than assumed
    by position.
    """

    if not callable(incident):
        raise ValueError("incident must be callable.")
    facets = _validated_inserted_pec_facets(system, inserted_pec_facets)
    boundary_values = np.zeros(system.ndofs, dtype=np.complex128)
    if not facets.size:
        return boundary_values

    component_bases = system.basis.split_bases()
    component_indices = system.basis.split_indices()

    def physical_incident(points: NDArray[np.floating]) -> NDArray[np.complex128]:
        return _incident_values(
            incident,
            system.length_scale * points[0],
            system.length_scale * points[1],
        )

    transverse_facet_basis = InteriorFacetBasis(
        system.basis.mesh,
        component_bases[0].elem,
        facets=facets,
        side=0,
        # A volume Basis carries a triangular quadrature rule.  FacetBasis
        # instead needs a one-dimensional rule on the reference edge; passing
        # the volume rule is accepted by scikit-fem but gives biased points and
        # half-scaled edge weights.
        intorder=system.quadrature_order,
    )
    transverse_coordinates = np.asarray(
        transverse_facet_basis.global_coordinates(), dtype=float
    )
    transverse_incident = physical_incident(transverse_coordinates)
    normal = np.asarray(transverse_facet_basis.normals, dtype=float)
    incident_tangential_trace = -(
        -normal[1] * transverse_incident[0]
        + normal[0] * transverse_incident[2]
    )
    transverse_dofs = np.asarray(
        component_bases[0].get_dofs(facets=facets).all(), dtype=np.int64
    )
    transverse = solve(
        *condense(
            asm(_tangential_trace_mass, transverse_facet_basis),
            asm(
                _tangential_trace_load,
                transverse_facet_basis,
                trace_values=incident_tangential_trace,
            ),
            I=transverse_dofs,
        )
    )
    invariant_facet_basis = InteriorFacetBasis(
        system.basis.mesh,
        component_bases[1].elem,
        facets=facets,
        side=0,
        intorder=system.quadrature_order,
    )
    invariant = invariant_facet_basis.project(
        lambda points: -physical_incident(points)[1],
        dtype=np.complex128,
    )
    projected = np.zeros(system.ndofs, dtype=np.complex128)
    projected[component_indices[0]] = np.asarray(
        transverse, dtype=np.complex128
    )
    projected[component_indices[1]] = np.asarray(invariant, dtype=np.complex128)
    constrained = np.asarray(
        system.basis.get_dofs(facets=facets).all(), dtype=np.int64
    )
    boundary_values[constrained] = projected[constrained]
    return boundary_values


def assemble_released_pec_source(
    system: MixedFEMSystem,
    *,
    released_pec_facets: ArrayLike,
    incident_magnetic: IncidentField,
) -> NDArray[np.complex128]:
    """Assemble the aperture load created by removing background PEC facets.

    The background guide is split into the two subdomains adjacent to each
    zero-thickness PEC sheet.  On a released facet, the scattered-field weak
    load is the negative sum of their incident reactions,

    ``-i*k0*L*ETA_0 * integral(conj(V) . (H_inc x n)) ds``.

    Here ``n`` is the outward normal of each adjacent element, ``L`` is the
    system length scale, and both one-sided magnetic traces are evaluated by
    probing infinitesimally inside their respective elements.  The loop below
    already sums the natural weak reactions from both adjacent sides; no
    additional image-equivalence multiplier is applied.  Consequently a PEC
    slot can scatter even when actual and background permittivities are
    identical everywhere.
    """

    if not callable(incident_magnetic):
        raise ValueError("incident_magnetic must be callable.")
    facets = _validated_interior_facets(system, released_pec_facets)
    load = np.zeros(system.ndofs, dtype=np.complex128)
    if not facets.size:
        return load

    for side in (0, 1):
        facet_basis = InteriorFacetBasis(
            system.basis.mesh,
            system.basis.elem,
            facets=facets,
            side=side,
            intorder=system.quadrature_order,
        )
        coordinates = np.asarray(facet_basis.global_coordinates(), dtype=float)
        # scikit-fem stores the side-0 outward orientation for both interior
        # traces.  Reverse it for side 1 to obtain that element's outward n.
        normal = np.asarray(facet_basis.normals, dtype=float)
        if side == 1:
            normal = -normal
        element_size = np.asarray(facet_basis.mesh_parameters(), dtype=float)
        roundoff = 256.0 * np.finfo(float).eps * np.maximum(
            1.0, np.max(np.abs(coordinates), axis=0)
        )
        trace_offset = np.maximum(1e-9 * element_size, roundoff)
        probe = coordinates - normal * trace_offset[np.newaxis, ...]
        magnetic = _incident_values(
            incident_magnetic,
            system.length_scale * probe[0],
            system.length_scale * probe[1],
        )
        nx, nz = normal[0], normal[1]
        hx, hy, hz = magnetic
        magnetic_cross_normal = np.asarray(
            (hy * nz, hz * nx - hx * nz, -hy * nx),
            dtype=np.complex128,
        )
        traction = (
            1j
            * system.dimensionless_k0
            * ETA_0
            * magnetic_cross_normal
        )
        load += np.asarray(
            asm(
                _released_pec_form,
                facet_basis,
                traction_values=traction,
            ),
            dtype=np.complex128,
        )
    return load


def assemble_equivalent_source(
    system: MixedFEMSystem,
    *,
    eps_background: ConstitutiveCoefficient,
    mu_background: ConstitutiveCoefficient = 1.0,
    incident: IncidentField,
    released_pec_facets: ArrayLike = (),
    inserted_pec_facets: ArrayLike = (),
    incident_magnetic: IncidentField | None = None,
) -> EquivalentSource:
    """Assemble volume contrast and record both PEC perturbation classes.

    ``incident`` supplies electric fields for the volume term.  When
    ``released_pec_facets`` is nonempty, ``incident_magnetic`` must supply the
    corresponding one-sided-capable magnetic field callback. Inserted facets
    are counted here; their essential data are projected by
    :func:`assemble_inserted_pec_boundary_values` during the field solve.
    """

    coordinates = system.physical_coordinates()
    x, z = coordinates[0], coordinates[1]
    actual = evaluate_diagonal_coefficient(
        system.parameters.eps_r, x, z, name="actual eps_r"
    )
    background = evaluate_diagonal_coefficient(
        eps_background, x, z, name="background eps_r"
    )
    actual_mu = evaluate_diagonal_coefficient(
        system.parameters.mu_r, x, z, name="actual mu_r"
    )
    background_mu = evaluate_diagonal_coefficient(
        mu_background, x, z, name="background mu_r"
    )
    if not np.allclose(actual_mu, background_mu, rtol=1e-12, atol=1e-14):
        raise ConfigurationError(
            "The MVP scattered-field source supports permittivity "
            "perturbations only; actual and background mu_r differ."
        )
    delta = actual - background
    incident_values = _incident_values(incident, x, z)
    def source_at_quadrature(
        x_computational: NDArray[np.floating],
        z_computational: NDArray[np.floating],
    ) -> NDArray[np.complex128]:
        # The FEM basis lives on the dimensionless computational mesh; public
        # material and incident callbacks continue to receive SI metres.
        x_request = system.length_scale * x_computational
        z_request = system.length_scale * z_computational
        actual_request = evaluate_diagonal_coefficient(
            system.parameters.eps_r, x_request, z_request, name="actual eps_r"
        )
        background_request = evaluate_diagonal_coefficient(
            eps_background, x_request, z_request, name="background eps_r"
        )
        return (
            system.dimensionless_k0**2
            * (actual_request - background_request)
            * _incident_values(incident, x_request, z_request)
        )

    facets = _validated_interior_facets(system, released_pec_facets)
    inserted_facets = _validated_inserted_pec_facets(
        system, inserted_pec_facets
    )
    load = assemble_load_vector(system.basis, source_at_quadrature)
    if facets.size:
        if incident_magnetic is None:
            raise ConfigurationError(
                "incident_magnetic is required when released_pec_facets is nonempty."
            )
        load += assemble_released_pec_source(
            system,
            released_pec_facets=facets,
            incident_magnetic=incident_magnetic,
        )
    active = np.any(np.abs(delta) > 0.0, axis=0)
    return EquivalentSource(
        load=np.asarray(load, dtype=np.complex128),
        active_quadrature_fraction=float(np.count_nonzero(active) / active.size),
        maximum_delta_eps=float(np.max(np.abs(delta))),
        released_pec_facet_count=int(facets.size),
        inserted_pec_facet_count=int(inserted_facets.size),
    )


def solve_scattered_pec(
    system: MixedFEMSystem,
    *,
    eps_background: ConstitutiveCoefficient,
    mu_background: ConstitutiveCoefficient = 1.0,
    incident: IncidentField,
    released_pec_facets: ArrayLike = (),
    inserted_pec_facets: ArrayLike = (),
    incident_magnetic: IncidentField | None = None,
    residual_tolerance: float = 1e-7,
) -> ScatteredFieldSolution:
    """Solve with released-background and inserted-actual PEC perturbations."""

    source = assemble_equivalent_source(
        system,
        eps_background=eps_background,
        mu_background=mu_background,
        incident=incident,
        released_pec_facets=released_pec_facets,
        inserted_pec_facets=inserted_pec_facets,
        incident_magnetic=incident_magnetic,
    )
    boundary_values = assemble_inserted_pec_boundary_values(
        system,
        inserted_pec_facets=inserted_pec_facets,
        incident=incident,
    )
    field = solve_prescribed_pec(
        system,
        source.load,
        boundary_values=boundary_values,
        residual_tolerance=residual_tolerance,
    )
    return ScatteredFieldSolution(field=field, source=source)


__all__ = [
    "EquivalentSource",
    "IncidentField",
    "ScatteredFieldSolution",
    "assemble_equivalent_source",
    "assemble_inserted_pec_boundary_values",
    "assemble_released_pec_source",
    "solve_scattered_pec",
]
