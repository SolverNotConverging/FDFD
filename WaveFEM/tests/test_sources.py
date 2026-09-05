import numpy as np
import pytest
from skfem import MeshTri

from wavefem.constants import ETA_0
from wavefem.exceptions import ConfigurationError
from wavefem.fem import MaxwellParameters, assemble_mixed_system
from wavefem.sources import (
    assemble_equivalent_source,
    assemble_inserted_pec_boundary_values,
    assemble_released_pec_source,
    solve_scattered_pec,
)


def incident(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(x, dtype=complex)
    return np.asarray((zeros, np.sin(np.pi * x) * np.sin(np.pi * z), zeros))


def test_identical_actual_and_background_material_has_exactly_zero_source() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    system = assemble_mixed_system(mesh, MaxwellParameters(k0=2.0, ky=0.4, eps_r=2.0))
    source = assemble_equivalent_source(system, eps_background=2.0, incident=incident)
    assert source.is_zero
    assert source.active_quadrature_fraction == 0.0
    assert source.maximum_delta_eps == 0.0
    assert source.released_pec_facet_count == 0


def test_released_pec_source_uses_two_one_sided_magnetic_traces() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.5))
    system = assemble_mixed_system(mesh, MaxwellParameters(k0=2.0, eps_r=2.0))

    def one_sided_h(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(x, dtype=np.complex128)
        return np.asarray((zeros, np.where(x < 0.5, 1.0, 0.0), zeros))

    aperture = assemble_released_pec_source(
        system,
        released_pec_facets=facets,
        incident_magnetic=one_sided_h,
    )
    assert np.linalg.norm(aperture) > 0.0

    source = assemble_equivalent_source(
        system,
        eps_background=2.0,
        incident=incident,
        released_pec_facets=facets,
        incident_magnetic=one_sided_h,
    )
    assert not source.is_zero
    assert source.active_quadrature_fraction == 0.0
    assert source.maximum_delta_eps == 0.0
    assert source.released_pec_facet_count == facets.size

    def continuous_h(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(x, dtype=np.complex128)
        return np.asarray((zeros, np.ones_like(x), zeros))

    cancelling = assemble_released_pec_source(
        system,
        released_pec_facets=facets,
        incident_magnetic=continuous_h,
    )
    assert np.linalg.norm(cancelling) < 1e-12 * np.linalg.norm(aperture)


def test_released_pec_constant_traction_has_exact_edge_moments() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.5))
    parameters = MaxwellParameters(k0=2.0, eps_r=2.0)
    system = assemble_mixed_system(mesh, parameters, intorder=4)

    def one_sided_constant_h(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(x, dtype=np.complex128)
        return np.asarray((zeros, np.where(x < 0.5, 1.0, 0.0), zeros))

    aperture = assemble_released_pec_source(
        system,
        released_pec_facets=facets,
        incident_magnetic=one_sided_constant_h,
    )
    transverse_indices = system.basis.split_indices()[0]
    transverse_facet_dofs = system.basis.split_bases()[0].get_dofs(
        facets=facets
    ).all()
    mixed_facet_dofs = transverse_indices[transverse_facet_dofs]

    # A first-order Nedelec edge basis has unit tangential line moment.  The
    # constant one-sided traction therefore produces exactly the physical
    # prefactor on every released edge.  A triangular volume quadrature rule
    # accidentally reused on the facet would return half this value.
    expected = system.dimensionless_k0 * ETA_0
    np.testing.assert_allclose(
        np.abs(aperture[mixed_facet_dofs]),
        expected,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.delete(aperture, mixed_facet_dofs),
        0.0,
        atol=1e-13,
    )


def test_released_and_actual_pec_facets_must_be_disjoint() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.5))
    system = assemble_mixed_system(
        mesh,
        MaxwellParameters(k0=2.0),
        internal_pec_facets=facets,
    )
    with pytest.raises(ValueError, match="cannot also be constrained"):
        assemble_released_pec_source(
            system,
            released_pec_facets=facets[:1],
            incident_magnetic=lambda x, z: np.ones((3, *x.shape)),
        )


@pytest.mark.parametrize("element_order", [1, 2])
def test_inserted_pec_prescribes_negative_incident_tangential_trace(element_order) -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.5))
    system = assemble_mixed_system(
        mesh,
        MaxwellParameters(k0=1.7, ky=0.3, eps_r=2.0),
        internal_pec_facets=facets,
        element_order=element_order,
    )

    def affine_incident(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.asarray(
            (
                0.2 - 0.1 * z,
                0.4 + 0.2 * x - 0.1 * z,
                -0.3 + 0.1 * x,
            ),
            dtype=np.complex128,
        )

    boundary_values = assemble_inserted_pec_boundary_values(
        system,
        inserted_pec_facets=facets,
        incident=affine_incident,
    )
    component_bases = system.basis.split_bases()
    component_indices = system.basis.split_indices()
    incident_projection = np.zeros(system.ndofs, dtype=np.complex128)
    incident_projection[component_indices[0]] = component_bases[0].project(
        lambda points: affine_incident(points[0], points[1])[[0, 2]],
        dtype=np.complex128,
    )
    incident_projection[component_indices[1]] = component_bases[1].project(
        lambda points: affine_incident(points[0], points[1])[1],
        dtype=np.complex128,
    )
    inserted_dofs = np.asarray(
        system.basis.get_dofs(facets=facets).all(), dtype=np.int64
    )
    np.testing.assert_allclose(
        boundary_values[inserted_dofs],
        -incident_projection[inserted_dofs],
    )
    assert not np.any(
        boundary_values[
            np.setdiff1d(np.arange(system.ndofs), inserted_dofs)
        ]
    )

    solution = solve_scattered_pec(
        system,
        eps_background=2.0,
        incident=affine_incident,
        inserted_pec_facets=facets,
    )
    np.testing.assert_allclose(
        solution.field.coefficients[inserted_dofs],
        boundary_values[inserted_dofs],
    )
    assert solution.source.inserted_pec_facet_count == facets.size
    assert not solution.source.is_zero
    assert np.linalg.norm(solution.field.coefficients) > 0.0


def test_inserted_pec_trace_is_independent_of_nontrace_incident_field() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 7))
    facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.4))
    system = assemble_mixed_system(
        mesh,
        MaxwellParameters(k0=1.7, ky=0.3, eps_r=2.0),
        internal_pec_facets=facets,
    )

    def nonpolynomial_trace(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.asarray(
            (
                np.cos(1.7 * z),
                np.exp(0.4 * z),
                np.sin(2.3 * z) + 0.2 * z**2,
            ),
            dtype=np.complex128,
        )

    def same_tangential_trace_different_elsewhere(
        x: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        off_facet = (x - 0.4) ** 2
        return nonpolynomial_trace(x, z) + 50.0 * np.asarray(
            (
                2.0 + np.sin(z),
                off_facet * np.cos(1.3 * z),
                off_facet * np.exp(0.2 * z),
            ),
            dtype=np.complex128,
        )

    reference = assemble_inserted_pec_boundary_values(
        system,
        inserted_pec_facets=facets,
        incident=nonpolynomial_trace,
    )
    perturbed = assemble_inserted_pec_boundary_values(
        system,
        inserted_pec_facets=facets,
        incident=same_tangential_trace_different_elsewhere,
    )

    inserted_dofs = system.basis.get_dofs(facets=facets).all()
    assert np.linalg.norm(reference[inserted_dofs]) > 0.0
    np.testing.assert_allclose(
        perturbed[inserted_dofs],
        reference[inserted_dofs],
        rtol=1e-12,
        atol=1e-12,
    )


def test_inserted_pec_nedelec_trace_uses_exact_line_quadrature() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.5))
    system = assemble_mixed_system(
        mesh,
        MaxwellParameters(k0=1.7, ky=0.3, eps_r=2.0),
        intorder=4,
        internal_pec_facets=facets,
    )

    def quadratic_tangential_trace(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(x, dtype=np.complex128)
        return np.asarray((zeros, zeros, z**2))

    boundary_values = assemble_inserted_pec_boundary_values(
        system,
        inserted_pec_facets=facets,
        incident=quadratic_tangential_trace,
    )
    transverse_indices = system.basis.split_indices()[0]
    transverse_facet_dofs = system.basis.split_bases()[0].get_dofs(
        facets=facets
    ).all()
    mixed_facet_dofs = transverse_indices[transverse_facet_dofs]
    facet_points = mesh.p[:, mesh.facets[:, facets]]
    z_lower = np.min(facet_points[1], axis=0)
    z_upper = np.max(facet_points[1], axis=0)
    exact_edge_moments = (z_upper**3 - z_lower**3) / 3.0

    # Sign depends only on global edge orientation; the magnitude is the exact
    # line integral of the prescribed tangential trace on each edge.
    np.testing.assert_allclose(
        np.abs(boundary_values[mixed_facet_dofs]),
        exact_edge_moments,
        rtol=1e-13,
        atol=1e-13,
    )


def test_inserted_pec_facets_must_be_registered_actual_constraints() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.5))
    system = assemble_mixed_system(mesh, MaxwellParameters(k0=2.0))

    with pytest.raises(ValueError, match="registered as an actual PEC constraint"):
        assemble_inserted_pec_boundary_values(
            system,
            inserted_pec_facets=facets,
            incident=incident,
        )


def _solve(delta: float) -> tuple[float, float]:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 13), np.linspace(0.0, 1.0, 13))

    def eps_actual(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        perturbation = (x > 0.35) & (x < 0.65) & (z > 0.35) & (z < 0.65)
        return 2.0 + delta * perturbation

    system = assemble_mixed_system(
        mesh,
        MaxwellParameters(k0=1.7, ky=0.3, eps_r=eps_actual),
        intorder=4,
    )
    solution = solve_scattered_pec(
        system, eps_background=2.0, incident=incident
    )
    et, ey = solution.field.interpolate()
    field_norm = float(
        np.sqrt(
            np.sum(
                system.basis.dx
                * (np.abs(et[0]) ** 2 + np.abs(ey) ** 2 + np.abs(et[1]) ** 2)
            )
        )
    )
    return field_norm, field_norm**2


@pytest.mark.slow
def test_weak_perturbation_field_and_squared_norm_scaling() -> None:
    field_1, squared_norm_1 = _solve(1e-4)
    field_2, squared_norm_2 = _solve(2e-4)
    assert field_2 / field_1 == pytest.approx(2.0, rel=5e-3)
    assert squared_norm_2 / squared_norm_1 == pytest.approx(4.0, rel=1e-2)


def test_permeability_perturbation_is_rejected_explicitly() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    system = assemble_mixed_system(
        mesh, MaxwellParameters(k0=2.0, ky=0.2, eps_r=2.0, mu_r=1.1)
    )
    with pytest.raises(ConfigurationError, match="permittivity perturbations only"):
        assemble_equivalent_source(
            system,
            eps_background=2.0,
            mu_background=1.0,
            incident=incident,
        )
