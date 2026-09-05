"""Low-level tests for the 2D Leontovich boundary weak form."""

from __future__ import annotations

import numpy as np
import pytest
from skfem import BilinearForm, FacetBasis, MeshTri, asm

from fem_waveguide_modes.assembly import assemble_mode_system_2d
from fem_waveguide_modes.constants import C_0, ETA_0
from fem_waveguide_modes.exceptions import ConfigurationError


def _vacuum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    del y
    return np.ones(x.shape, dtype=np.complex128), np.ones(
        x.shape, dtype=np.complex128
    )


def _assemble(
    mesh: MeshTri,
    *,
    boundary: str = "pmc",
    pec_facets: np.ndarray | None = None,
    impedance_boundaries: object = None,
    element_order: int = 1,
):
    return assemble_mode_system_2d(
        mesh,
        frequency=C_0,
        k0=2.0 * np.pi,
        material_at=_vacuum,
        boundary=boundary,
        quadrature_order=4,
        element_order=element_order,
        pec_facets=pec_facets,
        impedance_boundaries=impedance_boundaries,  # type: ignore[arg-type]
    )


@BilinearForm(dtype=np.complex128)
def _unit_tangential_trace(
    et: object,
    ez: object,
    vt: object,
    vz: object,
    w: object,
) -> object:
    """Independent unit-coefficient reference for the requested trace Gramian."""

    trial_tangent = -w.n[1] * et[0] + w.n[0] * et[1]
    test_tangent = -w.n[1] * vt[0] + w.n[0] * vt[1]
    return trial_tangent * np.conj(test_tangent) + ez * np.conj(vz)


@pytest.mark.parametrize("element_order", [1, 2])
def test_sibc_adds_exact_positive_j_eta0_over_zs_trace_term_to_a0(element_order) -> None:
    mesh = MeshTri.init_sqsymmetric()
    facets = np.asarray(mesh.boundary_facets()[:3], dtype=np.int64)
    impedance = 50.0
    baseline = _assemble(mesh, element_order=element_order)
    system = _assemble(mesh, impedance_boundaries=[(facets, impedance)], element_order=element_order)

    facet_basis = FacetBasis(
        system.computational_mesh,
        system.basis.elem,
        facets=facets,
        intorder=4,
    )
    unit_gramian = asm(_unit_tangential_trace, facet_basis).astype(
        np.complex128, copy=False
    )
    expected = (1j * ETA_0 / impedance) * unit_gramian
    difference = system.A0 - baseline.A0

    np.testing.assert_allclose(
        difference.toarray(), expected.toarray(), rtol=2e-13, atol=2e-13
    )
    assert (system.A1 - baseline.A1).nnz == 0
    assert (system.A2 - baseline.A2).nnz == 0
    assert system.impedance_boundaries[0][1] == impedance
    np.testing.assert_array_equal(system.impedance_boundaries[0][0], facets)


def test_real_positive_surface_resistance_has_dissipative_passivity_sign() -> None:
    mesh = MeshTri.init_sqsymmetric()
    facets = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    baseline = _assemble(mesh)
    system = _assemble(mesh, impedance_boundaries=[(facets, 75.0)])
    difference = (system.A0 - baseline.A0).toarray()

    # With +jwt and a positive real Zs, v^H DeltaA0 v is positive imaginary.
    # Perturbing A0 + neff*A1 + neff^2*A2 then moves a forward neff into the
    # lower half-plane, i.e. Im(neff) < 0 and positive attenuation.
    vector = np.arange(1, system.ndofs + 1, dtype=float)
    vector = vector + 0.37j * vector[::-1]
    quadratic_form = np.vdot(vector, difference @ vector)
    assert quadratic_form.imag > 0.0
    assert abs(quadratic_form.real) < 1e-12 * abs(quadratic_form.imag)

    gramian = difference / (1j * ETA_0 / 75.0)
    np.testing.assert_allclose(gramian, gramian.conj().T, atol=2e-13)
    assert np.linalg.eigvalsh(gramian).min() >= -2e-12


def test_impedance_scaling_dimensions_and_finite_matrices() -> None:
    mesh = MeshTri.init_sqsymmetric().refined(1)
    facets = np.asarray(mesh.boundary_facets()[::2], dtype=np.int64)
    baseline = _assemble(mesh)
    low = _assemble(mesh, impedance_boundaries=[(facets, 40.0)])
    high = _assemble(mesh, impedance_boundaries=[(facets, 80.0)])

    low_delta = low.A0 - baseline.A0
    high_delta = high.A0 - baseline.A0
    np.testing.assert_allclose(
        high_delta.toarray(), 0.5 * low_delta.toarray(), rtol=2e-13, atol=2e-13
    )
    assert low.A0.shape == low.A1.shape == low.A2.shape == (low.ndofs, low.ndofs)
    assert low.full_size == baseline.full_size
    for matrix in (low.A0, low.A1, low.A2):
        assert np.isfinite(matrix.data).all()


def test_impedance_facets_replace_default_outer_pec_facets() -> None:
    mesh = MeshTri.init_sqsymmetric()
    boundary_facets = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    impedance_facets = boundary_facets[:2]

    all_pec = _assemble(mesh, boundary="pec")
    mixed = _assemble(
        mesh,
        boundary="pec",
        impedance_boundaries=[(impedance_facets, 50.0)],
    )

    assert mixed.ndofs > all_pec.ndofs
    assert np.intersect1d(
        mixed.free_dofs,
        mixed.basis.get_dofs(facets=impedance_facets).all(),
    ).size > 0


def test_impedance_boundary_validation_rejects_ambiguous_or_invalid_data() -> None:
    mesh = MeshTri.init_sqsymmetric()
    boundary_facets = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    first = boundary_facets[:2]
    interior = np.setdiff1d(
        np.arange(mesh.nfacets, dtype=np.int64), boundary_facets
    )[0]

    with pytest.raises(ConfigurationError, match="nonzero"):
        _assemble(mesh, impedance_boundaries=[(first, 0.0)])
    with pytest.raises(ConfigurationError, match="nonnegative"):
        _assemble(mesh, impedance_boundaries=[(first, -1.0)])
    with pytest.raises(ConfigurationError, match="scalar"):
        _assemble(mesh, impedance_boundaries=[(first, True)])
    with pytest.raises(ConfigurationError, match="scalar"):
        _assemble(mesh, impedance_boundaries=[(first, np.asarray([50.0]))])
    with pytest.raises(ConfigurationError, match="too small"):
        _assemble(
            mesh,
            impedance_boundaries=[(first, np.nextafter(0.0, 1.0))],
        )
    with pytest.raises(ConfigurationError, match="boundary facets only"):
        _assemble(mesh, impedance_boundaries=[(np.asarray([interior]), 50.0)])
    with pytest.raises(ConfigurationError, match="more than one"):
        _assemble(
            mesh,
            impedance_boundaries=[(first, 50.0), (first[1:], 75.0)],
        )
    with pytest.raises(ConfigurationError, match="disjoint"):
        _assemble(
            mesh,
            boundary="pec",
            pec_facets=first,
            impedance_boundaries=[(first, 50.0)],
        )
