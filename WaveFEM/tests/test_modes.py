from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import norm as sparse_norm

from wavefem.constants import ETA_0
from wavefem.exceptions import ConfigurationError
from wavefem.materials import Material
from wavefem.modes import CrossSection, ModeSolver, PECBoundary


WAVELENGTH = 1.0
WIDTH = 0.2
EPS_R = 2.25
MU_R = 1.0
ETA = 0.35
K0 = 2.0 * np.pi / WAVELENGTH
KY = ETA * K0
EXPECTED_NEFF = np.sqrt(EPS_R * MU_R - ETA**2)


def homogeneous_solver(ky: float) -> ModeSolver:
    cross_section = CrossSection(
        x_span=(-WIDTH / 2.0, WIDTH / 2.0),
        background=Material(eps_r=EPS_R, mu_r=MU_R),
        boundary="pec",
    )
    return ModeSolver(
        cross_section,
        wavelength=WAVELENGTH,
        ky=ky,
        num_elements=12,
        dense_linearization_limit=256,
    )


def test_homogeneous_pec_tem_mode_is_full_vector_and_unit_power() -> None:
    modes = homogeneous_solver(KY).solve(
        num_modes=1,
        neff_guess=EXPECTED_NEFF,
        residual_tolerance=1e-9,
        divergence_tolerance=1e-9,
    )
    mode = modes[0]

    assert mode.neff == pytest.approx(EXPECTED_NEFF, rel=2e-10, abs=2e-10)
    assert mode.beta == pytest.approx(K0 * EXPECTED_NEFF, rel=2e-10)
    assert mode.direction == "forward"
    assert mode.classification == "propagating"
    assert mode.normalization == "unit-power"
    assert mode.power == pytest.approx(1.0, rel=2e-10, abs=2e-10)
    assert mode.residual < 1e-10
    assert mode.divergence_residual < 1e-10

    reference = np.mean(mode.E_x)
    assert reference.real > 0.0
    np.testing.assert_allclose(mode.E_x, reference, rtol=2e-9, atol=2e-9 * abs(reference))
    np.testing.assert_allclose(mode.E_y, 0.0, atol=2e-9 * abs(reference))
    np.testing.assert_allclose(mode.E_z, 0.0, atol=2e-9 * abs(reference))
    np.testing.assert_allclose(mode.H_x, 0.0, atol=2e-9 * abs(reference) / ETA_0)
    np.testing.assert_allclose(
        mode.H_y,
        EXPECTED_NEFF * mode.E_x / (ETA_0 * MU_R),
        rtol=2e-9,
        atol=2e-9 * abs(reference) / ETA_0,
    )
    np.testing.assert_allclose(
        mode.H_z,
        -ETA * mode.E_x / (ETA_0 * MU_R),
        rtol=2e-9,
        atol=2e-9 * abs(reference) / ETA_0,
    )

    hermiticity = modes.system.relative_hermiticity_errors()
    np.testing.assert_allclose(hermiticity, 0.0, atol=2e-14)


def test_ky_sign_symmetry_holds_for_pencil_and_mode_fields() -> None:
    positive_solver = homogeneous_solver(KY)
    negative_solver = homogeneous_solver(-KY)
    positive_system = positive_solver.assemble()
    negative_system = negative_solver.assemble()

    nx = positive_system.ex_slice.stop - positive_system.ex_slice.start
    nt = positive_system.ey_slice.stop - positive_system.ey_slice.start
    full_sign = np.concatenate((np.ones(nx), -np.ones(nt), np.ones(nt)))
    sign = diags(full_sign[positive_system.free_dofs], format="csr")
    transformed = sign @ positive_system.polynomial(EXPECTED_NEFF) @ sign
    denominator = float(sparse_norm(negative_system.polynomial(EXPECTED_NEFF)))
    symmetry_error = float(
        sparse_norm(negative_system.polynomial(EXPECTED_NEFF) - transformed)
        / denominator
    )
    assert symmetry_error < 2e-14

    positive = positive_solver.solve(
        num_modes=1,
        neff_guess=EXPECTED_NEFF,
        residual_tolerance=1e-9,
        divergence_tolerance=1e-9,
    )[0]
    negative = negative_solver.solve(
        num_modes=1,
        neff_guess=EXPECTED_NEFF,
        residual_tolerance=1e-9,
        divergence_tolerance=1e-9,
    )[0]

    assert negative.neff == pytest.approx(positive.neff, rel=2e-10, abs=2e-10)
    assert negative.power == pytest.approx(positive.power, rel=2e-10)
    np.testing.assert_allclose(negative.E_x, positive.E_x, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(negative.E_y, -positive.E_y, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(negative.E_z, positive.E_z, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(negative.H_x, -positive.H_x, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(negative.H_y, positive.H_y, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(negative.H_z, -positive.H_z, rtol=2e-9, atol=2e-9)


def test_nontrivial_parallel_plate_modes_converge_quadratically() -> None:
    # For a homogeneous PEC interval of width a, both n=1 polarizations have
    # beta^2/k0^2 = eps*mu - eta^2 - (pi/(k0*a))^2.  Unlike the exact TEM
    # regression above, these modes exercise every derivative/coupling block.
    width = 1.0
    cutoff = np.pi / (K0 * width)
    expected = np.sqrt(EPS_R * MU_R - ETA**2 - cutoff**2)
    cross_section = CrossSection(
        x_span=(-width / 2.0, width / 2.0),
        background=Material(eps_r=EPS_R, mu_r=MU_R),
        boundary="pec",
    )

    errors: list[float] = []
    for elements in (8, 16, 32):
        modes = ModeSolver(
            cross_section,
            wavelength=WAVELENGTH,
            ky=KY,
            num_elements=elements,
            dense_linearization_limit=400,
        ).solve(
            num_modes=2,
            neff_guess=expected,
            residual_tolerance=1e-9,
            divergence_tolerance=1e-9,
        )
        assert all(mode.power == pytest.approx(1.0, rel=2e-9) for mode in modes)
        np.testing.assert_allclose(
            [mode.neff.real for mode in modes],
            modes[0].neff.real,
            rtol=2e-10,
            atol=2e-10,
        )
        errors.append(abs(modes[0].neff - expected))

    assert errors[1] < errors[0] / 3.8
    assert errors[2] < errors[1] / 3.8


def test_passive_loss_uses_positive_imaginary_beta_for_minus_iwt() -> None:
    eps_r = 2.25 + 0.09j
    expected = np.sqrt(eps_r - ETA**2)
    cross_section = CrossSection(
        x_span=(-WIDTH / 2.0, WIDTH / 2.0),
        background=Material(eps_r=eps_r),
        boundary="pec",
    )
    mode = ModeSolver(
        cross_section,
        wavelength=WAVELENGTH,
        ky=KY,
        num_elements=10,
        dense_linearization_limit=256,
    ).solve(num_modes=1, neff_guess=expected)[0]

    assert mode.neff == pytest.approx(expected, rel=2e-9, abs=2e-9)
    assert mode.neff.imag > 0.0
    assert mode.direction == "forward"
    assert mode.power == pytest.approx(1.0, rel=2e-9)


def test_internal_pec_is_a_conforming_named_cross_section_interface() -> None:
    cross_section = CrossSection(
        x_span=(-0.5, 0.5),
        background=Material(eps_r=2.25),
        boundary="pec",
        pec_boundaries=[PECBoundary(0.137, "ground")],
    )

    assert cross_section.pec_boundaries == [PECBoundary(0.137, "ground")]
    assert 0.137 in cross_section.interfaces
    system = ModeSolver(
        cross_section,
        wavelength=1.0,
        num_elements=8,
        dense_linearization_limit=256,
    ).assemble()
    np.testing.assert_allclose(
        system.x_nodes[np.argmin(np.abs(system.x_nodes - 0.137))],
        0.137,
        rtol=0.0,
        atol=2e-15,
    )


@pytest.mark.parametrize("outer_boundary", ("pec", "pmc"))
def test_internal_pec_removes_only_tangential_node_dofs(
    outer_boundary: str,
) -> None:
    cross_section = CrossSection(
        x_span=(-0.5, 0.5),
        background=Material(eps_r=2.25),
        boundary=outer_boundary,
    )
    cross_section.add_pec(x=0.0, name="ground")
    system = ModeSolver(
        cross_section,
        wavelength=1.0,
        num_elements=10,
        dense_linearization_limit=256,
    ).assemble()

    nx = system.ex_slice.stop - system.ex_slice.start
    nt = system.ey_slice.stop - system.ey_slice.start
    pec_node = int(np.flatnonzero(system.x_nodes == 0.0)[0])
    free = set(system.free_dofs.tolist())

    # Every P0 E_x unknown remains free: E_x is normal to the sheet and has
    # independent left/right cell traces.  The two nodal tangential fields are
    # exactly constrained at the PEC coordinate.
    assert set(range(nx)) <= free
    assert nx + pec_node not in free
    assert nx + nt + pec_node not in free
    assert pec_node not in system.divergence_test_dofs

    expanded = system.expand(np.ones(system.ndofs, dtype=np.complex128))
    np.testing.assert_allclose(expanded[system.ex_slice], 1.0)
    assert expanded[system.ey_slice][pec_node] == 0.0
    assert expanded[system.ez_slice][pec_node] == 0.0

    if outer_boundary == "pec":
        assert nx not in free
        assert nx + nt - 1 not in free
        assert nx + nt not in free
        assert nx + 2 * nt - 1 not in free
    else:
        assert nx in free
        assert nx + nt - 1 in free
        assert nx + nt in free
        assert nx + 2 * nt - 1 in free


def test_internal_pec_coordinates_and_names_are_validated() -> None:
    cross_section = CrossSection(
        x_span=(-0.5, 0.5),
        background=Material(),
        boundary="pec",
    )
    boundary = cross_section.add_pec(x=0.0, name="ground")
    assert boundary == PECBoundary(0.0, "ground")

    with pytest.raises(ConfigurationError, match="strictly inside"):
        cross_section.add_pec(x=-0.5)
    with pytest.raises(ConfigurationError, match="finite real"):
        cross_section.add_pec(x=np.inf)
    with pytest.raises(ConfigurationError, match="already exists"):
        cross_section.add_pec(x=0.0, name="second_ground")
    with pytest.raises(ConfigurationError, match="not unique"):
        cross_section.add_pec(x=0.2, name="ground")
