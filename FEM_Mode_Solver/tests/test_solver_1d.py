"""Focused analytic and lifecycle tests for the standalone 1D FEM solver."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c

from FEM_Mode_Solver.exceptions import (
    BackendCapabilityError,
    ConfigurationError,
    MeshError,
    NotDiscretizedError,
)
from FEM_Mode_Solver.geometry import Interval
from FEM_Mode_Solver.materials import Material
from FEM_Mode_Solver.solver_1d import ModeSolver1D


def test_discretization_is_explicit_and_conforms_to_layer_interfaces() -> None:
    solver = ModeSolver1D(30e9, 15e-3, 2)
    solver.add_layer(4.0, 1.0, (3.7e-3, 11.3e-3), name="core")

    with pytest.raises(NotDiscretizedError, match="discretize"):
        solver.solve()

    mesh = solver.discretize(resolution=31)
    assert np.any(np.isclose(mesh.nodes, 3.7e-3, rtol=0.0, atol=1e-15))
    assert np.any(np.isclose(mesh.nodes, 11.3e-3, rtol=0.0, atol=1e-15))
    assert mesh.info.maximum_edge <= mesh.info.requested_maximum_edge * (1.0 + 1e-12)


def test_material_aware_mesh_is_denser_in_high_dk_layer() -> None:
    solver = ModeSolver1D(c, (0.0, 1.0), 1)
    solver.add_layer(9.0, 1.0, (0.5, 1.0), name="high_dk")

    mesh = solver.discretize(
        resolution=20,
        wavelength_elements=4,
        material_aware=True,
    )
    centres = 0.5 * (mesh.nodes[:-1] + mesh.nodes[1:])
    widths = np.diff(mesh.nodes)
    low_dk_width = float(np.median(widths[centres < 0.5]))
    high_dk_width = float(np.median(widths[centres > 0.5]))

    assert np.any(np.isclose(mesh.nodes, 0.5, rtol=0.0, atol=1e-15))
    assert low_dk_width / high_dk_width == pytest.approx(3.0, rel=0.08)


def test_material_aware_can_be_disabled_without_violating_shortest_wavelength() -> None:
    solver = ModeSolver1D(c, (0.0, 1.0), 1)
    solver.add_layer(9.0, 1.0, (0.5, 1.0))

    mesh = solver.discretize(
        resolution=20,
        wavelength_elements=4,
        material_aware=False,
    )
    widths = np.diff(mesh.nodes)

    np.testing.assert_allclose(widths, widths[0], rtol=0.0, atol=2e-15)
    assert widths.max() <= (2.0 * np.pi / solver.k0 / 3.0 / 4.0) * (1.0 + 1e-12)


def test_material_aware_mesh_grades_index_contrast_below_unity() -> None:
    solver = ModeSolver1D(
        1.0e6,
        (0.0, 1.0),
        1,
        background_epsilon=0.25,
    )
    solver.add_layer(1.0, 1.0, (0.5, 1.0), name="higher_index")
    mesh = solver.discretize(resolution=20, wavelength_elements=4)
    centres = 0.5 * (mesh.nodes[:-1] + mesh.nodes[1:])
    widths = np.diff(mesh.nodes)

    low_index_width = float(np.median(widths[centres < 0.5]))
    higher_index_width = float(np.median(widths[centres > 0.5]))
    assert low_index_width / higher_index_width == pytest.approx(2.0, rel=0.08)


def test_anisotropic_wavenumber_bound_uses_largest_epsilon_and_mu_axes() -> None:
    solver = ModeSolver1D(
        c,
        (0.0, 0.1),
        1,
        background_epsilon=(100.0, 1.0, 1.0),
        background_mu=(1.0, 100.0, 1.0),
    )
    mesh = solver.discretize(resolution=2, wavelength_elements=4)

    conservative_index = 100.0
    target = 2.0 * np.pi / solver.k0 / conservative_index / 4.0
    assert mesh.info.maximum_edge <= target * (1.0 + 1e-12)


def test_wavelength_elements_caps_size_in_homogeneous_high_dk_medium() -> None:
    solver = ModeSolver1D(
        c,
        (0.0, 1.0),
        1,
        background_epsilon=9.0,
    )
    mesh = solver.discretize(resolution=2, wavelength_elements=12)
    local_wavelength = 2.0 * np.pi / (solver.k0 * 3.0)

    assert mesh.info.maximum_edge <= local_wavelength / 12.0 * (1.0 + 1e-12)


def test_refine_remeshes_and_invalidates_previous_modes() -> None:
    solver = ModeSolver1D(20e9, 20e-3, 1)
    solver.add_layer(4.0, 1.0, (7e-3, 13e-3))
    coarse = solver.discretize(resolution=24, wavelength_elements=6)
    solver.solve()
    assert solver.solution is not None

    fine = solver.refine(factor=2.0)

    assert fine is solver.mesh
    assert fine.info.elements >= 2 * coarse.info.elements - 3
    assert fine.info.requested_maximum_edge == pytest.approx(
        0.5 * coarse.info.requested_maximum_edge
    )
    assert fine.info.maximum_edge <= fine.info.requested_maximum_edge * (1.0 + 1e-12)
    assert np.any(np.isclose(fine.nodes, 7e-3, rtol=0.0, atol=1e-15))
    assert np.any(np.isclose(fine.nodes, 13e-3, rtol=0.0, atol=1e-15))
    assert solver.solution is None
    assert solver.modes is None


def test_refine_requires_a_current_mesh_and_density_increase() -> None:
    solver = ModeSolver1D(10e9, 10e-3, 1)
    with pytest.raises(NotDiscretizedError, match="discretize"):
        solver.refine()

    solver.discretize(resolution=12)
    with pytest.raises(ConfigurationError, match="greater than one"):
        solver.refine(factor=1.0)


def test_failed_rediscretization_keeps_previous_mesh_settings_atomic() -> None:
    solver = ModeSolver1D(10e9, 10e-3, 1)
    original = solver.discretize(resolution=12, quadrature_order=4)

    with pytest.raises(MeshError):
        solver.discretize(resolution=1, quadrature_order=8)

    assert solver.mesh is original
    assert solver._quadrature_order == 4
    assert solver._discretization_settings is not None
    assert solver._discretization_settings["quadrature_order"] == 4


def test_geometry_placement_after_meshing_invalidates_the_mesh() -> None:
    solver = ModeSolver1D(25e9, 12e-3, 1)
    solver.discretize(resolution=30)
    solver.add_layer(2.25, 1.0, (4e-3, 8e-3))

    assert solver.mesh is None
    with pytest.raises(NotDiscretizedError):
        solver.solve()


def test_direct_geometry_edit_invalidates_mesh_and_solved_result() -> None:
    solver = ModeSolver1D(25e9, 12e-3, 1)
    solver.discretize(resolution=24)
    solver.solve()
    assert solver.solution is not None

    solver.geometry.add_region(
        Interval((4e-3, 8e-3)), Material(2.25, 1.0), name="direct_core"
    )

    assert solver.mesh is None
    assert solver.solution is None
    assert solver.modes is None
    with pytest.raises(NotDiscretizedError):
        solver.solve()


def test_homogeneous_pec_parallel_plate_spectrum_matches_analytic_modes() -> None:
    frequency = 20e9
    width = 20e-3
    solver = ModeSolver1D(frequency, width, 3)
    solver.discretize(resolution=100)
    modes = solver.solve(residual_tolerance=1e-9)

    k0 = 2.0 * np.pi * frequency / c
    first_order = np.sqrt(1.0 - (np.pi / (k0 * width)) ** 2)

    # PEC supports the TM/TEM constant solution plus degenerate first-order
    # TE and TM families.  Linear FEM converges quadratically in eigenvalue.
    assert modes[0].polarization == "TM"
    assert modes[0].neff == pytest.approx(1.0, abs=2e-10)
    assert {modes[1].polarization, modes[2].polarization} == {"TE", "TM"}
    assert modes[1].neff.real == pytest.approx(first_order, abs=1.5e-4)
    assert modes[2].neff.real == pytest.approx(first_order, abs=1.5e-4)
    assert all(mode.residual is not None and mode.residual < 1e-9 for mode in modes)
    assert all(mode.power == pytest.approx(1.0, abs=2e-10) for mode in modes)


def test_sparse_shift_invert_retains_degenerate_te_tm_pair() -> None:
    frequency = 20e9
    width = 20e-3
    solver = ModeSolver1D(frequency, width, 3, neff_guess=1.0)
    solver.discretize(resolution=220)
    modes = solver.solve(dense_limit=50, residual_tolerance=1e-8)

    k0 = 2.0 * np.pi * frequency / c
    first_order = np.sqrt(1.0 - (np.pi / (k0 * width)) ** 2)
    assert modes.metadata["methods"] == {
        "TE": "sparse-shift-invert",
        "TM": "sparse-shift-invert",
    }
    assert modes[0].polarization == "TM"
    assert modes[0].neff.real == pytest.approx(1.0, abs=2e-9)
    assert {modes[1].polarization, modes[2].polarization} == {"TE", "TM"}
    assert modes[1].neff.real == pytest.approx(first_order, abs=2e-4)
    assert modes[2].neff.real == pytest.approx(first_order, abs=2e-4)
    assert all(mode.residual is not None and mode.residual < 1e-8 for mode in modes)


def test_uniform_diagonal_tm_tem_mode_uses_correct_tensor_components() -> None:
    # The constant TM mode has neff^2 = eps_x * mu_y.  eps_z affects only the
    # derivative term and therefore drops out of this analytic TEM solution.
    epsilon = (4.0, 7.0, 11.0)
    mu = (2.0, 3.0, 5.0)
    expected = np.sqrt(epsilon[0] * mu[1])
    solver = ModeSolver1D(
        15e9,
        10e-3,
        1,
        neff_guess=expected,
        background_epsilon=epsilon,
        background_mu=mu,
    )
    solver.discretize(resolution=40)
    mode = solver.solve()[0]

    assert mode.polarization == "TM"
    assert mode.neff == pytest.approx(expected, abs=5e-10)
    assert np.max(np.abs(mode.component("Ez"))) < 1e-7 * np.max(
        np.abs(mode.component("Ex"))
    )


def test_pmc_outer_walls_produce_the_dual_te_tem_mode() -> None:
    epsilon = (2.0, 5.0, 7.0)
    mu = (3.0, 11.0, 13.0)
    expected = np.sqrt(epsilon[1] * mu[0])
    solver = ModeSolver1D(
        12e9,
        8e-3,
        1,
        neff_guess=expected,
        background_epsilon=epsilon,
        background_mu=mu,
    )
    solver.add_pmc()
    solver.discretize(resolution=32)
    mode = solver.solve()[0]

    assert mode.polarization == "TE"
    assert mode.neff == pytest.approx(expected, abs=5e-10)
    assert np.max(np.abs(mode.component("Hz"))) < 1e-7 * np.max(
        np.abs(mode.component("Hx"))
    )


def test_passive_loss_has_negative_neff_imaginary_part_for_public_convention() -> None:
    epsilon = 4.0 - 0.08j
    expected = np.sqrt(epsilon)
    solver = ModeSolver1D(
        18e9,
        10e-3,
        1,
        neff_guess=expected,
        background_epsilon=epsilon,
    )
    solver.discretize(resolution=36)
    mode = solver.solve()[0]

    assert mode.polarization == "TM"
    assert mode.neff == pytest.approx(expected, rel=2e-10, abs=2e-10)
    assert mode.neff.imag < 0.0
    assert mode.beta.imag < 0.0
    assert mode.alpha > 0.0


def test_pml_mode_solve_falls_back_cleanly_if_dense_ggev_is_unstable() -> None:
    solver = ModeSolver1D(
        c,
        (-2.0, 2.0),
        2,
        neff_guess=1.8,
    )
    solver.add_layer(4.0, 1.0, (-0.25, 0.25))
    solver.add_pml(0.6, sigma_max=4.0)
    solver.discretize(resolution=80)
    modes = solver.solve(residual_tolerance=1e-7)

    assert len(modes) == 2
    assert all(np.isfinite((mode.neff.real, mode.neff.imag)).all() for mode in modes)
    assert all(mode.neff.imag <= 1e-10 for mode in modes)
    assert all(mode.residual is not None and mode.residual < 1e-7 for mode in modes)
    assert set(modes.metadata["methods"].values()) <= {
        "dense-generalized",
        "sparse-shift-invert",
    }


def test_active_constitutive_sign_is_rejected_with_convention_hint() -> None:
    with pytest.raises(ConfigurationError, match="non-positive imaginary"):
        ModeSolver1D(10e9, 5e-3, 1, background_epsilon=2.0 + 0.01j)


def test_impedance_geometry_is_explicitly_rejected_until_robin_form_exists() -> None:
    solver = ModeSolver1D(10e9, 10e-3, 1)
    solver.add_impedance_surface(50.0, x_range=(4e-3, 6e-3))
    solver.discretize(resolution=40)

    with pytest.raises(BackendCapabilityError, match="impedance-surface"):
        solver.solve()
