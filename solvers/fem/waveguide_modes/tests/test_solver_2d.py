"""Focused validation of the standalone full-vector 2D FEM mode solver."""

from __future__ import annotations

import numpy as np
import pytest

from fem_waveguide_modes.constants import C_0
from fem_waveguide_modes.exceptions import (
    ConfigurationError,
    NotDiscretizedError,
)
from fem_waveguide_modes.geometry import Rectangle
from fem_waveguide_modes.materials import Material
from fem_waveguide_modes.results import ModeSet
from fem_waveguide_modes.solver_2d import ModeSolver2D


@pytest.mark.gmsh
def test_second_order_pair_improves_cutoff_on_the_same_mesh():
    expected = np.sqrt(0.75)
    errors = []
    cells = []
    for order in (1, 2):
        solver = ModeSolver2D(frequency=C_0, x_range=1.0, y_range=0.5)
        mesh = solver.mesh(resolution=(5, 3), element_order=order)
        mode = solver.solve(max_refinements=0, dense_linearization_limit=4, num_modes=1, neff_guess=expected)[0]
        errors.append(abs(mode.neff - expected))
        cells.append(mesh.elements)
        assert mode.divergence_residual < 1e-7
        assert mode.power.real == pytest.approx(1.0, rel=1e-9)
        assert mesh.info.element_order == order
    np.testing.assert_array_equal(cells[0], cells[1])
    assert errors[1] < errors[0] * 0.15


@pytest.mark.gmsh
def test_geometry_is_placed_before_explicit_discretization() -> None:
    solver = ModeSolver2D(frequency=C_0, x_range=1.0, y_range=0.5)
    assert solver.result is None
    solver.add_pmc()
    assert solver.geometry.outer_boundary == "pmc"
    solver.add_pec()
    assert solver.geometry.outer_boundary == "pec"
    region = solver.add_rectangle(epsilon=2.25, mu=1.0, x_range=(0.25, 0.75), y_range=(0.1, 0.4), name='core')

    assert region.name == "core"
    assert not solver.discretized
    with pytest.raises(NotDiscretizedError, match="discretized"):
        solver._solve_once(num_modes=1)

    mesh = solver.mesh(resolution=(6, 4))
    assert solver.discretized
    assert mesh.geometry_revision == solver.geometry.revision
    assert set(np.unique(mesh.element_tags)) == {1, 2}

    # Mutations through the public facade discard the obsolete discretization.
    solver.add_circle(epsilon=1.5, mu=1.0, center=(0.5, 0.25), r1=0.04, name='inclusion')
    assert not solver.discretized
    with pytest.raises(NotDiscretizedError):
        solver._solve_once(num_modes=1)


@pytest.mark.gmsh
def test_direct_scene_mutation_invalidates_mesh_and_result() -> None:
    solver = ModeSolver2D(frequency=C_0, x_range=1.0, y_range=0.5)
    solver.mesh(resolution=(5, 3))

    # Advanced callers may use the shared geometry object directly.  Its
    # revision counter still prevents an accidental solve on the old mesh.
    solver.geometry.add_region(
        Rectangle((0.2, 0.3), (0.2, 0.3)),
        Material(2.0, 1.0),
        name="direct_edit",
    )
    assert not solver.discretized
    assert solver.result is None
    with pytest.raises(NotDiscretizedError, match="discretized"):
        solver._solve_once(num_modes=1)


def test_numerical_controls_belong_to_solve_and_old_aliases_are_rejected() -> None:
    with pytest.raises(TypeError, match="neff_guess"):
        ModeSolver2D(frequency=C_0, x_range=1., y_range=.5, neff_guess=.8)
    solver = ModeSolver2D(frequency=C_0, x_range=1., y_range=.5)
    with pytest.raises(TypeError, match="guess"):
        solver.solve(guess=.8)


@pytest.mark.gmsh
def test_homogeneous_pec_te10_matches_the_rectangular_guide_limit() -> None:
    # a=1 m, b=0.5 m and lambda_0=1 m.  The nearest TE10 solution is
    # neff=sqrt(1-(pi/(k0*a))**2)=sqrt(3/4).
    expected = np.sqrt(0.75)
    solver = ModeSolver2D(frequency=C_0, x_range=1.0, y_range=0.5, boundary='pec')
    solver.mesh(resolution=(8, 4), quadrature_order=4)

    assert max(solver.system.relative_hermiticity_errors()) < 1e-12
    modes = solver.solve(max_refinements=0, residual_tolerance=1e-09, divergence_tolerance=1e-08, num_modes=1, neff_guess=expected)

    assert isinstance(modes, ModeSet)
    assert solver.result is modes
    assert len(modes) == 1
    mode = modes[0]
    assert mode.index == 0
    assert mode.polarization == "TE"
    assert mode.neff.real == pytest.approx(expected, rel=3e-3)
    assert abs(mode.neff.imag) < 1e-10
    assert mode.residual is not None and mode.residual < 1e-9
    assert mode.divergence_residual is not None
    assert mode.divergence_residual < 1e-8
    assert mode.normalization == "unit-power"
    assert mode.power is not None and mode.power.real == pytest.approx(1.0, rel=2e-9)

    assert mode.fields.components == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    assert mode.fields.mesh_points is not None
    assert mode.fields.mesh_cells is not None
    assert mode.component("ey").shape == mode.fields.sample_shape
    assert np.max(np.abs(mode.component("Ey"))) > 0.0

    # Legacy normalized aliases and dimensional propagation data deliberately
    # remain distinct.
    assert solver.propagation_constant is not None
    assert solver.attenuation_constant is not None
    assert solver.beta is not None
    assert solver.propagation_constant[0] == pytest.approx(mode.neff.real)
    assert solver.attenuation_constant[0] == pytest.approx(-mode.neff.imag)
    assert solver.beta[0] == pytest.approx(solver.k0 * mode.neff)
    assert mode.alpha == pytest.approx(-mode.beta.imag)


@pytest.mark.gmsh
def test_passive_bulk_loss_uses_negative_imaginary_neff() -> None:
    epsilon = 2.0 - 0.02j
    expected = np.sqrt(epsilon - 0.25)
    solver = ModeSolver2D(frequency=C_0, x_range=1.0, y_range=0.5, background_epsilon=epsilon)
    solver.mesh(resolution=(7, 4))
    mode = solver.solve(max_refinements=0, divergence_tolerance=1e-07, num_modes=1, neff_guess=expected)[0]

    assert mode.neff.real == pytest.approx(expected.real, rel=4e-3)
    assert mode.neff.imag == pytest.approx(expected.imag, rel=8e-2)
    assert mode.neff.imag < 0.0
    assert mode.attenuation_constant > 0.0
    assert mode.alpha > 0.0


@pytest.mark.gmsh
def test_passive_sibc_post_uses_negative_imaginary_neff() -> None:
    """Exercise the public geometry-to-facet SIBC path and its loss sign."""

    solver = ModeSolver2D(frequency=C_0, x_range=1.0, y_range=0.5, boundary='pec')
    post = solver.add_impedance_surface(Zs=20.0 + 20j, x_range=(0.56, 0.68), y_range=(0.11, 0.23), name='lossy_post')
    solver.mesh(resolution=(12, 6))

    assert post.impedance == 20.0 + 20.0j
    assert solver.mesh_data.boundary_facets[post.name].size > 0
    assert solver.system.impedance_boundaries[0][1] == post.impedance

    mode = solver.solve(max_refinements=0, residual_tolerance=1e-08, divergence_tolerance=1e-06, num_modes=1, neff_guess=0.82)[0]
    assert mode.neff.real > 0.0
    assert mode.neff.imag < 0.0
    assert mode.alpha > 0.0
    assert mode.residual is not None and mode.residual < 1e-8
    assert mode.divergence_residual is not None
    assert mode.divergence_residual < 1e-6


@pytest.mark.gmsh
def test_transformation_optics_pml_uses_forward_passive_branch() -> None:
    solver = ModeSolver2D(frequency=C_0, x_range=2.0, y_range=1.0)
    solver.add_pml(thickness=0.2, sigma_max=2.0)
    solver.mesh(resolution=(6, 3))
    mode = solver.solve(max_refinements=0, direction='all', residual_tolerance=1e-07, divergence_tolerance=1e-06, num_modes=1, neff_guess=0.85)[0]

    assert mode.neff.real > 0.0
    assert mode.neff.imag < 0.0
    assert mode.alpha > 0.0
    assert mode.residual is not None and mode.residual < 1e-7
    assert mode.divergence_residual is not None
    assert mode.divergence_residual < 1e-6
