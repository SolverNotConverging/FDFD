"""Focused validation of the standalone full-vector 2D FEM mode solver."""

from __future__ import annotations

import numpy as np
import pytest

from FEM_Mode_Solver.constants import C_0
from FEM_Mode_Solver.exceptions import (
    ConfigurationError,
    NotDiscretizedError,
)
from FEM_Mode_Solver.geometry import Rectangle
from FEM_Mode_Solver.materials import Material
from FEM_Mode_Solver.results import ModeSet
from FEM_Mode_Solver.solver_2d import ModeSolver2D


@pytest.mark.gmsh
def test_second_order_pair_improves_cutoff_on_the_same_mesh():
    expected = np.sqrt(0.75)
    errors = []
    cells = []
    for order in (1, 2):
        solver = ModeSolver2D(C_0, 1.0, 0.5, num_modes=1, guess=expected)
        mesh = solver.discretize(resolution=(5, 3), element_order=order)
        mode = solver.solve(max_refinements=0, dense_linearization_limit=4)[0]
        errors.append(abs(mode.neff - expected))
        cells.append(mesh.elements)
        assert mode.divergence_residual < 1e-7
        assert mode.power.real == pytest.approx(1.0, rel=1e-9)
        assert mesh.info.element_order == order
    np.testing.assert_array_equal(cells[0], cells[1])
    assert errors[1] < errors[0] * 0.15


@pytest.mark.gmsh
def test_geometry_is_placed_before_explicit_discretization() -> None:
    solver = ModeSolver2D(C_0, 1.0, 0.5, num_modes=1)
    assert solver.solution is None
    solver.add_pmc()
    assert solver.geometry.outer_boundary == "pmc"
    solver.add_pec()
    assert solver.geometry.outer_boundary == "pec"
    region = solver.add_rectangle(
        2.25,
        1.0,
        (0.25, 0.75),
        (0.10, 0.40),
        name="core",
        subpixels=8,
    )

    assert region.name == "core"
    assert not solver.discretized
    with pytest.raises(NotDiscretizedError, match="discretized"):
        solver._solve_once()

    mesh = solver.discretize(resolution=(6, 4))
    assert solver.discretized
    assert mesh.geometry_revision == solver.geometry.revision
    assert set(np.unique(mesh.element_tags)) == {1, 2}

    # Mutations through the public facade discard the obsolete discretization.
    solver.add_circle(1.5, 1.0, (0.5, 0.25), 0.04, name="inclusion")
    assert not solver.discretized
    with pytest.raises(NotDiscretizedError):
        solver._solve_once()


@pytest.mark.gmsh
def test_direct_scene_mutation_invalidates_mesh_and_result() -> None:
    solver = ModeSolver2D(C_0, 1.0, 0.5, num_modes=1)
    solver.discretize(resolution=(5, 3))

    # Advanced callers may use the shared geometry object directly.  Its
    # revision counter still prevents an accidental solve on the old mesh.
    solver.geometry.add_region(
        Rectangle((0.2, 0.3), (0.2, 0.3)),
        Material(2.0, 1.0),
        name="direct_edit",
    )
    assert not solver.discretized
    assert solver.solution is None
    with pytest.raises(NotDiscretizedError, match="discretized"):
        solver._solve_once()


def test_constructor_accepts_neff_guess_and_rejects_two_guess_names() -> None:
    solver = ModeSolver2D(C_0, 1.0, 0.5, num_modes=1, neff_guess=0.8)
    assert solver.neff_guess == 0.8
    assert solver.guess == solver.neff_guess

    alias_solver = ModeSolver2D(C_0, 1.0, 0.5, num_modes=1, guess=0.7)
    assert alias_solver.neff_guess == 0.7

    with pytest.raises(ConfigurationError, match="only one"):
        ModeSolver2D(
            C_0,
            1.0,
            0.5,
            num_modes=1,
            neff_guess=0.8,
            guess=0.7,
        )


@pytest.mark.gmsh
def test_homogeneous_pec_te10_matches_the_rectangular_guide_limit() -> None:
    # a=1 m, b=0.5 m and lambda_0=1 m.  The nearest TE10 solution is
    # neff=sqrt(1-(pi/(k0*a))**2)=sqrt(3/4).
    expected = np.sqrt(0.75)
    solver = ModeSolver2D(
        C_0,
        1.0,
        0.5,
        num_modes=1,
        guess=expected,
        boundary="pec",
    )
    solver.discretize(resolution=(8, 4), quadrature_order=4)

    assert max(solver.system.relative_hermiticity_errors()) < 1e-12
    modes = solver.solve(
        max_refinements=0,
        residual_tolerance=1e-9,
        divergence_tolerance=1e-8,
    )

    assert isinstance(modes, ModeSet)
    assert solver.solution is modes
    assert len(modes) == 1
    mode = modes[0]
    assert mode.index == 1
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
    solver = ModeSolver2D(
        C_0,
        1.0,
        0.5,
        num_modes=1,
        guess=expected,
        background_epsilon=epsilon,
    )
    solver.discretize(resolution=(7, 4))
    mode = solver.solve(max_refinements=0, divergence_tolerance=1e-7)[0]

    assert mode.neff.real == pytest.approx(expected.real, rel=4e-3)
    assert mode.neff.imag == pytest.approx(expected.imag, rel=8e-2)
    assert mode.neff.imag < 0.0
    assert mode.attenuation_constant > 0.0
    assert mode.alpha > 0.0


@pytest.mark.gmsh
def test_passive_sibc_post_uses_negative_imaginary_neff() -> None:
    """Exercise the public geometry-to-facet SIBC path and its loss sign."""

    solver = ModeSolver2D(
        C_0,
        1.0,
        0.5,
        num_modes=1,
        neff_guess=0.82,
        boundary="pec",
    )
    post = solver.add_impedance_surface(
        20.0 + 20.0j,
        x_range=(0.56, 0.68),
        y_range=(0.11, 0.23),
        name="lossy_post",
    )
    solver.discretize(resolution=(12, 6))

    assert post.impedance == 20.0 + 20.0j
    assert solver.mesh_data.boundary_facets[post.name].size > 0
    assert solver.system.impedance_boundaries[0][1] == post.impedance

    mode = solver.solve(
        max_refinements=0,
        residual_tolerance=1e-8,
        divergence_tolerance=1e-6,
    )[0]
    assert mode.neff.real > 0.0
    assert mode.neff.imag < 0.0
    assert mode.alpha > 0.0
    assert mode.residual is not None and mode.residual < 1e-8
    assert mode.divergence_residual is not None
    assert mode.divergence_residual < 1e-6


@pytest.mark.gmsh
def test_transformation_optics_pml_uses_forward_passive_branch() -> None:
    solver = ModeSolver2D(
        C_0,
        2.0,
        1.0,
        num_modes=1,
        neff_guess=0.85,
    )
    solver.add_pml(0.2, sigma_max=2.0)
    solver.discretize(resolution=(6, 3))
    mode = solver.solve(
        max_refinements=0,
        direction="all",
        residual_tolerance=1e-7,
        divergence_tolerance=1e-6,
    )[0]

    assert mode.neff.real > 0.0
    assert mode.neff.imag < 0.0
    assert mode.alpha > 0.0
    assert mode.residual is not None and mode.residual < 1e-7
    assert mode.divergence_residual is not None
    assert mode.divergence_residual < 1e-6
