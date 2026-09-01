from __future__ import annotations

import numpy as np
import pytest

from Electrostatic_Solver import Circle, ElectrostaticSolver, Interval


pytestmark = pytest.mark.gmsh


def test_1d_linear_dirichlet_solution_and_field() -> None:
    solver = ElectrostaticSolver(dim=1, domain=(0.0, 1.0), outer_potential=None)
    solver.set_potential("left", 0.0, name="ground")
    solver.set_potential("right", 10.0, name="hot")
    mesh = solver.discretize(max_element_size=0.08)
    result = solver.solve()

    x = mesh.nodes[:, 0]
    np.testing.assert_allclose(result.potential, 10.0 * x, atol=2e-11)
    np.testing.assert_allclose(result.electric_field[:, 0], -10.0, atol=2e-10)
    assert result.residual_norm < 1e-12


def test_1d_dielectric_interface_preserves_displacement() -> None:
    solver = ElectrostaticSolver(dim=1, domain=(0.0, 1.0), outer_potential=None)
    solver.add_object(Interval((0.5, 1.0)), erxx=4.0, name="high_dk")
    solver.set_potential("left", 0.0)
    solver.set_potential("right", 1.0)
    result = solver.solve()

    x = result.coordinates[:, 0]
    left_field = np.median(result.electric_field[x < 0.4, 0])
    right_field = np.median(result.electric_field[x > 0.6, 0])
    assert left_field / right_field == pytest.approx(4.0, rel=2e-2)


def test_2d_linear_solution_with_natural_top_and_bottom() -> None:
    solver = ElectrostaticSolver(dim=2, domain=((0.0, 1.0), (0.0, 0.5)), outer_potential=None)
    solver.set_potential("left", 0.0)
    solver.set_potential("right", 1.0)
    result = solver.solve()

    x = result.coordinates[:, 0]
    np.testing.assert_allclose(result.potential, x, atol=2e-10)
    np.testing.assert_allclose(result.electric_field[:, 0], -1.0, atol=2e-9)
    np.testing.assert_allclose(result.electric_field[:, 1], 0.0, atol=2e-9)


def test_charge_density_drives_poisson_solution() -> None:
    solver = ElectrostaticSolver(dim=1, domain=(0.0, 1.0))
    solver.add_charge_density(Interval((0.0, 1.0)), 1e-12)
    result = solver.solve()

    assert result.potential.max() > 0.0
    peak = result.potential[np.argmax(result.potential)]
    assert peak == pytest.approx(1e-12 / (8.854_187_812_8e-12) / 8.0, rel=2e-3)


def test_curved_internal_conductor_is_conforming_and_fixed() -> None:
    solver = ElectrostaticSolver(dim=2, domain=((0.0, 1.0), (0.0, 1.0)))
    electrode = Circle((0.5, 0.5), 0.15)
    solver.set_potential(electrode, 1.0, name="round_electrode")
    result = solver.solve()

    selected = electrode.contains(result.coordinates[:, 0], result.coordinates[:, 1])
    assert np.count_nonzero(selected) > 4
    np.testing.assert_allclose(result.potential[selected], 1.0)
    assert result.potential.min() >= -1e-12
    assert result.potential.max() <= 1.0 + 1e-12


def test_geometry_change_invalidates_mesh_and_solution() -> None:
    solver = ElectrostaticSolver(dim=1)
    solver.solve()
    assert solver.mesh is not None and solver.solution is not None

    solver.add_object(Interval((0.2, 0.4)), erxx=3.0)

    assert solver.mesh is None
    assert solver.solution is None
