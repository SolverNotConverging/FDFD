"""Solution-driven refinement must improve fields and preserve interfaces."""

import numpy as np
import pytest

from Electrostatic_Solver import ElectrostaticSolver, Interval, Rectangle, SolverError
from Electrostatic_Solver.solver import EPSILON_0


@pytest.mark.gmsh
@pytest.mark.parametrize("dimension", [1, 2])
def test_default_adaptation_reduces_poisson_field_error(dimension):
    domain = (0.0, 1.0) if dimension == 1 else ((0.0, 1.0), (0.0, 0.5))
    solver = ElectrostaticSolver(dim=dimension, domain=domain, outer_potential=None)
    solver.set_potential("left", 0.0)
    solver.set_potential("right", 0.0)
    shape = Interval((0.0, 1.0)) if dimension == 1 else Rectangle((0.0, 1.0), (0.0, 0.5))
    solver.add_charge_density(shape, EPSILON_0)
    mesh = solver.discretize(max_element_size=0.22, boundary_refinement=None)
    before = solver.solve(adaptive=False)
    result = solver.solve()
    assert len(result.elements) > len(mesh.elements)
    assert len(result.adaptive_history) == 3
    assert result.adaptive_residual == result.adaptive_history[-1]["residual"]
    assert result.adaptive_converged == (result.adaptive_history[-1]["status"] == "tolerance")
    assert result.adaptive_history[-1]["relative_indicator"] < before.adaptive_history[0]["relative_indicator"]
    # Integrate squared error against E_x=x-1/2 using element quadrature.
    from skfem import Basis
    from skfem.element import ElementLineP1, ElementTriP1

    def error(solution):
        basis = Basis(solution.mesh.mesh, ElementLineP1() if dimension == 1 else ElementTriP1(), intorder=4)
        exact = basis.global_coordinates()[0] - 0.5
        difference = solution.element_electric_field[:, 0, None] - exact
        return np.sqrt(np.sum(basis.dx * difference**2))

    assert error(result) < error(before)
    assert result.residual_norm < 1e-10


@pytest.mark.gmsh
def test_exact_dielectric_flux_does_not_trigger_refinement():
    solver = ElectrostaticSolver(dim=1, domain=(0.0, 1.0), outer_potential=None)
    solver.add_object(Interval((0.5, 1.0)), erxx=4.0)
    solver.set_potential("left", 0.0)
    solver.set_potential("right", 1.0)
    mesh = solver.discretize(max_element_size=0.1)
    result = solver.solve()
    assert result.mesh is mesh
    assert result.adaptive_history[-1]["status"] == "tolerance"


@pytest.mark.gmsh
def test_element_budget_preserves_last_usable_mesh():
    solver = ElectrostaticSolver(dim=1, domain=(0.0, 1.0))
    solver.add_charge_density(Interval((0.0, 1.0)), EPSILON_0)
    mesh = solver.discretize(max_element_size=0.2)
    result = solver.solve(adaptive_tolerance=1e-6, max_elements=len(mesh.elements))
    assert result.mesh is mesh
    assert result.adaptive_history[-1]["status"] == "element_limit"


@pytest.mark.gmsh
def test_repeated_local_refinement_preserves_parent_materials():
    solver = ElectrostaticSolver(dim=1, domain=(0.0, 1.0))
    solver.add_object(Interval((0.5, 1.0)), erxx=4.0)
    solver.add_charge_density(Interval((0.0, 1.0)), EPSILON_0)
    solver.discretize(max_element_size=0.2, material_aware=False,
                      interface_refinement=None, boundary_refinement=None)
    result = solver.solve(max_refinements=3, adaptive_tolerance=1e-6)
    assert len(result.adaptive_history) == 4
    assert result.adaptive_history[-1]["relative_indicator"] < result.adaptive_history[0]["relative_indicator"]
    centers = result.coordinates[result.elements].mean(axis=1)[:, 0]
    np.testing.assert_array_equal(result.mesh.element_tags, np.where(centers < 0.5, 1, 2))
    assert np.sum(np.abs(np.diff(result.coordinates[result.elements, 0], axis=1))) == pytest.approx(1.0)


@pytest.mark.parametrize("options", [{"max_refinements": -1}, {"max_refinements": np.nan},
                                     {"max_elements": True}, {"marking_fraction": 1.1}])
def test_invalid_adaptation_controls_fail_before_meshing(options):
    solver = ElectrostaticSolver(dim=1)
    with pytest.raises(SolverError):
        solver.solve(**options)
    assert solver.mesh is None
