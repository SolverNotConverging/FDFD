"""Adaptive API contracts and coarse-start integration for every FEM backend."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c

from fem_adaptivity import run_adaptive, validate_controls
from FEM_Mode_Solver import ModeSolver1D, ModeSolver2D
from FEM_Periodic_Solver import PeriodicModeSolver2D, PeriodicModeSolver3D
from wavefem.materials import Material
from wavefem.modes import CrossSection, ModeSolver
from wavefem.scattering import Scattering2D, SolverOptions
from wavefem.exceptions import ConfigurationError as WaveConfigurationError


def modal_case(kind):
    if kind == "scalar":
        return ModeSolver1D(c, 1., 3, neff_guess=.85), {}
    if kind == "vector":
        return ModeSolver2D(c, 1., .5, 1, guess=.85), {}
    if kind == "periodic":
        return PeriodicModeSolver2D(c, 1., .25, polarization="TE", num_modes=1, neff_guess=.85), {}
    return ModeSolver(CrossSection(x_span=(0., 1.), background=Material(), boundary="pec"),
                      wavelength=1., num_elements=8), {"num_modes": 1, "neff_guess": .85}


def diagnostics(result):
    return result.metadata if hasattr(result, "metadata") else result.solve_info


@pytest.mark.gmsh
@pytest.mark.parametrize("kind", ["scalar", "vector", "periodic", "lead"])
def test_default_budget_refines_coarse_modes_and_reduces_residual(kind):
    solver, options = modal_case(kind)
    coarse = solver.solve(max_refinements=0, adaptive_tolerance=1e-5, **options)
    before = diagnostics(coarse)
    assert len(before["adaptive_history"]) == 1
    assert not before["adaptive_converged"]
    result = solver.solve(adaptive_tolerance=1e-5, **options)
    info = diagnostics(result)
    history = info["adaptive_history"]
    assert info["max_refinements"] == 2
    assert len(history) == 3
    assert history[-1]["status"] == "refinement_limit"
    assert history[-1]["elements"] > history[0]["elements"]
    assert info["adaptive_residual"] < before["adaptive_residual"]
    assert abs(result[0].neff - np.sqrt(.75)) < 3e-3
    if kind == "periodic":
        mesh = solver.mesh_data
        np.testing.assert_allclose(mesh.nodes[mesh.slave_nodes, 0], mesh.nodes[mesh.master_nodes, 0])
    stopped = solver.solve(adaptive_tolerance=100., **options)
    assert len(diagnostics(stopped)["adaptive_history"]) == 1
    assert diagnostics(stopped)["adaptive_converged"]


@pytest.mark.gmsh
def test_3d_periodic_coarse_start_regenerates_valid_constraints():
    solver = PeriodicModeSolver3D(10e9, .02, .01, .005, num_modes=1,
                                  neff_guess=1.3, background_epsilon=2.25)
    result = solver.solve(adaptive_tolerance=1e-5)
    history = result.metadata["adaptive_history"]
    assert len(history) == 3
    assert history[-1]["status"] == "refinement_limit"
    assert result[0].residual < 1e-8
    expected = np.sqrt(2.25 - (np.pi / (.02 * solver.k0))**2)
    assert abs(result[0].neff - expected) < .01
    pairs = solver.mesh_data.periodic_node_pairs
    nodes = solver.mesh_data.nodes
    np.testing.assert_allclose(nodes[pairs[:, 0], :2], nodes[pairs[:, 1], :2], atol=1e-12)


def scattering_case(contrast=0., **controls):
    simulation = Scattering2D(wavelength=1., x_span=(0., .5), z_span=(-2., 2.),
                              transverse_boundary="pec", solver_options=SolverOptions(**controls))
    if contrast:
        simulation.add_rectangle(x=(0., .5), z=(-.3, .3), eps=1. + contrast)
    simulation.add_pml(z=.5)
    return simulation


@pytest.mark.gmsh
def test_uniform_scattering_auto_setup_stops_on_threshold():
    simulation = scattering_case()
    result = simulation.solve()
    assert result.solve_info["adaptive_converged"]
    assert len(result.solve_info["adaptive_history"]) == 1
    assert result.reflection < 1e-10
    assert result.transmission == pytest.approx(1., abs=1e-7)
    assert simulation.modes.solve_info["max_refinements"] == 2


@pytest.mark.gmsh
def test_scattering_refinement_budget_and_final_persistence(tmp_path):
    from wavefem import load_h5
    simulation = scattering_case(.5, max_refinements=1, adaptive_tolerance=1e-6)
    result = simulation.solve(h5_path=tmp_path / "adapted.h5")
    history = result.solve_info["adaptive_history"]
    assert len(history) == 2
    assert history[-1]["elements"] > history[0]["elements"]
    assert history[-1]["residual"] < history[0]["residual"]
    assert not result.solve_info["adaptive_converged"]
    stored = load_h5(result.h5_path).results[0]
    np.testing.assert_allclose(stored.E_total, result.E_total)
    np.testing.assert_array_equal(stored.scene.triangles, result.scene.triangles)


@pytest.mark.parametrize("value", [-1, 1.5, np.nan, np.inf, True, np.bool_(True), "2", [], 1j])
def test_invalid_refinement_budget(value):
    with pytest.raises(ValueError, match="max_refinements"):
        validate_controls(value, .05)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, np.bool_(True), "0.05", [], 1j, np.complex128(.05)])
def test_invalid_residual_threshold(value):
    with pytest.raises(ValueError, match="adaptive_tolerance"):
        validate_controls(2, value)


def test_manual_refinement_after_adaptation_retains_fine_intervals():
    solver, options = modal_case("scalar")
    solver.solve(max_refinements=3, adaptive_tolerance=1e-5, **options)
    previous_minimum = np.diff(solver.mesh.nodes).min()
    refined = solver.refine()
    assert np.diff(refined.nodes).max() <= previous_minimum / 2 * (1 + 1e-12)


@pytest.mark.parametrize("kind", ["scalar", "vector", "periodic", "lead"])
def test_public_controls_are_validated_before_solving(kind):
    solver, options = modal_case(kind)
    with pytest.raises((ValueError, WaveConfigurationError), match="max_refinements"):
        solver.solve(max_refinements=-1, **options)


def test_failed_solve_consumes_budget_and_remesh_failure_restores_state():
    @dataclass
    class Result:
        metadata: dict

    owner = SimpleNamespace(mesh="coarse", solution=None)
    def solve():
        if owner.mesh == "coarse":
            raise RuntimeError("coarse eigensolve failed")
        owner.solution = Result({})
        return owner.solution

    def refine(indicators):
        owner.mesh = "fine"

    result = run_adaptive(owner, solve, lambda result: (np.ones(2), .1), refine,
                          max_refinements=1, adaptive_tolerance=.1, retry_errors=(RuntimeError,))
    assert result is owner.solution
    assert [step["status"] for step in result.metadata["adaptive_history"]] == ["solve_failed", "tolerance"]
    def fail(indicators):
        owner.mesh = "broken"
        raise RuntimeError("remeshing failed")

    with pytest.raises(RuntimeError, match="remeshing failed"):
        run_adaptive(owner, lambda: result, lambda result: (np.ones(2), 1.), fail,
                     max_refinements=1, adaptive_tolerance=.1)
    assert owner.mesh == "fine"
