"""Adaptive API contracts and coarse-start integration for every FEM backend."""
from cem_common import Material, SurfaceImpedance, materials, shapes

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c

from fem_adaptivity import run_adaptive, validate_controls
from fem_waveguide_modes import ModeSolver1D, ModeSolver2D
from fem_periodic_modes import PeriodicModeSolver2D, PeriodicModeSolver3D
from fem_waveguide_scattering.materials import Material
from fem_waveguide_scattering.modes import CrossSection, ModeSolver
from fem_waveguide_scattering.scattering import WaveguideScatteringSolver2D
from cem_common.errors import ConfigurationError as WaveConfigurationError


def modal_case(kind):
    if kind == "scalar":
        return ModeSolver1D(frequency=c, x_range=1.0), {"num_modes": 3, "neff_guess": .85}
    if kind == "vector":
        return ModeSolver2D(frequency=c, x_range=1.0, y_range=0.5), {"num_modes": 1, "neff_guess": .85}
    if kind == "periodic":
        return PeriodicModeSolver2D(frequency=c, x_range=1.0, z_range=0.25, polarization='TE'), {"num_modes": 1, "neff_guess": .85}
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
    solver = PeriodicModeSolver3D(frequency=10000000000.0, x_range=0.02, y_range=0.01, z_range=0.005, background_material=materials.Material(epsilon=2.25, mu=1.0))
    result = solver.solve(adaptive_tolerance=1e-05, num_modes=1, neff_guess=1.3)
    history = result.metadata["adaptive_history"]
    assert len(history) == 3
    assert history[-1]["status"] == "refinement_limit"
    assert result[0].residual < 1e-8
    expected = np.sqrt(2.25 - (np.pi / (.02 * solver.k0))**2)
    assert abs(result[0].neff - expected) < .01
    pairs = solver.mesh_data.periodic_node_pairs
    nodes = solver.mesh_data.nodes
    np.testing.assert_allclose(nodes[pairs[:, 0], :2], nodes[pairs[:, 1], :2], atol=1e-12)


def scattering_case(contrast=0.):
    simulation = WaveguideScatteringSolver2D(frequency=299792458.0 / 1.0, x_range=(0.0, 0.5), z_range=(-2.0, 2.0), boundary=materials.PEC)
    if contrast:
        simulation.add_rectangle(x_range=(0.0, 0.5), z_range=(-0.3, 0.3), material=materials.Material(epsilon=1.0 + contrast, mu=1.0))
    simulation.add_pml(thickness=0.5, direction='z')
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
    from fem_waveguide_scattering import load_result
    simulation = scattering_case(.5)
    result = simulation.solve(max_refinements=1, adaptive_tolerance=1e-6)
    result.save(tmp_path / "adapted.h5")
    history = result.solve_info["adaptive_history"]
    assert len(history) == 2
    assert history[-1]["elements"] > history[0]["elements"]
    assert history[-1]["residual"] < history[0]["residual"]
    assert not result.solve_info["adaptive_converged"]
    stored = load_result(tmp_path / "adapted.h5")
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
    previous_minimum = np.diff(solver.mesh_data.nodes).min()
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
