from __future__ import annotations

import numpy as np
import pytest

from FEM_Periodic_Solver import (
    NotDiscretizedError,
    PeriodicModeSolver2D,
    SolverError,
)
from FEM_Periodic_Solver.constants import C_0


@pytest.mark.parametrize("stage", ["mesh", "assembly"])
def test_failed_refinement_preserves_previous_discretization(monkeypatch, stage):
    import FEM_Periodic_Solver.solver_2d as module

    solver = PeriodicModeSolver2D(10e9, 0.02, 0.005, num_modes=1, neff_guess=0.66)
    mesh = solver.discretize(max_element_size=0.004)
    systems = solver._require_current_systems()
    result = solver.solve(max_refinements=0)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated refinement failure")

    monkeypatch.setattr(module, "discretize_periodic_2d" if stage == "mesh"
                        else "assemble_periodic_system_2d", fail)
    with pytest.raises(RuntimeError, match="simulated refinement"):
        solver.refine()
    assert solver.mesh_data is mesh
    assert solver._require_current_systems() is systems
    assert solver.result is result


def _triangle_maximum_edges(mesh) -> np.ndarray:
    points = mesh.nodes[mesh.elements]
    return np.max(
        np.stack(
            (
                np.linalg.norm(points[:, 0] - points[:, 1], axis=1),
                np.linalg.norm(points[:, 1] - points[:, 2], axis=1),
                np.linalg.norm(points[:, 2] - points[:, 0], axis=1),
            )
        ),
        axis=0,
    )


def test_material_and_internal_pec_automatic_mesh_refinement() -> None:
    mm = 1.0e-3
    solver = PeriodicModeSolver2D(
        20.0e9,
        (0.0, 10.0 * mm),
        (0.0, 8.0 * mm),
        polarization="TM",
    )
    solver.add_rectangle(
        10.2,
        1.0,
        (0.0, 1.27 * mm),
        (0.0, 8.0 * mm),
        name="grounded_dielectric_slab",
    )
    solver.add_pec(
        (1.27 * mm, 1.32 * mm),
        (1.0 * mm, 2.0 * mm),
        name="top_pec_perturbation",
    )
    mesh = solver.discretize(max_element_size=0.65 * mm)

    centres = mesh.nodes[mesh.elements].mean(axis=1)
    maximum_edges = _triangle_maximum_edges(mesh)
    high_index = centres[:, 0] < 1.26 * mm
    near_pec = (
        (np.abs(centres[:, 0] - 1.295 * mm) < 0.4 * mm)
        & (centres[:, 1] > 0.7 * mm)
        & (centres[:, 1] < 2.3 * mm)
    )
    low_index_far = (
        (centres[:, 0] > 4.0 * mm)
        & (centres[:, 0] < 7.0 * mm)
        & (centres[:, 1] > 2.0 * mm)
        & (centres[:, 1] < 6.0 * mm)
    )
    assert np.count_nonzero(high_index) > 20
    assert np.count_nonzero(near_pec) > 10
    assert np.count_nonzero(low_index_far) > 10
    coarse_median = np.median(maximum_edges[low_index_far])
    assert np.median(maximum_edges[high_index]) < 0.5 * coarse_median
    assert np.median(maximum_edges[near_pec]) < 0.5 * coarse_median


def test_uniform_pec_te_mode_matches_analytic_cutoff() -> None:
    frequency = 10e9
    width = 20e-3
    solver = PeriodicModeSolver2D(
        frequency,
        (0.0, width),
        (0.0, 5e-3),
        num_modes=1,
        neff_guess=0.66,
        polarization="TE",
    )
    mesh = solver.discretize(max_element_size=2e-3)
    mode = solver.solve(max_refinements=0, direction="forward")[0]
    k0 = 2.0 * np.pi * frequency / C_0
    expected = np.sqrt(1.0 - (np.pi / (k0 * width)) ** 2)
    assert mode.neff.real == pytest.approx(expected, rel=6e-3)
    assert abs(mode.neff.imag) < 1e-10
    assert mode.residual is not None and mode.residual < 1e-9
    assert mode.power is not None and mode.power.real == pytest.approx(1.0, rel=1e-10)
    assert mode.polarization == "TE"
    assert mode.fields.metadata["cell_epsilon_r"].shape == (mesh.info.elements, 3)
    assert mode.fields.metadata["cell_mu_r"].shape == (mesh.info.elements, 3)
    assert mode.fields.metadata["cell_pml_fraction"].shape == (mesh.info.elements,)
    assert mode.fields.metadata["periodic_node_pairs"].shape[1] == 2
    assert mode.fields.metadata["periodic_node_pairs"].dtype == np.int64
    assert mode.fields.metadata["sampling"] == "element-barycentre"
    assert mode.fields.coordinates.shape == (mesh.info.elements, 2)
    np.testing.assert_allclose(
        mode.fields.coordinates,
        mesh.nodes[mesh.elements].mean(axis=1),
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        mode.component("Ey"),
        mode.coefficients[mesh.elements].mean(axis=1),
        rtol=2e-14,
        atol=2e-14,
    )
    assert np.max(
        np.linalg.norm(
            mesh.nodes[mesh.slave_nodes]
            - mesh.nodes[mesh.master_nodes]
            - np.asarray((0.0, solver.period)),
            axis=1,
        )
    ) < 1e-12
    assert solver.system.relative_hermiticity_errors() == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_uniform_pec_tm_plane_wave_and_reconstructed_fields() -> None:
    solver = PeriodicModeSolver2D(
        10e9,
        (0.0, 4e-3),
        (0.0, 5e-3),
        num_modes=1,
        neff_guess=1.5,
        polarization="TM",
        background_epsilon=2.25,
        boundary="pec",
    )
    solver.discretize(max_element_size=1e-3)
    mode = solver.solve(max_refinements=0, direction="forward", eigensolver="dense")[0]
    assert mode.neff == pytest.approx(1.5, rel=2e-4, abs=1e-8)
    assert mode.polarization == "TM"
    assert np.max(np.abs(mode.component("Hy"))) > 0.0
    assert np.max(np.abs(mode.component("Ex"))) > 0.0
    assert np.max(np.abs(mode.component("Ey"))) == 0.0
    assert np.max(np.abs(mode.component("Hz"))) == 0.0


def test_geometry_change_invalidates_discretization() -> None:
    solver = PeriodicModeSolver2D(10e9, 20e-3, 5e-3, polarization="TM")
    with pytest.raises(NotDiscretizedError):
        solver._solve_once()
    solver.discretize(max_element_size=2e-3)
    solver.add_rectangle(2.0, 1.0, (5e-3, 8e-3), (1e-3, 2e-3))
    with pytest.raises(NotDiscretizedError):
        solver._solve_once()


def test_periodic_seam_allows_a_material_interface_when_topology_matches() -> None:
    matching = PeriodicModeSolver2D(10e9, 20e-3, 5e-3, polarization="TM")
    matching.add_rectangle(2.0, 1.0, (5e-3, 10e-3), (0.0, 1e-3))
    matching.add_rectangle(2.0, 1.0, (5e-3, 10e-3), (4e-3, 5e-3))
    matching.discretize(max_element_size=1.5e-3)

    mismatched = PeriodicModeSolver2D(10e9, 20e-3, 5e-3, polarization="TM")
    mismatched.add_rectangle(2.0, 1.0, (5e-3, 10e-3), (0.0, 1e-3))
    mismatched.add_rectangle(3.0, 1.0, (5e-3, 10e-3), (4e-3, 5e-3))
    mesh = mismatched.discretize(max_element_size=1.5e-3)
    assert mesh.slave_nodes.shape == mesh.master_nodes.shape


def test_uniform_guide_returns_reciprocal_aliases() -> None:
    solver = PeriodicModeSolver2D(
        10e9, (0.0, 20e-3), (0.0, 5e-3), num_modes=2,
        neff_guess=0.0, polarization="TE",
    )
    solver.discretize(max_element_size=2e-3)
    result = solver.solve(max_refinements=0, direction="all", eigensolver="dense")
    assert result[0].neff == pytest.approx(-result[1].neff, abs=2e-12)
    assert {mode.direction for mode in result} == {"forward", "backward"}


def test_lossy_longitudinal_bilayer_matches_transfer_matrix() -> None:
    frequency = 10e9
    period = 5e-3
    thickness = period / 2.0
    epsilon_1 = 2.25 - 0.03j
    epsilon_2 = 4.0 - 0.08j
    k0 = 2.0 * np.pi * frequency / C_0
    n1, n2 = np.sqrt(epsilon_1), np.sqrt(epsilon_2)
    phase_1, phase_2 = k0 * n1 * thickness, k0 * n2 * thickness
    trace_half = (
        np.cos(phase_1) * np.cos(phase_2)
        - 0.5 * (n1 / n2 + n2 / n1) * np.sin(phase_1) * np.sin(phase_2)
    )
    phase = np.arccos(trace_half)
    candidates = [
        sign * phase / (k0 * period) + 2.0 * np.pi * branch / (k0 * period)
        for sign in (1.0, -1.0)
        for branch in range(-2, 3)
    ]
    expected = min(candidates, key=lambda value: abs(value - (1.8 - 0.02j)))

    solver = PeriodicModeSolver2D(
        frequency,
        (0.0, 4e-3),
        (0.0, period),
        num_modes=1,
        neff_guess=expected,
        polarization="TE",
        background_epsilon=epsilon_1,
        boundary="pmc",
    )
    solver.add_rectangle(epsilon_2, 1.0, (0.0, 4e-3), (thickness, period))
    solver.discretize(max_element_size=0.4e-3)
    mode = solver.solve(max_refinements=0, direction="all", eigensolver="dense")[0]
    assert mode.neff == pytest.approx(expected, rel=3e-4, abs=3e-5)
    assert mode.neff.imag < 0.0
    assert mode.residual is not None and mode.residual < 1e-9


def test_transverse_pml_has_passive_sign_and_converges() -> None:
    roots: list[complex] = []
    for maximum in (2.0e-3, 1.0e-3):
        solver = PeriodicModeSolver2D(
            10e9,
            (0.0, 20e-3),
            (0.0, 5e-3),
            num_modes=1,
            neff_guess=1.33 - 0.07j,
            polarization="TE",
            background_epsilon=2.25,
        )
        solver.add_pml(3e-3, direction="x", sigma_max=3.0)
        solver.discretize(max_element_size=maximum)
        mode = solver.solve(max_refinements=0, direction="all", eigensolver="dense")[0]
        assert mode.neff.imag < 0.0
        assert mode.gamma.real > 0.0
        assert 0.0 < mode.pml_fraction < 1.0
        if maximum == 2.0e-3:
            with pytest.raises(SolverError, match=r"PML=[1-9]"):
                solver.solve(
                    max_refinements=0,
                    direction="all",
                    eigensolver="dense",
                    max_pml_fraction=0.0,
                )
            unfiltered = solver.solve(
                max_refinements=0,
                direction="all",
                eigensolver="dense",
                max_pml_fraction=None,
            )[0]
            assert unfiltered.pml_fraction == pytest.approx(mode.pml_fraction)
        roots.append(mode.neff)
    assert abs(roots[1] - roots[0]) / abs(roots[1]) < 1.5e-2


def test_uniform_fem_and_fdfd_agree() -> None:
    from Periodic_Solver_2D.Periodic_Solver_2D import (
        PeriodicModeSolver2D as FDFDModeSolver2D,
    )

    frequency = 10e9
    epsilon = 2.25
    width, period = 4e-3, 5e-3
    k0 = 2.0 * np.pi * frequency / C_0
    fdfd = FDFDModeSolver2D(
        "TE", frequency, width, period, 30, 30, 1,
        guess=1j * k0 * np.sqrt(epsilon),
    )
    fdfd.add_rectangle(epsilon, 1.0, (0.0, width), (0.0, period))
    fdfd.solve(method="eigs")

    fem = PeriodicModeSolver2D(
        frequency,
        (0.0, width),
        (0.0, period),
        num_modes=1,
        neff_guess=np.sqrt(epsilon),
        polarization="TE",
        background_epsilon=epsilon,
        boundary="pmc",
    )
    fem.discretize(max_element_size=1e-3)
    fem_neff = fem.solve(max_refinements=0, direction="all", eigensolver="dense")[0].neff
    assert fem_neff == pytest.approx(fdfd.neff[0], rel=2.0e-2, abs=1.0e-8)
