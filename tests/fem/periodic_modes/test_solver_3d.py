from __future__ import annotations
from cem_common import Material, SurfaceImpedance, materials, shapes

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import h5py

from fem_periodic_modes import PeriodicModeSolver2D, PeriodicModeSolver3D
from cem_common.shapes import Box, Cylinder, Sphere
from cem_common.errors import SolverError
from fem_periodic_modes.exceptions import NotDiscretizedError
from fem_periodic_modes.geometry import GeometryModel3D
from fem_periodic_modes.materials import Material
from fem_periodic_modes.assembly_3d import assemble_periodic_system_3d
from fem_periodic_modes.persistence import load_periodic_h5, save_periodic_h5, validate_periodic_h5
from fem_periodic_modes.meshing_3d import discretize_3d
import fem_periodic_modes.solver_3d as solver_3d_module
from fem_periodic_modes.solver_3d import _dense_candidates


FREQUENCY = 10.0e9
WIDTH = 20.0e-3
HEIGHT = 10.0e-3
PERIOD = 5.0e-3
EPSILON = 2.25


def test_refinement_failure_rolls_back_density(monkeypatch):
    solver = PeriodicModeSolver3D(frequency=FREQUENCY, x_range=WIDTH, y_range=HEIGHT, z_range=PERIOD)
    solver._discretize_options = {"max_element_size": 0.006}

    def fail(**kwargs):
        raise RuntimeError("simulated mesh failure")

    monkeypatch.setattr(solver, "_mesh_impl", fail)
    with pytest.raises(RuntimeError, match="simulated mesh"):
        solver.refine()
    assert solver._refinement_scale == 1.0


def test_geometry_changes_clear_all_modal_arrays():
    solver = PeriodicModeSolver3D(frequency=FREQUENCY, x_range=WIDTH, y_range=HEIGHT, z_range=PERIOD)
    for name in ("neff", "beta", "gamma"):
        assert getattr(solver, name) is None
        setattr(solver, name, np.ones(1))
    solver.set_boundary(material=materials.PMC)
    for name in ("result", "modes", "neff", "beta", "gamma"):
        assert getattr(solver, name) is None


@pytest.mark.parametrize("invalid", [True, np.nan, np.inf, 1.5, [], "3"])
def test_invalid_3d_integer_controls_raise_configuration_error(invalid):
    from cem_common.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match="num_modes"):
        PeriodicModeSolver3D(frequency=FREQUENCY, x_range=WIDTH, y_range=HEIGHT, z_range=PERIOD).solve(num_modes=invalid, max_refinements=0)


@pytest.fixture(scope="module")
def guide() -> PeriodicModeSolver3D:
    solver = PeriodicModeSolver3D(frequency=FREQUENCY, x_range=(0.0, WIDTH), y_range=(0.0, HEIGHT), z_range=(0.0, PERIOD), background_material=materials.Material(epsilon=EPSILON, mu=1.0))
    solver.mesh(max_element_size=0.006, wavelength_elements=8)
    return solver


@pytest.fixture(scope="module")
def guide_modes(guide: PeriodicModeSolver3D):
    return guide.solve(max_refinements=0, direction='all', eigensolver='dense', num_modes=1, neff_guess=1.3)


def test_nedelec_algebra_periodicity_and_orientation(guide: PeriodicModeSolver3D) -> None:
    system = guide.system
    mesh = guide.mesh_data
    assert system is not None and mesh is not None
    assert max(system.relative_hermiticity_errors()) < 1.0e-12

    rng = np.random.default_rng(20260831)
    reduced = rng.standard_normal(system.ndofs) + 1j * rng.standard_normal(system.ndofs)
    full = system.expand(reduced)
    assert system.prolongation.equality_error(full) < 1.0e-12
    assert mesh.periodic_edge_pairs.shape[1] == 3
    for slave, master, sign in mesh.periodic_edge_pairs:
        assert system.prolongation.representatives[slave] == master
        assert system.prolongation.signs[slave] == sign
    assert set(np.unique(mesh.cell_edge_signs)).issubset({-1, 1})

    neff = 1.1 - 0.03j
    first = system.gauss0 @ reduced
    second = neff * (system.gauss1 @ reduced)
    scale = (
        np.linalg.norm(system.gauss0.toarray())
        + abs(neff) * np.linalg.norm(system.gauss1.toarray())
    ) * np.linalg.norm(reduced)
    amplitude_ratio = np.linalg.norm(first + second) / scale
    assert system.divergence_residual(reduced, neff) == pytest.approx(
        amplitude_ratio**2,
        rel=2e-14,
        abs=1e-20,
    )


@pytest.mark.parametrize(('shape', 'kind'), ((shapes.Cylinder(center=(0.32, 0.32), radius=0.12, z_range=(0.2, 0.8)), 'pec'), (shapes.Sphere(center=(0.68, 0.68, 0.5), radius=0.14), 'pmc')))
def test_curved_conductor_facets_keep_occ_ownership(shape, kind: str) -> None:
    geometry = GeometryModel3D((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), Material())
    name = f"round_{kind}"
    geometry.add_boundary(shape, kind, name=name)
    mesh = discretize_3d(
        geometry,
        max_element_size=0.25,
        wavelength_elements=4,
        material_aware=False,
    )
    assert mesh.boundary_facets[kind].size > 0
    np.testing.assert_array_equal(mesh.boundary_facets[name], mesh.boundary_facets[kind])


def test_material_aware_false_uses_free_space_wavelength() -> None:
    geometry = GeometryModel3D((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), Material(epsilon=16.0, mu=1.0))
    mesh = discretize_3d(
        geometry,
        max_element_size=0.5,
        wavelength_elements=4,
        material_aware=False,
        k0=2.0 * np.pi,
    )
    assert mesh.info.requested_maximum_edge == pytest.approx(0.25)


def test_material_aware_mesh_refines_high_index_volume_locally() -> None:
    geometry = GeometryModel3D(
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        Material(),
    )
    geometry.add_region(shapes.Box(bounds=((0.0, 0.3), (0.0, 1.0), (0.0, 1.0))), Material(epsilon=16.0, mu=1.0), name='high_index')
    mesh = discretize_3d(
        geometry,
        max_element_size=0.35,
        wavelength_elements=4,
        material_aware=True,
    )
    points = mesh.nodes[mesh.elements]
    edge_pairs = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
    maximum_edges = np.max(
        np.stack(
            [
                np.linalg.norm(points[:, first] - points[:, second], axis=1)
                for first, second in edge_pairs
            ]
        ),
        axis=0,
    )
    centres = points.mean(axis=1)
    high_index = centres[:, 0] < 0.29
    low_index_far = centres[:, 0] > 0.55
    assert np.count_nonzero(high_index) > 100
    assert np.count_nonzero(low_index_far) > 20
    assert np.median(maximum_edges[high_index]) < 0.5 * np.median(
        maximum_edges[low_index_far]
    )


def test_3d_refine_and_geometry_invalidation() -> None:
    solver = PeriodicModeSolver3D(frequency=20000000000.0, x_range=(0.0, 0.004), y_range=(0.0, 0.003), z_range=(0.0, 0.002))
    coarse = solver.mesh(max_element_size=0.002, wavelength_elements=4)
    fine = solver.refine(2.0)
    assert fine.info.requested_maximum_edge == pytest.approx(
        coarse.info.requested_maximum_edge / 2.0
    )
    assert fine.info.elements > coarse.info.elements
    solver.add_box(x_range=(0.001, 0.002), y_range=(0.001, 0.002), z_range=(0.0005, 0.0015), material=materials.Material(epsilon=2.0, mu=1.0))
    with pytest.raises(NotDiscretizedError):
        solver._solve_once(num_modes=1)


@pytest.mark.slow
def test_rectangular_guide_te10_and_extruded_2d_agreement(
    guide: PeriodicModeSolver3D,
    guide_modes,
) -> None:
    mode = guide_modes[0]
    cutoff_ratio = np.pi / (WIDTH * guide.k0)
    analytic = np.sqrt(EPSILON - cutoff_ratio**2)
    assert mode.neff.real == pytest.approx(analytic, rel=1.0e-2)
    assert abs(mode.neff.imag) < 1.0e-10
    assert mode.residual is not None and mode.residual < 1.0e-8
    assert mode.gauss_residual is not None and mode.gauss_residual < 1.0e-6
    assert mode.polarization == "TE"

    solver_2d = PeriodicModeSolver2D(frequency=FREQUENCY, x_range=(0.0, WIDTH), z_range=(0.0, PERIOD), polarization='TE', background_material=materials.Material(epsilon=EPSILON, mu=1.0))
    solver_2d.mesh(max_element_size=0.0015)
    result_2d = solver_2d.solve(max_refinements=0, direction='all', eigensolver='dense', num_modes=1, neff_guess=analytic)
    assert result_2d[0].neff.real == pytest.approx(mode.neff.real, rel=2.0e-2)


def test_3d_pml_fraction_filter_can_reject_or_be_disabled(
    guide: PeriodicModeSolver3D,
    guide_modes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert guide.system is not None
    source = guide_modes[0]
    roots = guide.system.prolongation.independent_representatives
    reduced = np.asarray(source.coefficients[roots], dtype=np.complex128)

    def candidates(*args, **kwargs):
        return (
            np.asarray([source.neff], dtype=np.complex128),
            reduced[:, None],
            np.asarray([source.residual], dtype=np.float64),
            "test-candidate",
        )

    original_make_mode = guide._make_mode

    def pml_dominated(*args, **kwargs):
        return replace(original_make_mode(*args, **kwargs), pml_fraction=0.9)

    monkeypatch.setattr(solver_3d_module, "_dense_candidates", candidates)
    monkeypatch.setattr(guide, "_make_mode", pml_dominated)
    with pytest.raises(SolverError, match=r"PML=1"):
        guide.solve(max_refinements=0, direction='all', eigensolver='dense', max_pml_fraction=0.5, num_modes=1, neff_guess=1.3)
    mode = guide.solve(max_refinements=0, direction='all', eigensolver='dense', max_pml_fraction=None, num_modes=1, neff_guess=1.3)[0]
    assert mode.pml_fraction == pytest.approx(0.9)


def test_3d_hdf5_round_trip_keeps_canonical_edges(guide_modes, tmp_path) -> None:
    path = save_periodic_h5(guide_modes, tmp_path / "guide-3d.h5")
    assert validate_periodic_h5(path, deep=True).mode_count == 1
    with h5py.File(path, "r") as archive:
        mesh = archive["meshes/000000"]
        assert mesh["edge_nodes"].shape[1] == 2
        assert mesh["cell_edges"].shape[1] == 6
        assert mesh["cell_edge_sign"].shape == mesh["cell_edges"].shape
        assert mesh["periodic/edge_pairs"].shape[1] == 2
        assert mesh["periodic/edge_sign"].shape[0] == mesh["periodic/edge_pairs"].shape[0]
    restored = load_periodic_h5(path)
    assert restored[0].neff == pytest.approx(guide_modes[0].neff)
    np.testing.assert_array_equal(restored[0].coefficients, guide_modes[0].coefficients)


def test_3d_visualize_gui_saves_and_launches_all_modes(
    guide, guide_modes, tmp_path, monkeypatch
) -> None:
    from fem_periodic_modes import visualization

    marker = object()
    launched: list[Path] = []
    monkeypatch.setattr(visualization.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        visualization,
        "launch_viewer",
        lambda path, *, _remove_on_exit=False: (
            launched.append(Path(path)) or marker
        ),
    )
    assert guide.show(block=False) is marker
    assert len(launched) == 1
    assert validate_periodic_h5(launched[0], deep=True).mode_count == len(guide_modes)
    launched[0].unlink()


def test_3d_visualize_creates_matplotlib_figure(guide, guide_modes) -> None:
    from fem_periodic_modes.visualization import visualize
    figure, axes = visualize(
        guide.result.mode(0),
        component="E",
        quantity="abs",
        slice_axis="z",
        slice_fraction=0.5,
        show=False,
    )
    assert axes.name == "3d"
    assert figure is axes.figure
    assert "Mode 0" in axes.get_title()


@pytest.mark.slow
def test_diagonal_anisotropic_rectangular_guide_te10(
    guide: PeriodicModeSolver3D,
) -> None:
    assert guide.mesh_data is not None
    epsilon = np.asarray((1.8, 2.25, 2.6), dtype=np.complex128)
    mu = np.asarray((1.2, 1.0, 0.8), dtype=np.complex128)

    def material(x, y, z):
        return (
            np.broadcast_to(epsilon.reshape(3, 1, 1), (3, *x.shape)),
            np.broadcast_to(mu.reshape(3, 1, 1), (3, *x.shape)),
        )

    system = assemble_periodic_system_3d(
        guide.mesh_data,
        frequency=guide.frequency,
        k0=guide.k0,
        material_at=material,
    )
    cutoff_ratio = np.pi / (WIDTH * guide.k0)
    analytic = np.sqrt(mu[0].real * epsilon[1].real - (mu[0].real / mu[2].real) * cutoff_ratio**2)
    values, vectors, _, _ = _dense_candidates(system, analytic, 4)
    mode = vectors[:, 0] / np.linalg.norm(vectors[:, 0])
    assert values[0].real == pytest.approx(analytic, rel=1.0e-2)
    assert abs(values[0].imag) < 1.0e-10
    assert system.relative_residual(mode, values[0]) < 1.0e-8
    assert system.divergence_residual(mode, values[0]) < 1.0e-6
