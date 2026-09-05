from __future__ import annotations
from cem_common import Material, SurfaceImpedance, materials, shapes
from fem_waveguide_scattering.constants import C0 as _internal_C0
from fem_waveguide_scattering.hdf5 import load_h5 as _internal_load_h5

import numpy as np
import pytest

import fem_waveguide_scattering as wf


def _uniform_simulation(frequency_hz: float) -> wf.WaveguideScatteringSolver2D:
    simulation = wf.WaveguideScatteringSolver2D(frequency=frequency_hz, ky=0.0, x_range=(0.0, 1e-06), z_range=(-3e-06, 3e-06), background_material=materials.Material(epsilon=1.0, mu=1.0), boundary=materials.PEC)
    simulation.add_pml(order=3, target_reflection=1e-07, thickness=8e-07, direction='z')
    return simulation


@pytest.mark.gmsh
def test_run_persists_complete_result_and_records_absolute_path(tmp_path) -> None:
    frequency_hz = _internal_C0 / 1.55e-6
    simulation = _uniform_simulation(frequency_hz)
    simulation.mesh(max_element_size=0.25e-6, wavelength_elements=6)
    modes = simulation.solve_modes(
        max_refinements=0,
        num_modes=1,
        neff_guess=1.0,
        num_elements=40,
    )
    simulation.set_incident_mode(modes[0], amplitude=1e-13)
    destination = tmp_path / "single-run.h5"

    result = simulation.solve()
    result.save(destination)
    stored = _internal_load_h5(destination)

    assert result.h5_path is None
    assert stored.path == destination.resolve()
    assert stored.kind == "single"
    assert stored.frequencies_hz == pytest.approx([frequency_hz])
    assert len(stored.results) == 1
    saved_result = stored.results[0]
    assert saved_result.frequency_hz == pytest.approx(frequency_hz)
    assert saved_result.ky == pytest.approx(0.0)
    assert saved_result.s_parameters == result.s_parameters
    assert len(saved_result.modes) == 1
    np.testing.assert_allclose(saved_result.E_total, result.E_total)
    np.testing.assert_allclose(saved_result.H_total, result.H_total)
    assert result.scene is not None
    assert saved_result.scene is not None
    assert result.scene.x_span == simulation.x_span
    assert result.scene.z_span == simulation.z_span
    np.testing.assert_allclose(result.scene.eps_r, 1.0)
    assert [line.kind for line in result.scene.lines].count("pec") == 4
    assert [line.kind for line in result.scene.lines].count("wave_port") == 2
    assert [line.kind for line in result.scene.lines].count("pml") == 2
    np.testing.assert_array_equal(saved_result.scene.points, result.scene.points)
    np.testing.assert_array_equal(saved_result.scene.triangles, result.scene.triangles)


@pytest.mark.gmsh
def test_frequency_sweep_solves_each_point_and_persists_one_h5_file(tmp_path) -> None:
    frequencies_hz = np.asarray((_internal_C0 / 1.55e-6, _internal_C0 / 1.50e-6))
    simulation = _uniform_simulation(float(frequencies_hz[0]))
    destination = tmp_path / "frequency-sweep.h5"

    sweep = simulation.sweep(frequencies_hz, mesh_options={'max_element_size': 2.5e-07, 'wavelength_elements': 6}, mode_options={'num_modes': 1, 'neff_guess': 1.0, 'num_elements': 40}, amplitude=1e-13)
    sweep.save(destination)
    stored = _internal_load_h5(destination)

    assert sweep.h5_path is None
    np.testing.assert_allclose(sweep.frequencies_hz, frequencies_hz)
    np.testing.assert_allclose(sweep.S11, 0.0, atol=1e-12)
    np.testing.assert_allclose(sweep.S21, 1.0, atol=3e-11)
    assert stored.kind == "sweep"
    np.testing.assert_allclose(stored.frequencies_hz, frequencies_hz)
    assert len(stored.results) == len(frequencies_hz)
    assert all(len(result.modes) == 1 for result in stored.results)
    assert all(result.scene is not None for result in stored.results)
    # A sweep clones the physical configuration instead of consuming the
    # caller's mesh, modes, or incident-mode state.
    assert simulation.mesh_data is None
    assert simulation.modes is None
    assert simulation.incident is None
