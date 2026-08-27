from __future__ import annotations

import numpy as np
import pytest

import wavefem as wf


def _uniform_simulation(frequency_hz: float) -> wf.Scattering2D:
    simulation = wf.Scattering2D(
        frequency=frequency_hz,
        ky=0.0,
        x_span=(0.0, 1.0e-6),
        z_span=(-3.0e-6, 3.0e-6),
        background_eps=1.0,
        transverse_boundary="pec",
    )
    simulation.add_pml(z=0.8e-6, order=3, target_reflection=1e-7)
    return simulation


@pytest.mark.gmsh
def test_run_persists_complete_result_and_records_absolute_path(tmp_path) -> None:
    frequency_hz = wf.C0 / 1.55e-6
    simulation = _uniform_simulation(frequency_hz)
    simulation.mesh(max_element_size=0.25e-6, wavelength_elements=6)
    modes = simulation.solve_modes(
        num_modes=1,
        neff_guess=1.0,
        num_elements=40,
    )
    simulation.set_incident_mode(modes[0], amplitude=1e-13)
    destination = tmp_path / "single-run.h5"

    result = simulation.run(h5_path=destination)
    stored = wf.load_h5(destination)

    assert result.h5_path == destination.resolve()
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


@pytest.mark.gmsh
def test_frequency_sweep_solves_each_point_and_persists_one_h5_file(tmp_path) -> None:
    frequencies_hz = np.asarray((wf.C0 / 1.55e-6, wf.C0 / 1.50e-6))
    simulation = _uniform_simulation(float(frequencies_hz[0]))
    destination = tmp_path / "frequency-sweep.h5"

    sweep = simulation.sweep_frequencies(
        frequencies_hz,
        h5_path=destination,
        mesh_options={
            "max_element_size": 0.25e-6,
            "wavelength_elements": 6,
        },
        mode_options={
            "num_modes": 1,
            "neff_guess": 1.0,
            "num_elements": 40,
        },
        amplitude=1e-13,
    )
    stored = wf.load_h5(destination)

    assert sweep.h5_path == destination.resolve()
    np.testing.assert_allclose(sweep.frequencies_hz, frequencies_hz)
    np.testing.assert_allclose(sweep.S11, 0.0, atol=1e-12)
    np.testing.assert_allclose(sweep.S21, 1.0, atol=3e-11)
    assert stored.kind == "sweep"
    np.testing.assert_allclose(stored.frequencies_hz, frequencies_hz)
    assert len(stored.results) == len(frequencies_hz)
    assert all(len(result.modes) == 1 for result in stored.results)
    # A sweep clones the physical configuration instead of consuming the
    # caller's mesh, modes, or incident-mode state.
    assert simulation.mesh_data is None
    assert simulation.modes is None
    assert simulation.incident is None
