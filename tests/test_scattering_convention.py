"""Physical conjugation regression against the pre-1.0 negative-time solver.

The fixture was captured before the sign migration. Inputs represent the same
real-time device and excitation, with conjugated loss and source amplitude.
"""
from pathlib import Path

import numpy as np
import pytest

from fem_waveguide_scattering.scattering import WaveguideScatteringSolver2D


@pytest.mark.gmsh
@pytest.mark.parametrize("name,angle", [("lossy", 0.0), ("oblique", 17.0)])
def test_positive_time_fields_and_s_parameters_match_physical_baseline(name, angle, tmp_path):
    simulation = WaveguideScatteringSolver2D(
        frequency=193.414489e12, angle=angle,
        x_range=(0.0, 1e-6), z_range=(-3e-6, 3e-6),
        background_epsilon=1.0, transverse_boundary="pec",
    )
    simulation.add_rectangle(x_range=(0.0, 1e-6), z_range=(-0.35e-6, 0.35e-6), epsilon=1.02-0.01j)
    simulation.add_pml(thickness=0.9e-6, direction="z")
    simulation.mesh(max_element_size=0.2e-6)
    modes = simulation.solve_modes(num_modes=1, neff_guess=1.0, num_elements=32, max_refinements=0)
    simulation.set_incident_mode(modes[0], amplitude=1.0-0.2j)
    result = simulation.solve(max_refinements=0)
    with np.load(Path(__file__).parent / "data/scattering_convention_baseline.npz") as baseline:
        np.testing.assert_allclose(result.coordinates, baseline[name+"_coordinates"], rtol=0, atol=1e-18)
        for key, values in [("E", result.E_total), ("H", result.H_total),
                            ("S11", result.S11), ("S21", result.S21), ("beta", modes[0].beta)]:
            expected = baseline[name+"_"+key].conj()
            scale = max(float(np.max(np.abs(expected))), 1e-12)
            np.testing.assert_allclose(values, expected, rtol=2e-8, atol=2e-10*scale)
        np.testing.assert_allclose(
            [result.reflected_power, result.transmitted_power, result.absorbed_power, result.incident_power],
            baseline[name+"_powers"], rtol=2e-8, atol=1e-12,
        )
        assert simulation.ky == pytest.approx(float(baseline[name+"_ky"]))
    assert result.solve_info["raw_absorbed_power"] > 0
    assert result.power_balance_error < 1e-3
    from fem_waveguide_scattering import load_result
    result.save(tmp_path / 'scattering.h5')
    loaded = load_result(tmp_path / 'scattering.h5')
    np.testing.assert_array_equal(loaded.E_total, result.E_total)
    np.testing.assert_array_equal(loaded.mesh_data.elements, result.mesh_data.elements)
    assert loaded.S21 == result.S21
    np.testing.assert_array_equal(loaded.modes[0].E, result.modes[0].E)
    loaded.save(tmp_path / 'scattering-again.h5')
    from matplotlib.figure import Figure
    assert isinstance(loaded.plot(component='Ey'), Figure)
