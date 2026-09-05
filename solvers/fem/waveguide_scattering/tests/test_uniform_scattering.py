import numpy as np
import pytest

from fem_waveguide_scattering.scattering import WaveguideScatteringSolver2D


@pytest.mark.gmsh
@pytest.mark.parametrize("element_order", [1, 2])
def test_uniform_pec_guide_has_zero_scattering_and_unit_transmission(element_order) -> None:
    simulation = WaveguideScatteringSolver2D(frequency=299792458.0 / 1.55e-06, ky=0.0, x_range=(0.0, 1e-06), z_range=(-3e-06, 3e-06), background_epsilon=1.0, transverse_boundary='pec')
    simulation.add_pml(order=3, target_reflection=1e-07, thickness=8e-07, direction='z')
    simulation.mesh(max_element_size=2.5e-07, wavelength_elements=6, element_order=element_order)
    modes = simulation.solve_modes(
        max_refinements=0,
        num_modes=1, neff_guess=1.0, num_elements=48
    )
    simulation.set_incident_mode(modes[0], amplitude=1e-13)

    result = simulation.solve(max_refinements=0)
    assert result.solve_info["element_order"] == element_order

    assert np.linalg.norm(result.E_scattered) == 0.0
    assert abs(result.S11) < 1e-12
    assert result.S21 == pytest.approx(1.0 + 0.0j, abs=2e-11)
    assert result.reflection < 1e-20
    assert result.transmission == pytest.approx(1.0, abs=2e-11)
    assert result.power_balance_error < 2e-10
    assert result.solve_info["length_scale"] == pytest.approx(1.0 / simulation.k0)
