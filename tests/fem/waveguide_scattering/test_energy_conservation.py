from cem_common import Material, SurfaceImpedance, materials, shapes
import pytest

from fem_waveguide_scattering.scattering import WaveguideScatteringSolver2D


@pytest.mark.gmsh
@pytest.mark.slow
def test_compact_loss_is_positive_and_closes_the_power_balance() -> None:
    simulation = WaveguideScatteringSolver2D(frequency=299792458.0 / 1.55e-06, ky=0.0, x_range=(0.0, 1e-06), z_range=(-3e-06, 3e-06), background_material=materials.Material(epsilon=1.0, mu=1.0), boundary=materials.PEC)
    simulation.add_rectangle(x_range=(0.0, 1e-06), z_range=(-3.5e-07, 3.5e-07), name='lossy_insert', material=materials.Material(epsilon=1.02 - 0.01j, mu=1.0))
    simulation.add_pml(order=3, target_reflection=1e-08, thickness=9e-07, direction='z')
    simulation.mesh(max_element_size=0.13e-6, wavelength_elements=10)
    modes = simulation.solve_modes(max_refinements=0, num_modes=1, neff_guess=1.0, num_elements=64)
    simulation.set_incident_mode(modes[0])

    result = simulation.solve(max_refinements=0)

    assert result.absorbed_power > 0.0
    assert result.absorbed_power / result.incident_power > 1e-3
    assert result.solve_info["raw_absorbed_power"] > 0.0
    assert result.solve_info["independent_energy_residual"] < 2e-3
    assert result.power_balance_error < 2e-3
