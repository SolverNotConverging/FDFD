import pytest

from wavefem.scattering import Scattering2D


@pytest.mark.gmsh
@pytest.mark.slow
def test_compact_loss_is_positive_and_closes_the_power_balance() -> None:
    simulation = Scattering2D(
        wavelength=1.55e-6,
        ky=0.0,
        x_span=(0.0, 1.0e-6),
        z_span=(-3.0e-6, 3.0e-6),
        background_eps=1.0,
        transverse_boundary="pec",
    )
    simulation.add_rectangle(
        x=(0.0, 1.0e-6),
        z=(-0.35e-6, 0.35e-6),
        eps=1.02 + 0.01j,
        name="lossy_insert",
    )
    simulation.add_pml(z=0.9e-6, order=3, target_reflection=1e-8)
    simulation.mesh(max_element_size=0.13e-6, wavelength_elements=10)
    modes = simulation.solve_modes(max_refinements=0, num_modes=1, neff_guess=1.0, num_elements=64)
    simulation.set_incident_mode(modes[0])

    result = simulation.solve(max_refinements=0)

    assert result.absorbed_power > 0.0
    assert result.absorbed_power / result.incident_power > 1e-3
    assert result.solve_info["raw_absorbed_power"] > 0.0
    assert result.solve_info["independent_energy_residual"] < 2e-3
    assert result.power_balance_error < 2e-3
