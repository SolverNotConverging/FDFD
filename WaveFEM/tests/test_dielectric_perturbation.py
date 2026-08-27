import numpy as np
import pytest

from wavefem.scattering import Scattering2D


def _solve(delta_eps: float):
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
        z=(-0.30e-6, 0.30e-6),
        eps=1.0 + delta_eps,
        name="weak_step",
    )
    simulation.add_pml(z=0.8e-6, order=3, target_reflection=1e-8)
    simulation.mesh(max_element_size=0.20e-6, wavelength_elements=8)
    modes = simulation.solve_modes(num_modes=1, neff_guess=1.0, num_elements=56)
    simulation.set_incident_mode(modes[0])
    return simulation.solve()


@pytest.mark.gmsh
@pytest.mark.slow
def test_physical_weak_scattering_field_and_reflected_power_scaling() -> None:
    first = _solve(2.0e-3)
    second = _solve(4.0e-3)

    field_ratio = np.linalg.norm(second.E_scattered) / np.linalg.norm(first.E_scattered)
    amplitude_ratio = abs(second.S11) / abs(first.S11)
    reflected_power_ratio = second.reflection / first.reflection

    assert field_ratio == pytest.approx(2.0, rel=3e-2)
    assert amplitude_ratio == pytest.approx(2.0, rel=3e-2)
    assert reflected_power_ratio == pytest.approx(4.0, rel=6e-2)
    assert first.power_balance_error < 2e-2
    assert second.power_balance_error < 2e-2
