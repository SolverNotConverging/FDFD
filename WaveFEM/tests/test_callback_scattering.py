import numpy as np
import pytest

import wavefem as wf


@pytest.mark.gmsh
@pytest.mark.slow
def test_callback_material_device_runs_with_injected_mode_set() -> None:
    simulation = wf.Scattering2D.from_material_function(
        wavelength=1.0,
        ky=0.0,
        domain=((0.0, 1.0), (-2.0, 2.0)),
        eps_r=lambda x, z: 1.0 + 0.02 * (np.abs(z) < 0.2),
        eps_background=lambda x: np.ones_like(x),
        transverse_boundary="pec",
    )
    simulation.add_pml(z=0.5, order=3, target_reflection=1e-8)
    simulation.mesh(max_element_size=0.12, wavelength_elements=8)

    cross_section = wf.CrossSection(
        (0.0, 1.0),
        background=wf.Material(),
        boundary="pec",
    )
    modes = wf.ModeSolver(
        cross_section,
        wavelength=1.0,
        ky=0.0,
        num_elements=32,
    ).solve(max_refinements=0, num_modes=1, neff_guess=np.sqrt(0.75))
    bound = simulation.set_modes(modes)
    simulation.set_incident_mode(bound[0])

    result = simulation.solve(max_refinements=0)

    assert np.linalg.norm(result.E_scattered) > 0.0
    assert abs(result.S11) > 0.0
    assert np.isfinite(result.S21)
    assert result.power_balance_error < 2e-2
