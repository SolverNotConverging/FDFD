from cem_common import materials
from fem_waveguide_scattering.materials import Material as _internal_Material
from fem_waveguide_scattering.modes import CrossSection as _internal_CrossSection
from fem_waveguide_scattering.modes import ModeSolver as _internal_ModeSolver
import numpy as np
import pytest

import fem_waveguide_scattering as wf


@pytest.mark.gmsh
@pytest.mark.slow
def test_callback_material_device_runs_with_injected_mode_set() -> None:
    simulation = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.0, ky=0.0, x_range=((0.0, 1.0), (-2.0, 2.0))[0], z_range=((0.0, 1.0), (-2.0, 2.0))[1], boundary=materials.PEC)
    simulation.set_material_field(material=materials.SpatialMaterial(name="actual", epsilon=lambda x, z: 1.0 + 0.02 * (np.abs(z) < 0.2)), background_material=materials.SpatialMaterial(name="background", epsilon=lambda x: np.ones_like(x)))
    simulation.add_pml(order=3, target_reflection=1e-08, thickness=0.5, direction='z')
    simulation.mesh(max_element_size=0.12, wavelength_elements=8)

    cross_section = _internal_CrossSection(
        (0.0, 1.0),
        background=_internal_Material(),
        boundary="pec",
    )
    modes = _internal_ModeSolver(
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
