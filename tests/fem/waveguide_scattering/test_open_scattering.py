from cem_common import Material, SurfaceImpedance, materials, shapes
import numpy as np
import pytest

from fem_waveguide_scattering.scattering import WaveguideScatteringSolver2D


def _solve_open_slab(*, perturbed: bool):
    wavelength = 1.55e-6
    simulation = WaveguideScatteringSolver2D(frequency=299792458.0 / wavelength, angle=np.degrees(np.arcsin(0.1)), x_range=(-1.5e-06, 1.5e-06), z_range=(-2.5e-06, 2.5e-06), background_material=materials.Material(epsilon=1.44 ** 2, mu=1.0))
    simulation.add_rectangle(x_range=(-2.2e-07, 2.2e-07), z_range=simulation.z_range, background=True, name='core', material=materials.Material(epsilon=3.45 ** 2, mu=1.0))
    if perturbed:
        simulation.add_rectangle(x_range=(-2.2e-07, 2.2e-07), z_range=(-2.5e-07, 2.5e-07), name='perturbation', material=materials.Material(epsilon=3.46 ** 2, mu=1.0))
    simulation.add_pml(order=3, target_reflection=1e-07, thickness=3.5e-07, direction='x')
    simulation.add_pml(order=3, target_reflection=1e-07, thickness=6.5e-07, direction='z')
    simulation.mesh(wavelength_elements=9, refine_interfaces=False)
    modes = simulation.solve_modes(
        max_refinements=0,
        num_modes=1,
        neff_guess=3.2,
        num_elements=54,
    )
    simulation.set_incident_mode(modes[0])
    return simulation.solve(max_refinements=0)


@pytest.mark.gmsh
@pytest.mark.slow
def test_open_transverse_oblique_scattering_is_finite_and_balanced() -> None:
    result = _solve_open_slab(perturbed=True)

    assert np.isfinite((result.S11.real, result.S11.imag)).all()
    assert np.isfinite((result.S21.real, result.S21.imag)).all()
    assert np.linalg.norm(result.E_scattered) > 0.0
    assert result.solve_info["source_active_fraction"] > 0.0
    assert result.solve_info["left_projection_residual"] < 1e-2
    assert result.solve_info["right_projection_residual"] < 1e-2
    assert result.solve_info["independent_energy_residual"] < 1e-2
    assert result.power_balance_error < 1e-2


@pytest.mark.gmsh
@pytest.mark.slow
def test_unperturbed_open_guide_has_unit_transmission_and_no_radiation() -> None:
    result = _solve_open_slab(perturbed=False)

    assert np.linalg.norm(result.E_scattered) == 0.0
    assert abs(result.S11) < 1e-12
    assert result.S21 == pytest.approx(1.0 + 0.0j, abs=2e-11)
    assert result.transmission == pytest.approx(1.0, abs=2e-10)
    assert result.radiated_power / result.incident_power < 2e-10
    assert result.power_balance_error < 2e-10
