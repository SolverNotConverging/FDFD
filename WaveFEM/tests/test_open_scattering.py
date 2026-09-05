import numpy as np
import pytest

from wavefem.scattering import Scattering2D


def _solve_open_slab(*, perturbed: bool):
    wavelength = 1.55e-6
    simulation = Scattering2D(
        wavelength=wavelength,
        angle=np.degrees(np.arcsin(0.10)),
        x_span=(-1.5e-6, 1.5e-6),
        z_span=(-2.5e-6, 2.5e-6),
        background_eps=1.44**2,
    )
    simulation.add_rectangle(
        x=(-0.22e-6, 0.22e-6),
        z="all",
        eps=3.45**2,
        background=True,
        name="core",
    )
    if perturbed:
        simulation.add_rectangle(
            x=(-0.22e-6, 0.22e-6),
            z=(-0.25e-6, 0.25e-6),
            eps=3.46**2,
            name="perturbation",
        )
    simulation.add_pml(
        x=0.35e-6,
        z=0.65e-6,
        order=3,
        target_reflection=1e-7,
    )
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
