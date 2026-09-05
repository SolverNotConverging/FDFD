from cem_common import Material, SurfaceImpedance, materials, shapes
import numpy as np
import pytest

from fem_waveguide_scattering.scattering import WaveguideScatteringSolver2D


_WAVELENGTH = 1.55e-6
_DELTA_EPS = 0.02
_SLAB_HALF_WIDTH = 0.30e-6
_PHYSICAL_Z_HALF_SPAN = 2.20e-6
_LEFT_REFERENCE = -1.25e-6


def _analytic_s11() -> complex:
    n = np.sqrt(1.0 + _DELTA_EPS)
    interface_r = (1.0 - n) / (1.0 + n)
    slab_phase = np.exp(
        -2j * (2.0 * np.pi / _WAVELENGTH) * n * (2.0 * _SLAB_HALF_WIDTH)
    )
    at_slab = interface_r * (1.0 - slab_phase) / (
        1.0 - interface_r**2 * slab_phase
    )
    distance = -_SLAB_HALF_WIDTH - _LEFT_REFERENCE
    return complex(
        at_slab * np.exp(-2j * (2.0 * np.pi / _WAVELENGTH) * distance)
    )


def _solve(pml_thickness: float, maximum_edge: float) -> complex:
    outer = _PHYSICAL_Z_HALF_SPAN + pml_thickness
    simulation = WaveguideScatteringSolver2D(frequency=299792458.0 / _WAVELENGTH, ky=0.0, x_range=(0.0, 1e-06), z_range=(-outer, outer), background_material=materials.Material(epsilon=1.0, mu=1.0), boundary=materials.PEC)
    simulation.add_rectangle(x_range=(0.0, 1e-06), z_range=(-_SLAB_HALF_WIDTH, _SLAB_HALF_WIDTH), material=materials.Material(epsilon=1.0 + _DELTA_EPS, mu=1.0))
    simulation.add_pml(order=3, target_reflection=1e-07, thickness=pml_thickness, direction='z')
    if pml_thickness < 3.0 * maximum_edge:
        with pytest.warns(RuntimeWarning, match="PML thickness"):
            simulation.mesh(max_element_size=maximum_edge, wavelength_elements=5)
    else:
        simulation.mesh(max_element_size=maximum_edge, wavelength_elements=5)
    modes = simulation.solve_modes(max_refinements=0, num_modes=1, neff_guess=1.0, num_elements=64)
    simulation.set_incident_mode(modes[0])
    return simulation.solve(max_refinements=0).S11


@pytest.mark.gmsh
@pytest.mark.slow
def test_s11_converges_as_z_pml_and_mesh_are_refined() -> None:
    exact = _analytic_s11()
    approximations = (
        _solve(0.35e-6, 0.28e-6),
        _solve(0.70e-6, 0.19e-6),
        _solve(1.05e-6, 0.13e-6),
    )
    errors = np.abs(np.asarray(approximations) - exact)
    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 3e-4


@pytest.mark.gmsh
@pytest.mark.slow
def test_s11_converges_under_mesh_refinement_at_fixed_pml() -> None:
    exact = _analytic_s11()
    approximations = (
        _solve(1.05e-6, 0.28e-6),
        _solve(1.05e-6, 0.19e-6),
        _solve(1.05e-6, 0.13e-6),
    )
    errors = np.abs(np.asarray(approximations) - exact)
    assert np.all(np.diff(errors) < 0.0), errors
    assert errors[-1] < 3e-4
