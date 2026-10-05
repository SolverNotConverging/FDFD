"""Waveguide searches start above the materials and return fundamental modes first."""
import numpy as np
import pytest
from fdfd_waveguide_modes import Material, materials, ModeSolver1D, ModeSolver2D


@pytest.mark.parametrize('dimension', [1, 2])
def test_guess_bounds_anisotropic_materials_and_permeability(dimension):
    background = Material(name='background', epsilon=(2., 5., 3.), mu=(4., 1., 2.))
    core = Material(name='core', epsilon=(9.-.5j, 2., 3.), mu=(2., 3., 1.))
    if dimension == 1:
        solver = ModeSolver1D(frequency=30e9, x_range=20e-3, background_material=background)
        solver.add_layer(x_range=(5e-3, 10e-3), material=core)
        solver.add_layer(x_range=(0., 1e-3), material=materials.PEC)
    else:
        solver = ModeSolver2D(frequency=30e9, x_range=20e-3, y_range=16e-3, background_material=background)
        solver.add_rectangle(x_range=(5e-3, 10e-3), y_range=(5e-3, 10e-3), material=core)
        solver.add_rectangle(x_range=(0., 1e-3), y_range=(0., 16e-3), material=materials.PEC)
    bound = np.sqrt(abs(9.-.5j)*3.)
    for requested in (None, .5, bound):
        np.testing.assert_allclose(solver._resolve_neff_guess(requested), 1.01*bound)
    assert solver._resolve_neff_guess(10.) == 10.


def test_low_guess_returns_same_fundamental_spectrum_as_high_guess():
    width, cells, frequency = 20e-3, 80, 30e9
    step = width/cells
    solver = ModeSolver1D(frequency=frequency, x_range=(-step, width+step),
                          background_material=Material(name='fill', epsilon=4., mu=3.))
    solver.add_layer(x_range=(-step, 0.), material=materials.PEC)
    solver.add_layer(x_range=(width, width+step), material=materials.PEC)
    solver.mesh(resolution=cells+2, subpixels=1)
    automatic = solver.solve(num_modes=3, polarization='TE')
    low = solver.solve(num_modes=3, polarization='TE', neff_guess=.1)
    high = solver.solve(num_modes=3, polarization='TE', neff_guess=5.)
    assert automatic.solve_info['neff_guess'] > np.sqrt(12.)
    assert low.solve_info['neff_guess'] == automatic.solve_info['neff_guess']
    assert high.solve_info['neff_guess'] == 5.
    np.testing.assert_allclose(low.neff, automatic.neff, atol=1e-10)
    np.testing.assert_allclose(high.neff, automatic.neff, atol=1e-10)
    assert np.all(np.diff(automatic.neff.real) < 0)
    k0 = 2*np.pi*frequency/299792458.
    expected = np.sqrt(12.-(np.arange(1, 4)*np.pi/(k0*width))**2)
    np.testing.assert_allclose(automatic.neff.real, expected, rtol=2e-3)


def test_vector_modes_are_returned_in_descending_neff_order():
    solver = ModeSolver2D(frequency=30e9, x_range=20e-3, y_range=16e-3,
                          background_material=Material(name='fill', epsilon=4., mu=3.))
    solver.mesh(resolution=(18, 14))
    automatic = solver.solve(num_modes=4)
    low = solver.solve(num_modes=4, neff_guess=.1)
    assert automatic.solve_info['neff_guess'] > np.sqrt(12.)
    assert np.all(np.diff(automatic.neff.real) <= 0)
    np.testing.assert_allclose(low.neff, automatic.neff, rtol=1e-10, atol=1e-10)
