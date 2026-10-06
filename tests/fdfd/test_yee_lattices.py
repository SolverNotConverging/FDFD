"""Analytic Yee locations, constitutive sampling and discrete plane waves."""
import numpy as np
import pytest
from fdfd_scattering.solver_2d import _ScatteringSolver2D
from fdfd_band_structure.solver_2d import _BandStructureSolver2D
from fdfd_periodic_modes import PeriodicModeSolver2D, PeriodicModeSolver3D
from fdfd_waveguide_modes import ModeSolver1D, ModeSolver2D


@pytest.mark.parametrize('polarization', ('TE', 'TM'))
def test_scattering_discrete_vacuum_wave_has_no_scattered_field(polarization):
    solver = _ScatteringSolver2D(3e9, 200e-3, 150e-3, 40, 30)
    solver.add_source(polarization=polarization)
    # Choose the exact Yee dispersion relation; this isolates the operator
    # and TF/SF interface from continuous-wave discretization error.
    kx = 2*np.arcsin(solver.k0*solver.dx/2)/solver.dx
    solver.source = np.exp(-1j*kx*solver.X).ravel()
    solver.set_total_field_region(30e-3)
    operation = solver.solve_total_field_TE if polarization == 'TE' else solver.solve_total_field_TM
    operation()
    fields = solver.transverse_fields(polarization)
    eta = 1/(solver.c0*solver.eps0)
    factors = {'Ez': 1., 'Hx': 0., 'Hy': -1/eta} if polarization == 'TE' else {'Hz': 1., 'Ex': 0., 'Ey': eta}
    for name, field in fields.items():
        x, y = solver.coordinates[name]
        wave = np.exp(-1j*kx*x)[None, :]*np.ones((len(y), 1))
        inside = (1-solver.field_masks[name].diagonal()).reshape(field.shape)
        np.testing.assert_allclose(field, factors[name]*wave*inside, atol=1e-9, rtol=1e-10)


def test_band_materials_are_sampled_at_six_distinct_yee_locations():
    solver = _BandStructureSolver2D(a=12e-3, b=10e-3, Nx=6, Ny=5)
    solver.ER2[:] = 2+solver.X2/12e-3+2*solver.Y2/10e-3
    solver.UR2[:] = 3+2*solver.X2/12e-3+solver.Y2/10e-3
    offsets = {'ERxx': (.5, 0.), 'ERyy': (0., .5), 'ERzz': (0., 0.),
               'URxx': (0., .5), 'URyy': (.5, 0.), 'URzz': (.5, .5)}
    for name, (ox, oy) in offsets.items():
        x = -solver.a/2+(np.arange(solver.Nx)+ox)*solver.dx
        y = -solver.b/2+(np.arange(solver.Ny)+oy)*solver.dy
        xx, yy = np.meshgrid(x, y, indexing='ij')
        expected = 2+xx/12e-3+2*yy/10e-3 if name.startswith('ER') else 3+2*xx/12e-3+yy/10e-3
        np.testing.assert_allclose(solver._yee_tensors()[name], expected)


def test_lossy_band_frequencies_obey_discrete_bloch_dispersion():
    eps, mu = 2.25-.09j, 1.3-.02j
    solver = _BandStructureSolver2D(a=1., b=1.2, Nx=6, Ny=5, background_er=eps, background_ur=mu)
    beta = np.array([[.3], [.2]])
    result = solver.compute_band_structure(beta, num_bands=1, eig_sigma=.1)
    k2 = (2*np.sin(beta[0, 0]*solver.dx/2)/solver.dx)**2 + (2*np.sin(beta[1, 0]*solver.dy/2)/solver.dy)**2
    for pol in ('TE', 'TM'):
        np.testing.assert_allclose(result.eigenvalues[pol], [[k2/(eps*mu)]], rtol=1e-10)


@pytest.mark.parametrize('dimension', (2, 3))
def test_periodic_materials_and_coordinates_share_the_z_lattice(dimension):
    settings = dict(frequency=20e9, x_range=10e-3, z_range=8e-3)
    solver = (PeriodicModeSolver2D(**settings) if dimension == 2 else
              PeriodicModeSolver3D(y_range=6e-3, **settings))
    resolution = (6, 8) if dimension == 2 else (6, 5, 8)
    solver.mesh(resolution=resolution)
    backend = solver._backend
    z_cells = (np.arange(8)+.5)*8e-3/8
    cell_profile = 3+np.sin(2*np.pi*z_cells/8e-3)
    for comp in ('xx', 'yy', 'zz'):
        for prefix, d3 in (('eps', 'Er'), ('mu', 'Mr')):
            name = 'cell_'+prefix+'_r_'+comp if dimension == 2 else 'cell_'+d3+comp+'_3D'
            getattr(backend, name)[:] = cell_profile
    backend.update_component_materials()
    result = solver.solve(num_modes=1, neff_guess=.8, eigensolver_tolerance=1e-9)
    for field in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        z_node = field in ('Ex', 'Ey', 'Hz')
        expected_z = (np.arange(8)+(0. if z_node else .5))*8e-3/8
        np.testing.assert_allclose(result.field_coordinates[field][-1], expected_z, atol=1e-15)
        profile = 3+np.sin(2*np.pi*expected_z/8e-3)*(np.cos(np.pi/8) if z_node else 1.)
        if dimension == 2:
            material = getattr(backend, ('eps_r_' if field[0] == 'E' else 'mu_r_')+field[1].lower()*2)
        else:
            material = getattr(backend, ('Er' if field[0] == 'E' else 'Mr')+field[1].lower()*2+'_3D')
        # Outer-wall replacement tensors are excluded; compare interior traces.
        interior = material[1:-1] if dimension == 2 else material[1:-1, 1:-1]
        np.testing.assert_allclose(interior, np.broadcast_to(profile, interior.shape))


@pytest.mark.parametrize('dimension', (1, 2))
def test_waveguide_fields_keep_complete_transverse_yee_lattices(dimension):
    solver = (ModeSolver1D(frequency=20e9, x_range=10e-3) if dimension == 1 else
              ModeSolver2D(frequency=20e9, x_range=10e-3, y_range=8e-3))
    solver.mesh(resolution=(12,) if dimension == 1 else (12, 10))
    result = solver.solve(num_modes=1)
    offsets = {'Ex': (.5, 0.), 'Ey': (0., .5), 'Ez': (0., 0.),
               'Hx': (0., .5), 'Hy': (.5, 0.), 'Hz': (.5, .5)}
    for name, field in result.fields.items():
        for n, coordinates, cells, bounds, offset in zip(field.shape[:-1], result.field_coordinates[name],
                result.mesh_data.resolution, result.mesh_data.bounds, offsets[name]):
            assert n == cells+(offset == 0.)
            expected = bounds[0]+(np.arange(n)+offset)*(bounds[1]-bounds[0])/cells
            np.testing.assert_allclose(coordinates, expected, atol=1e-15)
