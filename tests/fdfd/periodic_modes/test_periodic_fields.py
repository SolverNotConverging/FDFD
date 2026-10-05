"""Complete periodic fields obey the longitudinal Maxwell equations."""
import numpy as np
import pytest

from fdfd_periodic_modes import Material, PeriodicModeSolver2D, load_result, materials


@pytest.mark.parametrize('polarization', ['TE', 'TM'])
def test_all_periodic_components_and_longitudinal_maxwell_equation(polarization, tmp_path):
    solver = PeriodicModeSolver2D(
        frequency=30e9, x_range=8e-3, z_range=6e-3, polarization=polarization,
        background_material=Material(name='fill', epsilon=2.25-.02j),
    )
    solver.mesh(resolution=(10, 8))
    result = solver.solve(num_modes=2, neff_guess=1.4)
    backend = solver._backend
    assert set(result.fields) == {'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'}
    for name, values in result.fields.items():
        assert values.shape == (*getattr(backend, 'shape_'+name.lower()), 2)
        assert tuple(map(len, result.field_coordinates[name])) == values.shape[:-1]
        assert np.isfinite(values).all()
    if polarization == 'TM':
        derivative = backend.DHX_HY_TO_EZ @ backend.Hy
        field = 1j * backend.omega * backend.epsilon0 * backend.eps_r_zz.ravel(order='F')[:, None] * backend.Ez
        inactive = ('Ey', 'Hx', 'Hz')
    else:
        derivative = backend.DEX_EY_TO_HZ @ backend.Ey
        field = -1j * backend.omega * backend.mu0 * backend.mu_r_zz.ravel(order='F')[:, None] * backend.Hz
        inactive = ('Ex', 'Ez', 'Hy')
    assert np.linalg.norm(derivative) > 0
    np.testing.assert_allclose(field, derivative, rtol=1e-12, atol=1e-12)
    for name in inactive:
        assert not np.any(result.fields[name])
    restored = load_result(result.save(tmp_path / 'modes.h5'))
    for name in result.fields:
        np.testing.assert_array_equal(restored.fields[name], result.fields[name])
    for name in ('epsilon', 'mu', 'conductor'):
        np.testing.assert_array_equal(restored.metadata['material_background'][name],
                                      result.metadata['material_background'][name])


@pytest.mark.parametrize(('polarization', 'boundary', 'name', 'mask'), [
    ('TM', materials.PEC, 'Ez', 'pec_zz_mask'),
    ('TE', materials.PMC, 'Hz', 'pmc_zz_mask'),
])
def test_longitudinal_field_is_zero_on_conductor_constraints(polarization, boundary, name, mask):
    solver = PeriodicModeSolver2D(frequency=30e9, x_range=8e-3, z_range=6e-3,
                                polarization=polarization)
    solver.add_rectangle(x_range=(0., 800e-6), z_range=(0., 6e-3), material=boundary)
    solver.mesh(resolution=(10, 8), subpixels=1)
    result = solver.solve(num_modes=1, neff_guess=.8)
    constrained = getattr(solver._backend, mask)
    assert np.any(constrained)
    np.testing.assert_array_equal(result.fields[name][constrained], 0.)


def test_periodic_3d_returns_all_six_fields_with_material_background():
    from fdfd_periodic_modes import PeriodicModeSolver3D
    solver = PeriodicModeSolver3D(frequency=30e9, x_range=8e-3, y_range=7e-3, z_range=6e-3,
        background_material=Material(name='fill', epsilon=2.25-.02j))
    solver.mesh(resolution=(4, 4, 4))
    result = solver.solve(num_modes=2, neff_guess=1.3, eigensolver='eigs')
    backend = solver._backend
    assert set(result.fields) == {'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'}
    for name, values in result.fields.items():
        assert values.shape == (*getattr(backend, 'shape_'+name.lower()), 2)
        assert np.isfinite(values).all()
    def flat(name):
        return np.column_stack([result.fields[name][..., i].ravel(order='F') for i in range(2)])
    electric_curl = backend.DHX_HY_TO_EZ @ flat('Hy') - backend.DHY_HX_TO_EZ @ flat('Hx')
    magnetic_curl = backend.DEX_EY_TO_HZ @ flat('Ey') - backend.DEY_EX_TO_HZ @ flat('Ex')
    np.testing.assert_allclose(1j*backend.omega*backend.epsilon0 *
        backend.Erzz_3D.ravel(order='F')[:, None]*flat('Ez'), electric_curl, atol=1e-12)
    np.testing.assert_allclose(-1j*backend.omega*backend.mu0 *
        backend.Mrzz_3D.ravel(order='F')[:, None]*flat('Hz'), magnetic_curl, atol=1e-12)
    assert result.metadata['material_background']['epsilon'].shape == (3, 4, 4, 4)
