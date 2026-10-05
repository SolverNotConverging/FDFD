"""Scattering fields retain Maxwell derivatives and the TF/SF boundary."""
import numpy as np
import pytest

from fdfd_scattering import ScatteringSolver2D, load_result
from fdfd_scattering.solver_2d import _ScatteringSolver2D


@pytest.mark.parametrize('polarization', ('TE', 'TM'))
def test_transverse_reconstruction_removes_incident_interface_jump(polarization):
    backend = _ScatteringSolver2D(3e9, 120e-3, 100e-3, 12, 10)
    backend.add_source(polarization=polarization, angle_deg=30.)
    backend.add_mask(2)
    inside = 1.-backend.Q.diagonal()
    scalar = (inside*backend.source).reshape(backend.Ny, backend.Nx)
    setattr(backend, 'Ez' if polarization == 'TE' else 'Hz', scalar)
    fields = backend.transverse_fields(polarization)
    DEX, DEY, DHX, DHY = backend._yeeder2d()
    eta0 = 1./(backend.c0*backend.eps0)
    if polarization == 'TE':
        expected = {'Hx': inside*1j*(DEY@backend.source)/eta0,
                    'Hy': -inside*1j*(DEX@backend.source)/eta0}
    else:
        expected = {'Ex': -inside*1j*eta0*(DHY@backend.source),
                    'Ey': inside*1j*eta0*(DHX@backend.source)}
    for name, values in expected.items():
        np.testing.assert_allclose(fields[name].ravel(), values, atol=1e-12)
        np.testing.assert_allclose(fields[name].ravel()[inside == 0], 0., atol=1e-12)


@pytest.mark.parametrize(('polarization', 'components', 'offsets'), [
    ('TE', {'Ez', 'Hx', 'Hy'}, {'Ez': (0., 0.), 'Hx': (0., .5), 'Hy': (.5, 0.)}),
    ('TM', {'Ex', 'Ey', 'Hz'}, {'Hz': (0., 0.), 'Ex': (0., -.5), 'Ey': (-.5, 0.)}),
])
def test_public_scattering_returns_three_staggered_fields(polarization, components, offsets, tmp_path):
    solver = ScatteringSolver2D(frequency=3e9, x_range=(-60e-3, 60e-3),
                                y_range=(-50e-3, 50e-3), polarization=polarization)
    solver.mesh(resolution=(12, 10))
    solver.add_source(angle=30.)
    solver.set_source_region(inset=20e-3)
    result = solver.solve()
    assert set(result.fields) == components
    restored = load_result(result.save(tmp_path/'scattering.h5'))
    for name, values in result.fields.items():
        assert values.shape == (12, 10, 1)
        assert np.isfinite(values).all()
        assert np.linalg.norm(values) > 0
        for axis, (coordinates, centers, step) in enumerate(zip(
                result.field_coordinates[name], result.mesh_data.coordinates, (10e-3, 10e-3))):
            np.testing.assert_allclose(coordinates, centers+offsets[name][axis]*step)
        np.testing.assert_array_equal(restored.fields[name], values)
        for actual, expected in zip(restored.field_coordinates[name], result.field_coordinates[name]):
            np.testing.assert_array_equal(actual, expected)
