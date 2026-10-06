"""Scattering fields retain Maxwell derivatives and the TF/SF boundary."""
import numpy as np
import pytest

from fdfd_scattering import Material, ScatteringSolver2D, load_result, materials
from fdfd_scattering.solver_2d import _ScatteringSolver2D


@pytest.mark.parametrize('polarization', ('TE', 'TM'))
def test_transverse_reconstruction_removes_incident_interface_jump(polarization):
    backend = _ScatteringSolver2D(3e9, 120e-3, 100e-3, 12, 10)
    backend.add_source(polarization=polarization, angle_deg=30.)
    backend.add_mask(2)
    inside = 1.-backend.Q.diagonal()
    scalar = (inside*backend.source).reshape(backend.primary_shape)
    setattr(backend, 'Ez' if polarization == 'TE' else 'Hz', scalar)
    fields = backend.transverse_fields(polarization)
    DEX, DEY, DHX, DHY = backend._yeeder2d()
    eta0 = 1./(backend.c0*backend.eps0)
    if polarization == 'TE':
        expected = {'Hx': 1j*(DEY@backend.source)/eta0,
                    'Hy': -1j*(DEX@backend.source)/eta0}
    else:
        expected = {'Ex': -1j*eta0*(DHY@backend.source),
                    'Ey': 1j*eta0*(DHX@backend.source)}
    for name, values in expected.items():
        component_inside = 1-backend.field_masks[name].diagonal()
        np.testing.assert_allclose(fields[name].ravel(), component_inside*values, atol=1e-12)
        np.testing.assert_allclose(fields[name].ravel()[component_inside == 0], 0., atol=1e-12)


@pytest.mark.parametrize(('polarization', 'locations'), [
    ('TE', {'Ez': ('node', 'node'), 'Hx': ('node', 'cell'), 'Hy': ('cell', 'node')}),
    ('TM', {'Hz': ('cell', 'cell'), 'Ex': ('cell', 'node'), 'Ey': ('node', 'cell')}),
])
def test_public_scattering_returns_three_staggered_fields(polarization, locations, tmp_path):
    solver = ScatteringSolver2D(frequency=3e9, x_range=(-60e-3, 60e-3),
                                y_range=(-50e-3, 50e-3), polarization=polarization)
    solver.mesh(resolution=(12, 10))
    solver.add_source(angle=30.)
    solver.set_source_region(inset=20e-3)
    result = solver.solve()
    assert set(result.fields) == set(locations)
    restored = load_result(result.save(tmp_path/'scattering.h5'))
    for name, values in result.fields.items():
        assert values.shape == (*(n+(location == 'node') for n, location in zip((12, 10), locations[name])), 1)
        assert np.isfinite(values).all()
        assert np.linalg.norm(values) > 0
        for coordinates, bounds, n, location in zip(
                result.field_coordinates[name], result.mesh_data.bounds, (12, 10), locations[name]):
            expected = (np.linspace(*bounds, n+1) if location == 'node'
                        else bounds[0]+(np.arange(n)+.5)*(bounds[1]-bounds[0])/n)
            np.testing.assert_allclose(coordinates, expected, atol=1e-15)
        np.testing.assert_array_equal(restored.fields[name], values)
        for actual, expected in zip(restored.field_coordinates[name], result.field_coordinates[name]):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('polarization', ('TE', 'TM'))
@pytest.mark.parametrize('scatterer', ('dielectric', 'PEC'))
def test_cylinder_converges_to_cylindrical_wave_solution(polarization, scatterer):
    from scipy.special import jv, jvp, hankel2, h2vp

    errors = []
    for cells in (120, 240):
        solver = ScatteringSolver2D(frequency=3e9, x_range=(-150e-3, 150e-3),
            y_range=(-150e-3, 150e-3), polarization=polarization)
        solver.add_circle(center=(0., 0.), radius=25e-3,
                          material=materials.PEC if scatterer == 'PEC' else Material(name='cylinder', epsilon=4.))
        solver.add_pml(thickness=50e-3)
        solver.mesh(resolution=(cells, cells))
        solver.add_source()
        solver.set_source_region(inset=75e-3)
        result = solver.solve()
        component = 'Ez' if polarization == 'TE' else 'Hz'
        xx, yy = np.meshgrid(*result.field_coordinates[component], indexing='ij')
        radius, theta = np.hypot(xx, yy), np.arctan2(yy, xx)
        selected = (radius > 35e-3) & (radius < 65e-3)
        r, angle = radius[selected], theta[selected]
        k, a = solver._backend.k0, 25e-3
        # Scalar and (1/mu)*normal derivative are continuous for Ez;
        # Hz instead uses (1/epsilon)*normal derivative. Interior k is 2*k.
        q = 2. if polarization == 'TE' else .5
        exact = np.exp(-1j*k*r*np.cos(angle))
        for n in range(-12, 13):
            if scatterer == 'PEC':
                # Ez = 0 for TE; the normal derivative of Hz is zero for TM.
                coefficient = (-jv(n, k*a)/hankel2(n, k*a) if polarization == 'TE'
                               else -jvp(n, k*a)/h2vp(n, k*a))
            else:
                coefficient = (
                    q*jv(n, k*a)*jvp(n, 2*k*a)-jvp(n, k*a)*jv(n, 2*k*a)
                ) / (h2vp(n, k*a)*jv(n, 2*k*a)-q*hankel2(n, k*a)*jvp(n, 2*k*a))
            exact += (-1j)**n*coefficient*hankel2(n, k*r)*np.exp(1j*n*angle)
        calculated = result.fields[component][..., 0][selected]
        errors.append(np.linalg.norm(calculated-exact)/np.linalg.norm(exact))
        if scatterer == 'PEC':
            assert np.any(result.metadata['material_background']['conductor'])
            for name, field in result.fields.items():
                x, y = np.meshgrid(*result.field_coordinates[name], indexing='ij')
                # Strict interior samples have zero fields for both polarizations.
                np.testing.assert_array_equal(field[..., 0][x*x+y*y < (20e-3)**2], 0.)
    assert errors[1] < .65*errors[0]
    assert errors[1] < (.03 if scatterer == 'PEC' else .025)


def test_pec_object_must_be_inside_total_field_region():
    from fdfd_common.errors import ConfigurationError
    solver = ScatteringSolver2D(frequency=3e9, x_range=(-150e-3, 150e-3), y_range=(-150e-3, 150e-3))
    solver.add_circle(center=(0., 0.), radius=25e-3, material=materials.PEC)
    solver.mesh(resolution=(60, 60))
    solver.add_source()
    solver.set_source_region(inset=130e-3)
    with pytest.raises(ConfigurationError, match='enclose the PEC'):
        solver.solve()
