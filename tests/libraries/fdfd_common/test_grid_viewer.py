"""Viewer controls, effective indices, and physical plotting coordinates."""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pytest

from fdfd_common.grid import GridData
from fdfd_periodic_modes import PeriodicModeSet
from fdfd_scattering import ScatteringResult
from fdfd_waveguide_modes import ModeSet


def result_for(result_type, family, axes, neff):
    coordinates = (np.array([1e-3, 2e-3]), np.array([4e-3, 5e-3, 6e-3]))
    count = max(1, len(neff))
    values = np.arange(1, 6 * count + 1).reshape(2, 3, count).astype(complex)
    return result_type(family, GridData(axes, ((0., 3e-3), (3e-3, 7e-3)), (2, 3)),
                       30e9, {'Ex': values, 'Ey': 2j * values},
                       {'Ex': coordinates, 'Ey': coordinates}, np.array(neff),
                       {'k0': 1.})


def test_scattering_viewer_has_no_mode_control_or_mode_title(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda **kwargs: None)
    result = result_for(ScatteringResult, 'fdfd_scattering', ('x', 'y'), [])
    result.fields['Hz'] = result.fields['Ex'].copy()
    result.field_coordinates['Hz'] = result.field_coordinates['Ex']
    figure = result.show(block=False)
    try:
        viewer = figure._scattering_viewer
        assert not hasattr(viewer, 'mode_control')
        assert len(viewer.field_axes) == 3
        assert 'Mode' not in figure._suptitle.get_text()
        viewer.quantity_control.set_active(2)
        for axis in viewer.field_axes:
            assert '(imag)' in axis.get_title()
        np.testing.assert_array_equal(viewer.field_axes[0].collections[0].get_array(),
                                      result.fields['Ex'][..., 0].imag.T)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(('result_type', 'family'), [
    (ModeSet, 'fdfd_waveguide_modes'),
    (PeriodicModeSet, 'fdfd_periodic_modes'),
])
def test_modal_viewer_updates_complex_neff_when_mode_changes(monkeypatch, result_type, family):
    monkeypatch.setattr(plt, 'show', lambda **kwargs: None)
    result = result_for(result_type, family, ('x', 'y'), [1.5-.02j, 2.1+.03j])
    figure = result.show(block=False)
    try:
        viewer = (figure._waveguide_mode_viewer if family == 'fdfd_waveguide_modes'
                  else figure._periodic_mode_viewer)
        assert len(viewer.components) == 2
        assert 'neff = 1.5-0.02j' in figure._suptitle.get_text()
        assert viewer.mode_control.valmin == 1
        viewer.mode_control.set_val(2)
        assert 'Mode 2: neff = 2.1+0.03j' in figure._suptitle.get_text()
        viewer.quantity_control.set_active(2)
        assert all('(imag)' in ax.get_title() for ax in viewer.field_axes[:2])
    finally:
        plt.close(figure)


def test_periodic_2d_plot_uses_z_horizontal_and_x_vertical():
    result = result_for(PeriodicModeSet, 'fdfd_periodic_modes', ('x', 'z'), [1.5-.02j])
    figure = result.plot(component='Ex')
    ax = figure.axes[0]
    assert ax.get_xlabel() == 'z (m)'
    assert ax.get_ylabel() == 'x (m)'
    mesh = ax.collections[0]
    np.testing.assert_array_equal(mesh.get_array(), result.fields['Ex'][..., 0].real)
    coordinates = mesh.get_coordinates()
    np.testing.assert_allclose(coordinates[0, :, 0], [3.5e-3, 4.5e-3, 5.5e-3, 6.5e-3])
    np.testing.assert_allclose(coordinates[:, 0, 1], [.5e-3, 1.5e-3, 2.5e-3])


@pytest.mark.parametrize(('result_type', 'family', 'axes'), [
    (ModeSet, 'fdfd_waveguide_modes', ('x', 'y')),
    (PeriodicModeSet, 'fdfd_periodic_modes', ('x', 'z')),
])
def test_modal_viewer_shows_fields_with_same_material_geometry(monkeypatch, result_type, family, axes):
    monkeypatch.setattr(plt, 'show', lambda **kwargs: None)
    result = result_for(result_type, family, axes, [1.5])
    if family == 'fdfd_periodic_modes':
        result.metadata['polarization'] = 'TM'
    epsilon = np.ones((3, 2, 3))
    epsilon[:, 1, :] = 4.
    result.metadata['material_background'] = dict(epsilon=epsilon, mu=np.ones_like(epsilon),
                                                conductor=np.zeros((2, 3), dtype=bool))
    for name in ('Ez', 'Hx', 'Hy', 'Hz'):
        result.fields[name] = result.fields['Ex'].copy()
        result.field_coordinates[name] = result.field_coordinates['Ex']
    figure = result.show(block=False)
    try:
        viewer = (figure._waveguide_mode_viewer if family == 'fdfd_waveguide_modes'
                  else figure._periodic_mode_viewer)
        expected_components = (('Ex', 'Ez', 'Hy') if family == 'fdfd_periodic_modes'
                               else ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'))
        assert viewer.components == expected_components
        assert len(viewer.field_axes) == len(expected_components)
        assert 'neff = 1.5+0j' in figure._suptitle.get_text()
        expected = (np.array([[1., 1., 1.], [2., 2., 2.]]) if axes == ('x', 'z')
                    else np.array([[1., 2.], [1., 2.], [1., 2.]]))
        for ax in viewer.field_axes:
            np.testing.assert_array_equal(ax.collections[1].get_array(), expected)
            assert ax.collections[0].get_alpha() == .85
            if axes == ('x', 'z'):
                assert ax.get_xlabel() == 'z (m)' and ax.get_ylabel() == 'x (m)'
    finally:
        plt.close(figure)


@pytest.mark.parametrize(('polarization', 'components'), [
    ('TE', ('Ey', 'Hx', 'Hz')), ('TM', ('Ex', 'Ez', 'Hy')),
])
def test_periodic_2d_viewer_shows_only_active_polarization(monkeypatch, polarization, components):
    monkeypatch.setattr(plt, 'show', lambda **kwargs: None)
    result = result_for(PeriodicModeSet, 'fdfd_periodic_modes', ('x', 'z'), [1.5])
    for name in ('Ez', 'Hx', 'Hy', 'Hz'):
        result.fields[name] = result.fields['Ex'].copy()
        result.field_coordinates[name] = result.field_coordinates['Ex']
    result.metadata['polarization'] = polarization
    figure = result.show(block=False)
    try:
        assert figure._periodic_mode_viewer.components == components
        assert len(figure._periodic_mode_viewer.field_axes) == 3
        assert 'Mode 1:' in figure._suptitle.get_text()
    finally:
        plt.close(figure)


def test_plot_mode_numbers_start_at_one():
    from fdfd_common.errors import ConfigurationError
    result = result_for(ModeSet, 'fdfd_waveguide_modes', ('x', 'y'), [1.5, 2.1])
    for number in (1, 2):
        figure = result.plot(component='Ex', mode=number)
        np.testing.assert_array_equal(figure.axes[0].collections[0].get_array(),
                                      result.fields['Ex'][..., number-1].real.T)
        assert f'mode {number},' in figure.axes[0].get_title()
    for number in (0, -1, 3):
        with pytest.raises(ConfigurationError):
            result.plot(mode=number)


def test_periodic_3d_slice_controls_change_physical_plane(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda **kwargs: None)
    coordinates = (np.array([1e-3, 2e-3]), np.array([4e-3, 5e-3, 6e-3]),
                   np.array([8e-3, 9e-3, 10e-3, 11e-3]))
    values = np.arange(24).reshape(2, 3, 4, 1).astype(complex)
    result = PeriodicModeSet('fdfd_periodic_modes',
        GridData(('x', 'y', 'z'), ((.5e-3, 2.5e-3), (3.5e-3, 6.5e-3), (7.5e-3, 11.5e-3)), (2, 3, 4)),
        30e9, {'Ex': values}, {'Ex': coordinates}, np.array([1.5-.02j]), {'k0': 1.})
    figure = result.show(block=False)
    try:
        viewer = figure._periodic_mode_viewer
        viewer.plane_control.set_active(1)
        assert viewer.field_axes[0].get_xlabel() == 'z (m)'
        assert viewer.field_axes[0].get_ylabel() == 'x (m)'
        viewer.slice_control.set_val(1.)
        np.testing.assert_array_equal(viewer.field_axes[0].collections[0].get_array(),
                                      np.abs(values[:, -1, :, 0]))
        assert 'y = 0.0065 m' in figure._suptitle.get_text()
    finally:
        plt.close(figure)


def test_band_viewer_has_polarization_and_normalized_frequency_controls(monkeypatch):
    from fdfd_band_structure.api import BandStructureResult
    monkeypatch.setattr(plt, 'show', lambda **kwargs: None)
    result = BandStructureResult(GridData(('x', 'y'), ((0., 1e-3), (0., 1e-3)), (2, 2)),
        np.array([[0., 1., 2.], [0., 0., 0.]]),
        {'TE': np.array([[1e9, 2e9, 3e9]]), 'TM': np.array([[2e9, 3e9, 4e9]])},
        {}, {'x_period': 1e-3, 'solve_info': {}})
    figure = result.show(block=False)
    try:
        viewer = figure._band_structure_viewer
        assert viewer.axis.lines[0].get_label() == 'TE 1'
        viewer.polarization_control.set_active(1)
        assert len(viewer.axis.lines) == 1
        viewer.scale_control.set_active(1)
        assert 'fa/c' in viewer.axis.get_ylabel()
        np.testing.assert_allclose(viewer.axis.lines[0].get_ydata(), result.normalized_frequencies['TE'][0])
    finally:
        plt.close(figure)
