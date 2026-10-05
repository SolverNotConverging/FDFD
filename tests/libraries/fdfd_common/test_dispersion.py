"""Dispersion traces preserve complex effective indices and mode numbers."""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pytest

from fdfd_common.errors import ConfigurationError
from fdfd_periodic_modes import plot_dispersion as periodic_plot
from fdfd_waveguide_modes import plot_dispersion as waveguide_plot


@pytest.mark.parametrize('plot_dispersion', [periodic_plot, waveguide_plot])
def test_dispersion_has_real_and_signed_imaginary_trace_per_mode(plot_dispersion):
    frequencies = np.array([30e9, 10e9, 20e9])
    neff = np.array([[1.5-.03j, 2.5+.06j], [1.1-.01j, 2.1+.02j], [1.3-.02j, 2.3+.04j]])
    figure = plot_dispersion(frequencies, neff, show=False)
    try:
        real_axis, imag_axis = figure.axes
        assert real_axis.get_ylabel() == 'Re(neff)'
        assert imag_axis.get_ylabel() == 'Im(neff)'
        assert imag_axis.get_xlabel() == 'Frequency (GHz)'
        assert len(real_axis.lines) == len(imag_axis.lines) == 2
        for mode, (real, imag) in enumerate(zip(real_axis.lines, imag_axis.lines), start=1):
            assert real.get_label() == imag.get_label() == f'Mode {mode}'
            assert real.get_color() == imag.get_color()
            np.testing.assert_array_equal(real.get_xdata(), [10., 20., 30.])
            np.testing.assert_array_equal(real.get_ydata(), neff[[1, 2, 0], mode-1].real)
            np.testing.assert_array_equal(imag.get_ydata(), neff[[1, 2, 0], mode-1].imag)
    finally:
        plt.close(figure)


def test_single_mode_dispersion_opens_figure(monkeypatch):
    called = []
    monkeypatch.setattr(plt, 'show', lambda: called.append(True))
    figure = periodic_plot([10e9, 20e9], [1.2-.01j, 1.3-.02j])
    try:
        assert called == [True]
        assert all(len(axis.lines) == 1 for axis in figure.axes)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(('frequencies', 'neff'), [
    ([], []), ([0.], [[1.]]), ([10e9, 20e9], [[1.]]),
    ([10e9], [[np.nan]]), ([10e9], np.zeros((1, 2, 2))),
])
def test_dispersion_rejects_incompatible_sweep_data(frequencies, neff):
    with pytest.raises(ConfigurationError):
        periodic_plot(frequencies, neff, show=False)
