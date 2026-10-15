"""One plot_dispersion: complex effective indices per mode, connected or not."""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from fdfd import ConfigurationError, plot_dispersion


@pytest.mark.parametrize("scatter", [True, False])
def test_dispersion_has_real_and_signed_imaginary_values_per_mode(scatter):
    frequencies = np.array([30e9, 10e9, 20e9])
    neff = np.array([[2.5 + 0.06j, 1.5 - 0.03j], [1.1 - 0.01j, 2.1 + 0.02j], [1.3 - 0.02j, 2.3 + 0.04j]])
    figure = plot_dispersion(frequencies, neff, show=False, connect=not scatter)
    try:
        real_axis, imag_axis = figure.axes
        assert real_axis.get_ylabel() == "Re(neff)"
        assert imag_axis.get_ylabel() == "Im(neff)"
        assert imag_axis.get_xlabel() == "Frequency (GHz)"
        real_artists = real_axis.collections if scatter else real_axis.lines
        imag_artists = imag_axis.collections if scatter else imag_axis.lines
        assert len(real_artists) == len(imag_artists) == 2
        if scatter:
            assert not real_axis.lines and not imag_axis.lines
        for mode, (real, imag) in enumerate(zip(real_artists, imag_artists, strict=True), start=1):
            assert real.get_label() == imag.get_label() == f"Mode {mode}"
            if scatter:
                np.testing.assert_array_equal(real.get_facecolors(), imag.get_facecolors())
                np.testing.assert_array_equal(real.get_offsets()[:, 0], [10.0, 20.0, 30.0])
                np.testing.assert_array_equal(real.get_offsets()[:, 1], neff[[1, 2, 0], mode - 1].real)
                np.testing.assert_array_equal(imag.get_offsets()[:, 1], neff[[1, 2, 0], mode - 1].imag)
            else:
                assert real.get_color() == imag.get_color()
                np.testing.assert_array_equal(real.get_xdata(), [10.0, 20.0, 30.0])
                np.testing.assert_array_equal(real.get_ydata(), neff[[1, 2, 0], mode - 1].real)
                np.testing.assert_array_equal(imag.get_ydata(), neff[[1, 2, 0], mode - 1].imag)
    finally:
        plt.close(figure)


def test_single_mode_dispersion_opens_figure(monkeypatch):
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    figure = plot_dispersion([10e9, 20e9], [1.2 - 0.01j, 1.3 - 0.02j], connect=False)
    try:
        assert called == [True]
        assert all(len(axis.collections) == 1 and not axis.lines for axis in figure.axes)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("frequencies", "neff"),
    [
        ([], []),
        ([0.0], [[1.0]]),
        ([10e9, 20e9], [[1.0]]),
        ([10e9], [[np.nan]]),
        ([10e9], np.zeros((1, 2, 2))),
    ],
)
def test_dispersion_rejects_incompatible_sweep_data(frequencies, neff):
    with pytest.raises(ConfigurationError):
        plot_dispersion(frequencies, neff, show=False)
