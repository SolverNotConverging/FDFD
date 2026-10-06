"""Periodic mode samples need not keep their order across frequencies."""
from fdfd_common.dispersion import plot_dispersion as _plot_dispersion


def plot_dispersion(frequencies, neff, *, show=True):
    """Scatter Re(neff) and Im(neff) without connecting exchanging modes.

    Frequencies are in Hz; neff has shape (frequencies, modes). Mode numbers
    refer to the returned order at each frequency, rather than tracked branches.
    """
    return _plot_dispersion(frequencies, neff, show=show, connect=False)
