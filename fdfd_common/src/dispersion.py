"""Dispersion plots shared by waveguide and periodic mode solvers."""
import numpy as np
from .errors import ConfigurationError


def plot_dispersion(frequencies, neff, *, show=True):
    """Plot Re(neff) and Im(neff), with a trace per mode on each panel.

    Frequencies are in hertz. ``neff`` has shape (frequencies, modes), with
    each column following the same mode through the supplied sweep.
    Return the Matplotlib figure; ``show=False`` supports saving without a window.
    """
    from matplotlib import pyplot as plt

    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(neff, dtype=complex)
    if values.ndim == 1:
        values = values[:, None]
    if (frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(frequencies).all()
            or np.any(frequencies <= 0) or values.ndim != 2
            or values.shape[0] != len(frequencies) or not values.shape[1]
            or not np.isfinite(values).all()):
        raise ConfigurationError('Use positive frequencies in Hz and finite neff with shape (frequencies, modes).')
    if not isinstance(show, bool):
        raise ConfigurationError('show must be a boolean.')
    order = np.argsort(frequencies, kind='stable')
    if show:
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    else:
        from matplotlib.figure import Figure
        figure = Figure(figsize=(8, 7))
        axes = figure.subplots(2, 1, sharex=True)
    for index in range(values.shape[1]):
        line, = axes[0].plot(frequencies[order]/1e9, values[order, index].real,
                            'o-', label=f'Mode {index+1}')
        axes[1].plot(frequencies[order]/1e9, values[order, index].imag,
                     'o-', color=line.get_color(), label=f'Mode {index+1}')
    axes[0].set(ylabel='Re(neff)', title='Mode dispersion')
    axes[1].set(xlabel='Frequency (GHz)', ylabel='Im(neff)')
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    figure.tight_layout()
    if show:
        plt.show()
    return figure
