"""Dedicated photonic-band viewer with polarization and frequency controls."""
import numpy as np
from fdfd_common.errors import ConfigurationError


class BandStructureViewer:
    def __init__(self, result):
        from matplotlib import pyplot as plt
        from matplotlib.widgets import CheckButtons, RadioButtons

        self.result = result
        self.quantity = 'real'
        self.scale = 'GHz'
        self.polarizations = tuple(result.frequencies)
        self.selected = set(self.polarizations)
        self.figure, self.axis = plt.subplots(figsize=(10, 7))
        self.figure.subplots_adjust(left=.3, bottom=.12, top=.9)
        self.polarization_control = CheckButtons(self.figure.add_axes((.03, .7, .18, .15)),
                                                 self.polarizations, [True]*len(self.polarizations))
        self.polarization_control.ax.set_title('Polarization')
        self.quantity_control = RadioButtons(self.figure.add_axes((.03, .4, .18, .2)),
                                             ('real', 'imag', 'magnitude'))
        self.quantity_control.ax.set_title('Frequency quantity')
        self.scale_control = RadioButtons(self.figure.add_axes((.03, .12, .18, .16)),
                                          ('GHz', 'normalized'))
        self.scale_control.ax.set_title('Frequency scale')
        self.polarization_control.on_clicked(self._set_polarization)
        self.quantity_control.on_clicked(self._set_quantity)
        self.scale_control.on_clicked(self._set_scale)
        self.figure._band_structure_viewer = self
        self.draw()

    def _set_polarization(self, name):
        if name in self.selected:
            self.selected.remove(name)
        else:
            self.selected.add(name)
        self.draw()

    def _set_quantity(self, value):
        self.quantity = value
        self.draw()

    def _set_scale(self, value):
        self.scale = value
        self.draw()

    def draw(self):
        self.axis.clear()
        operation = {'real': np.real, 'imag': np.imag, 'magnitude': np.abs}[self.quantity]
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.result.beta_path, axis=1), axis=0))]
        frequencies = (self.result.normalized_frequencies if self.scale == 'normalized'
                       else {key: value/1e9 for key, value in self.result.frequencies.items()})
        for name in self.polarizations:
            if name in self.selected:
                for index, values in enumerate(frequencies[name]):
                    self.axis.plot(distance, operation(values), label=f'{name} {index+1}')
        unit = 'fa/c' if self.scale == 'normalized' else 'GHz'
        self.axis.set(xlabel='Distance along Bloch path (rad/m)',
                      ylabel=f'Frequency ({self.quantity}, {unit})', title='Photonic band structure')
        if self.selected:
            self.axis.legend()
        self.axis.grid(alpha=.25)
        self.figure.canvas.draw_idle()


def show_result(result, *, block=True):
    from matplotlib import pyplot as plt
    if not isinstance(block, bool):
        raise ConfigurationError('block must be a boolean.')
    viewer = BandStructureViewer(result)
    plt.show(block=block)
    return viewer.figure
