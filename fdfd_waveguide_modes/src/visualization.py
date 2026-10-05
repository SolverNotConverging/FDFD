"""Dedicated waveguide viewer with simultaneous electric and magnetic fields."""
import numpy as np
from fdfd_common.errors import ConfigurationError
from fdfd_common.field_plot import draw_field_panel, effective_index_label


class WaveguideModeViewer:
    def __init__(self, result):
        from matplotlib import pyplot as plt
        from matplotlib.widgets import RadioButtons, Slider

        self.result = result
        self.mode = 0
        self.quantity = 'magnitude'
        self.components = tuple(name for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
                                if name in result.fields)
        columns = min(3, len(self.components))
        rows = int(np.ceil(len(self.components)/columns))
        self.figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(12, 8))
        self.figure.subplots_adjust(left=.08, right=.95, top=.85, bottom=.25,
                                    hspace=.55, wspace=.5)
        self.field_axes = tuple(axes.flat)
        self.colorbars = []
        self.quantity_control = RadioButtons(self.figure.add_axes((.04, .025, .2, .15)),
                                             ('magnitude', 'real', 'imag', 'phase'))
        self.quantity_control.ax.set_title('Field quantity')
        self.quantity_control.on_clicked(self._set_quantity)
        self.mode_control = None
        if len(result) > 1:
            self.mode_control = Slider(self.figure.add_axes((.4, .07, .48, .035)),
                                       'Mode', 0, len(result)-1, valstep=1, valinit=0)
            self.mode_control.on_changed(self._set_mode)
        self.figure._waveguide_mode_viewer = self
        self.draw()

    def _set_mode(self, value):
        self.mode = int(value)
        self.draw()

    def _set_quantity(self, value):
        self.quantity = value
        self.draw()

    def draw(self):
        for colorbar in self.colorbars:
            colorbar.remove()
        self.colorbars = []
        for ax, name in zip(self.field_axes, self.components):
            ax.clear()
            colorbar = draw_field_panel(self.result, ax, name, self.quantity, self.mode)
            if colorbar is not None:
                self.colorbars.append(colorbar)
        for ax in self.field_axes[len(self.components):]:
            ax.set_visible(False)
        self.figure.suptitle('Waveguide modes — ' + effective_index_label(self.result, self.mode))
        self.figure.canvas.draw_idle()


def show_result(result, *, block=True):
    from matplotlib import pyplot as plt
    if not isinstance(block, bool):
        raise ConfigurationError('block must be a boolean.')
    viewer = WaveguideModeViewer(result)
    plt.show(block=block)
    return viewer.figure
