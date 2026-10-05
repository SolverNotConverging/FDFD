"""Three-component TE/TM scattering viewer."""
from fdfd_common.errors import ConfigurationError
from fdfd_common.field_plot import draw_field_panel


class ScatteringViewer:
    def __init__(self, result):
        from matplotlib import pyplot as plt
        from matplotlib.widgets import RadioButtons

        self.result = result
        self.components = tuple(result.fields)
        self.quantity = 'magnitude'
        self.figure, axes = plt.subplots(1, len(self.components), squeeze=False, figsize=(13, 6))
        self.figure.subplots_adjust(left=.08, right=.94, top=.88, bottom=.3, wspace=.45)
        self.field_axes = tuple(axes.flat)
        self.colorbars = []
        self.quantity_control = RadioButtons(self.figure.add_axes((.04, .02, .2, .18)),
                                             ('magnitude', 'real', 'imag', 'phase'))
        self.quantity_control.ax.set_title('Field quantity')
        self.quantity_control.on_clicked(self._set_quantity)
        self.figure._scattering_viewer = self
        self.draw()

    def _set_quantity(self, value):
        self.quantity = value
        self.draw()

    def draw(self):
        for colorbar in self.colorbars:
            colorbar.remove()
        self.colorbars = []
        for ax, component in zip(self.field_axes, self.components):
            ax.clear()
            colorbar = draw_field_panel(self.result, ax, component, self.quantity)
            if colorbar is not None:
                self.colorbars.append(colorbar)
        polarization = self.result.solve_info.get('polarization', '')
        self.figure.suptitle(f'Scattering {polarization} — {self.result.frequency/1e9:.6g} GHz')
        self.figure.canvas.draw_idle()


def show_result(result, *, block=True):
    from matplotlib import pyplot as plt
    if not isinstance(block, bool):
        raise ConfigurationError('block must be a boolean.')
    viewer = ScatteringViewer(result)
    plt.show(block=block)
    return viewer.figure
