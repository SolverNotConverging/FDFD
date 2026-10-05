"""Scattering field display: real, imaginary, magnitude, and phase together."""
from fdfd_common.errors import ConfigurationError
from fdfd_common.field_plot import draw_field_panel


class ScatteringViewer:
    def __init__(self, result):
        from matplotlib import pyplot as plt

        self.result = result
        self.component = next(iter(result.fields))
        self.quantities = ('real', 'imag', 'magnitude', 'phase')
        self.figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        self.figure.subplots_adjust(left=.08, right=.94, top=.88, bottom=.08,
                                    hspace=.4, wspace=.4)
        self.field_axes = tuple(axes.flat)
        for ax, quantity in zip(self.field_axes, self.quantities):
            draw_field_panel(result, ax, self.component, quantity)
        self.figure.suptitle(f'Scattering — {self.component}, {result.frequency/1e9:.6g} GHz')
        self.figure._scattering_viewer = self


def show_result(result, *, block=True):
    from matplotlib import pyplot as plt
    if not isinstance(block, bool):
        raise ConfigurationError('block must be a boolean.')
    viewer = ScatteringViewer(result)
    plt.show(block=block)
    return viewer.figure
