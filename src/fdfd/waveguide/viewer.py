"""Dedicated waveguide viewer with simultaneous electric and magnetic fields."""

import numpy as np

from ..core.plotting import ModeDropdown, draw_field_panel, effective_index_label, show_figure


class WaveguideModeViewer:
    def __init__(self, result):
        from matplotlib import pyplot as plt
        from matplotlib.widgets import RadioButtons

        self.result = result
        self.mode = 1
        self.quantity = "magnitude"
        self.components = tuple(
            name for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") if name in result.fields
        )
        columns = min(3, len(self.components))
        rows = int(np.ceil(len(self.components) / columns))
        self.figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(12, 8))
        self.figure.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.25, hspace=0.55, wspace=0.5)
        self.field_axes = tuple(axes.flat)
        self.colorbars = []
        self.quantity_control = RadioButtons(
            self.figure.add_axes((0.04, 0.025, 0.2, 0.15)), ("magnitude", "real", "imag", "phase")
        )
        self.quantity_control.ax.set_title("Field quantity")
        self.quantity_control.on_clicked(self._set_quantity)
        self.mode_control = ModeDropdown(self.figure, (0.4, 0.07, 0.2, 0.05), len(result), self._set_mode)
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
        for ax, name in zip(self.field_axes, self.components, strict=False):
            ax.clear()
            colorbar = draw_field_panel(self.result, ax, name, self.quantity, self.mode)
            if colorbar is not None:
                self.colorbars.append(colorbar)
        for ax in self.field_axes[len(self.components) :]:
            ax.set_visible(False)
        self.figure.suptitle("Waveguide modes — " + effective_index_label(self.result, self.mode))
        self.figure.canvas.draw_idle()


def show_result(result, *, block=True):
    return show_figure(WaveguideModeViewer(result).figure, block)
