"""Periodic Bloch-field viewer with a dedicated 3D slice explorer."""

import numpy as np

from ..core.plotting import ModeDropdown, draw_field_panel, effective_index_label, show_figure


class PeriodicModeViewer:
    def __init__(self, result):
        from matplotlib import pyplot as plt
        from matplotlib.widgets import RadioButtons, Slider

        self.result = result
        self.mode = 1
        self.quantity = "magnitude"
        self.plane = "xy" if len(result.grid.axes) == 3 else None
        self.slice_fraction = 0.5
        self.components = tuple(
            name for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") if name in result.fields
        )
        if len(result.grid.axes) == 2:
            polarizations = set(result.metadata.get("polarizations", ()))
            polarization = polarizations.pop() if len(polarizations) == 1 else None
            if polarization in ("TE", "TM"):
                active = ("Ey", "Hx", "Hz") if polarization == "TE" else ("Ex", "Ez", "Hy")
                self.components = tuple(name for name in active if name in result.fields)
        columns = min(3, len(self.components))
        rows = int(np.ceil(len(self.components) / columns))
        self.figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(13, 9))
        self.figure.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.27, hspace=0.55, wspace=0.5)
        self.field_axes = tuple(axes.flat)
        self.colorbars = []
        self.quantity_control = RadioButtons(
            self.figure.add_axes((0.025, 0.02, 0.18, 0.17)), ("magnitude", "real", "imag", "phase")
        )
        self.quantity_control.ax.set_title("Field quantity")
        self.quantity_control.on_clicked(self._set_quantity)
        self.mode_control = ModeDropdown(self.figure, (0.48, 0.16, 0.2, 0.05), len(result), self._set_mode)
        self.plane_control = self.slice_control = None
        if self.plane is not None:
            self.plane_control = RadioButtons(
                self.figure.add_axes((0.25, 0.025, 0.12, 0.14)), ("xy", "xz", "yz")
            )
            self.plane_control.ax.set_title("Slice plane")
            self.plane_control.on_clicked(self._set_plane)
            self.slice_control = Slider(
                self.figure.add_axes((0.48, 0.07, 0.4, 0.03)), "Slice", 0.0, 1.0, valinit=0.5
            )
            self.slice_control.on_changed(self._set_slice)
        self.figure._periodic_mode_viewer = self
        self.draw()

    def _set_mode(self, value):
        self.mode = int(value)
        self.draw()

    def _set_quantity(self, value):
        self.quantity = value
        self.draw()

    def _set_plane(self, value):
        self.plane = value
        self.draw()

    def _set_slice(self, value):
        self.slice_fraction = float(value)
        self.draw()

    def draw(self):
        position = None
        slice_label = ""
        if self.plane is not None:
            cut = next(i for i, axis in enumerate(self.result.grid.axes) if axis not in self.plane)
            lo, hi = self.result.grid.bounds[cut]
            position = lo + self.slice_fraction * (hi - lo)
            axis = self.result.grid.axes[cut]
            slice_label = f"\n{self.plane} slice: {axis} = {position:.6g} m"
            self.slice_control.valtext.set_text(f"{position:.6g} m")
        for colorbar in self.colorbars:
            colorbar.remove()
        self.colorbars = []
        for ax, name in zip(self.field_axes, self.components, strict=False):
            ax.clear()
            colorbar = draw_field_panel(self.result, ax, name, self.quantity, self.mode, self.plane, position)
            if colorbar is not None:
                self.colorbars.append(colorbar)
        for ax in self.field_axes[len(self.components) :]:
            ax.set_visible(False)
        self.figure.suptitle(
            "Periodic modes — " + effective_index_label(self.result, self.mode) + slice_label
        )
        self.figure.canvas.draw_idle()


def show_result(result, *, block=True):
    return show_figure(PeriodicModeViewer(result).figure, block)
