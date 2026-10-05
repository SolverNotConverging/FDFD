"""Matplotlib dropdown controls shared by the dedicated modal viewers."""
from matplotlib.widgets import Button, RadioButtons


class ModeDropdown:
    """A click-to-open mode selection box, without an additional GUI toolkit."""
    def __init__(self, figure, bounds, count, on_select):
        self.figure = figure
        self.count = count
        self.mode = 1
        self.on_select = on_select
        self.button = Button(figure.add_axes(bounds), 'Mode 1 ▾')
        self.button.on_clicked(self._toggle)
        left, bottom, width, height = bounds
        menu_height = min(.65, max(.08, .035*count))
        # Open upwards so the menu stays inside the figure.
        self.menu_axis = figure.add_axes((left, bottom+height, width, menu_height),
                                         zorder=100, facecolor='white')
        # Blitting can redraw the radio markers after the dropdown axes are hidden.
        self.options = RadioButtons(self.menu_axis, [f'Mode {i}' for i in range(1, count+1)],
                                    useblit=False)
        self.options.on_clicked(self._select)
        self.menu_axis.set_visible(False)
        self.outside_click = figure.canvas.mpl_connect('button_press_event', self._close_outside)

    def _toggle(self, event):
        self.menu_axis.set_visible(not self.menu_axis.get_visible())
        self.figure.canvas.draw_idle()

    def _close_outside(self, event):
        if event.inaxes not in (self.button.ax, self.menu_axis) and self.menu_axis.get_visible():
            self.menu_axis.set_visible(False)
            self.figure.canvas.draw_idle()

    def _select(self, label):
        self.mode = int(label.split()[-1])
        self.button.label.set_text(f'Mode {self.mode} ▾')
        self.menu_axis.set_visible(False)
        self.on_select(self.mode)
        self.figure.canvas.draw_idle()
