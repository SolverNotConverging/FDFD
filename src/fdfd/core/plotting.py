"""Matplotlib helpers shared by the result viewers, and :func:`plot_dispersion`."""

import numpy as np

from .errors import ConfigurationError


def plot_dispersion(frequencies, neff, *, connect=True, show=True):
    """Plot ``Re(neff)`` and ``Im(neff)`` against frequency, one trace per mode.

    Parameters
    ----------
    frequencies : array_like
        Positive frequencies in hertz, shape ``(F,)``.
    neff : array_like
        Complex effective indices, shape ``(F, modes)`` (or ``(F,)``).
        Column ``j`` holds the ``j``-th returned mode at each frequency.
    connect : bool
        Join samples with lines. Use ``False`` when the solver's mode order
        can change between frequencies (e.g. periodic or untracked sweeps);
        :class:`fdfd.ModeTracker1D` / ``ModeTracker2D`` give tracked branches.
    show : bool
        Call ``pyplot.show()``. With ``False`` a detached figure is returned
        for saving, without opening a window.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure

    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(neff, dtype=complex)
    if values.ndim == 1:
        values = values[:, None]
    if (
        frequencies.ndim != 1
        or not len(frequencies)
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0)
        or values.ndim != 2
        or values.shape[0] != len(frequencies)
        or not values.shape[1]
        or not np.isfinite(values).all()
    ):
        raise ConfigurationError(
            "Use positive frequencies in Hz and finite neff with shape (frequencies, modes)."
        )
    if not isinstance(show, bool) or not isinstance(connect, bool):
        raise ConfigurationError("show and connect must be booleans.")
    order = np.argsort(frequencies, kind="stable")
    if show:
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    else:
        figure = Figure(figsize=(8, 7))
        axes = figure.subplots(2, 1, sharex=True)
    ghz = frequencies[order] / 1e9
    for index in range(values.shape[1]):
        label = f"Mode {index + 1}"
        if connect:
            (line,) = axes[0].plot(ghz, values[order, index].real, "o-", label=label)
            axes[1].plot(ghz, values[order, index].imag, "o-", color=line.get_color(), label=label)
        else:
            points = axes[0].scatter(ghz, values[order, index].real, s=24, label=label)
            axes[1].scatter(
                ghz, values[order, index].imag, s=24, color=points.get_facecolor()[0], label=label
            )
    axes[0].set(ylabel="Re(neff)", title="Mode dispersion")
    axes[1].set(xlabel="Frequency (GHz)", ylabel="Im(neff)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def _draw_material_background(result, ax, plane, position):
    background = result.metadata.get("material_background")
    if background is None:
        return
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # The same geometry backdrop for E and H, including nonmagnetic dielectrics.
    values = np.sqrt(
        np.mean(np.abs(background["epsilon"]), axis=0) * np.mean(np.abs(background["mu"]), axis=0)
    )
    conductor = np.asarray(background["conductor"])
    coordinates = result.grid.coordinates
    axes = result.grid.axes
    if values.ndim == 3:
        plane = "xy" if plane is None else plane
        cut = next(i for i, axis in enumerate(axes) if axis not in plane)
        position = np.mean(coordinates[cut]) if position is None else position
        index = int(np.argmin(abs(coordinates[cut] - position)))
        values = np.take(values, index, axis=cut)
        conductor = np.take(conductor, index, axis=cut)
        coordinates = tuple(c for i, c in enumerate(coordinates) if i != cut)
        axes = tuple(axis for i, axis in enumerate(axes) if i != cut)
    if values.ndim == 1:
        lo, hi = result.grid.bounds[0]
        ax.imshow(
            values[None, :],
            extent=(lo, hi, 0.0, 1.0),
            transform=ax.get_xaxis_transform(),
            aspect="auto",
            cmap="Greys",
            alpha=0.22,
            zorder=-1,
        )
        step = (hi - lo) / len(values)
        for index in np.flatnonzero(np.diff(values)):
            ax.axvline(lo + (index + 1) * step, color=".4", lw=0.7, alpha=0.6)
        for index in np.flatnonzero(conductor):
            ax.axvspan(lo + index * step, lo + (index + 1) * step, color=".35", alpha=0.3, zorder=-1)
    else:
        if axes == ("x", "z"):
            coordinates, values, conductor = coordinates[::-1], values.T, conductor.T
        ax.pcolormesh(*coordinates, values.T, shading="auto", cmap="Greys", alpha=0.25, zorder=-1)
        if values.min() < values.max():
            levels = np.linspace(values.min(), values.max(), 5)[1:-1]
            ax.contour(*coordinates, values.T, levels=levels, colors=".25", linewidths=0.7, alpha=0.65)
        if conductor.any():
            ax.contourf(
                *coordinates, conductor.T, levels=(0.5, 1.5), colors=[".45"], alpha=0.55, hatches=["////"]
            )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def draw_field_panel(result, ax, component, quantity, mode=1, plane=None, position=None):
    """Draw one component over the material backdrop and return its colorbar (or None)."""
    result._draw(ax, mode=mode, component=component, quantity=quantity, plane=plane, position=position)
    ax.set_title(f"{component} ({quantity})")
    field = ax.collections[0] if ax.collections else None
    if result.metadata.get("material_background") is not None:
        if field is not None:
            field.set_alpha(0.85)
        _draw_material_background(result, ax, plane, position)
    if field is not None:
        colorbar = ax.figure.colorbar(field, ax=ax, fraction=0.046, pad=0.04)
        if quantity == "phase":
            colorbar.set_label("rad")
        return colorbar
    return None


def effective_index_label(result, mode):
    value = complex(result.neff[mode - 1])
    return f"Mode {mode}: neff = {value.real:.6g}{value.imag:+.6g}j"


class ModeDropdown:
    """A click-to-open mode selector built from Matplotlib widgets only."""

    def __init__(self, figure, bounds, count, on_select):
        from matplotlib.widgets import Button, RadioButtons

        self.figure = figure
        self.count = count
        self.mode = 1
        self.on_select = on_select
        self.button = Button(figure.add_axes(bounds), "Mode 1 ▾")
        self.button.on_clicked(self._toggle)
        left, bottom, width, height = bounds
        menu_height = min(0.65, max(0.08, 0.035 * count))
        # Open upwards so the menu stays inside the figure.
        self.menu_axis = figure.add_axes(
            (left, bottom + height, width, menu_height), zorder=100, facecolor="white"
        )
        # Blitting can redraw radio markers after the dropdown axes are hidden.
        self.options = RadioButtons(self.menu_axis, [f"Mode {i}" for i in range(1, count + 1)], useblit=False)
        self.options.on_clicked(self._select)
        self.menu_axis.set_visible(False)
        self.outside_click = figure.canvas.mpl_connect("button_press_event", self._close_outside)

    def _toggle(self, event):
        self.menu_axis.set_visible(not self.menu_axis.get_visible())
        self.figure.canvas.draw_idle()

    def _close_outside(self, event):
        if event.inaxes not in (self.button.ax, self.menu_axis) and self.menu_axis.get_visible():
            self.menu_axis.set_visible(False)
            self.figure.canvas.draw_idle()

    def _select(self, label):
        self.mode = int(label.split()[-1])
        self.button.label.set_text(f"Mode {self.mode} ▾")
        self.menu_axis.set_visible(False)
        self.on_select(self.mode)
        self.figure.canvas.draw_idle()


def show_figure(figure, block):
    """Show a viewer figure with ``pyplot.show(block=...)`` and return it."""
    from matplotlib import pyplot as plt

    if not isinstance(block, bool):
        raise ConfigurationError("block must be a boolean.")
    plt.show(block=block)
    return figure
