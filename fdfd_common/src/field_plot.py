"""Small drawing helpers used by the solver-specific Matplotlib viewers."""
import numpy as np


def _draw_material_background(result, ax, component, plane, position):
    background = result.metadata.get('material_background')
    if background is None:
        return
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # Use the same geometry backdrop for E and H, including nonmagnetic dielectrics.
    values = np.sqrt(np.mean(np.abs(background['epsilon']), axis=0)
                     * np.mean(np.abs(background['mu']), axis=0))
    conductor = np.asarray(background['conductor'])
    coordinates = result.mesh_data.coordinates
    axes = result.mesh_data.axes
    if values.ndim == 3:
        plane = 'xy' if plane is None else plane
        cut = next(i for i, axis in enumerate(axes) if axis not in plane)
        position = np.mean(coordinates[cut]) if position is None else position
        index = int(np.argmin(abs(coordinates[cut]-position)))
        values = np.take(values, index, axis=cut)
        conductor = np.take(conductor, index, axis=cut)
        coordinates = tuple(c for i, c in enumerate(coordinates) if i != cut)
        axes = tuple(axis for i, axis in enumerate(axes) if i != cut)
    if values.ndim == 1:
        lo, hi = result.mesh_data.bounds[0]
        ax.imshow(values[None, :], extent=(lo, hi, 0., 1.),
                  transform=ax.get_xaxis_transform(), aspect='auto', cmap='Greys',
                  alpha=.22, zorder=-1)
        step = (hi-lo)/len(values)
        for index in np.flatnonzero(np.diff(values)):
            ax.axvline(lo+(index+1)*step, color='.4', lw=.7, alpha=.6)
        for index in np.flatnonzero(conductor):
            ax.axvspan(lo+index*step, lo+(index+1)*step, color='.35', alpha=.3, zorder=-1)
    else:
        if result.family == 'fdfd_periodic_modes' and axes == ('x', 'z'):
            coordinates = coordinates[::-1]
            values = values.T
            conductor = conductor.T
        ax.pcolormesh(*coordinates, values.T, shading='auto', cmap='Greys',
                      alpha=.25, zorder=-1)
        if values.min() < values.max():
            levels = np.linspace(values.min(), values.max(), 5)[1:-1]
            ax.contour(*coordinates, values.T, levels=levels, colors='.25',
                       linewidths=.7, alpha=.65)
        if conductor.any():
            ax.contourf(*coordinates, conductor.T, levels=(.5, 1.5),
                        colors=['.45'], alpha=.55, hatches=['////'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def draw_field_panel(result, ax, component, quantity, mode=1, plane=None, position=None):
    result._draw(ax, component, quantity, mode, plane, position)
    ax.set_title(f'{component} ({quantity})')
    field = ax.collections[0] if ax.collections else None
    if result.metadata.get('material_background') is not None:
        if field is not None:
            field.set_alpha(.85)
        _draw_material_background(result, ax, component, plane, position)
    if field is not None:
        colorbar = ax.figure.colorbar(field, ax=ax, fraction=.046, pad=.04)
        if quantity == 'phase':
            colorbar.set_label('rad')
        return colorbar
    return None


def effective_index_label(result, mode):
    value = complex(result.neff[mode-1])
    return f'Mode {mode}: neff = {value.real:.6g}{value.imag:+.6g}j'
