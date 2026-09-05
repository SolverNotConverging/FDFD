"""Data-only electrostatic archives and Matplotlib inspection."""
from dataclasses import fields
from pathlib import Path
import h5py
import numpy as np

from fem_common import MeshSnapshot, mesh_snapshot
from fem_common.errors import PersistenceError, ConfigurationError
from fem_common.persistence import atomic_h5, write_envelope, validate_envelope, write_value, read_value
from .results import ElectrostaticResult


def save_result(result, path):
    mesh = result.mesh_data
    payload = {item.name: getattr(result, item.name) for item in fields(result) if item.name != "mesh"}
    with atomic_h5(path) as handle:
        write_envelope(handle, family="electrostatics", kind="potential", dimension=len(mesh.axes),
                       representation="nodal-potential; nodal-and-cell-fields", static=True)
        write_value(handle, "mesh", mesh)
        write_value(handle, "result", payload)
    return Path(path)


def load_result(path):
    try:
        with h5py.File(path, "r") as handle:
            validate_envelope(handle, family="electrostatics", static=True)
            registry = {"MeshSnapshot": MeshSnapshot}
            mesh = read_value(handle["mesh"], registry)
            return ElectrostaticResult(mesh=mesh, **read_value(handle["result"], registry))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PersistenceError(f"Cannot load electrostatic result from {path}: {exc}") from exc


def _draw(ax, result, component, quantity):
    import matplotlib.tri as mtri
    points, cells = result.coordinates, result.elements
    if component == "potential":
        value, unit, location = result.potential, "V", "node"
    elif component == "mesh":
        if points.shape[1] == 1:
            ax.plot(points[:, 0], np.zeros(len(points)), "|-", color="black")
        else:
            ax.triplot(points[:, 0], points[:, 1], cells, color="black", linewidth=.5)
        ax.set_title("FEM mesh")
        return
    elif component in ("Ex", "Ey", "E", "Dx", "Dy", "D"):
        field = result.element_electric_field if component.startswith("E") else result.element_displacement_field
        if field is None:
            raise ConfigurationError("The result has no cell fields.")
        index = {"x": 0, "y": 1}.get(component[-1])
        if index is not None and index >= points.shape[1]:
            raise ConfigurationError(f"{component} is not defined in one dimension.")
        value = np.linalg.norm(field, axis=1) if index is None else field[:, index]
        unit, location = ("V/m" if component.startswith("E") else "C/m²"), "cell"
    else:
        raise ConfigurationError("component must be potential, mesh, E, Ex, Ey, D, Dx, or Dy.")
    if quantity in ("magnitude", "abs"): value = np.abs(value)
    elif quantity != "real": raise ConfigurationError("Static fields support quantity='real' or 'magnitude'.")
    if points.shape[1] == 1:
        x = points[:, 0] if location == "node" else points[cells, 0].mean(axis=1)
        ax.plot(x, value)
        ax.set_ylabel(unit)
    else:
        triangle = mtri.Triangulation(points[:, 0], points[:, 1], cells)
        artist = ax.tripcolor(triangle, value, shading="gouraud") if location == "node" else ax.tripcolor(triangle, facecolors=value, shading="flat")
        ax.figure.colorbar(artist, ax=ax, label=unit)
        ax.set_aspect("equal")
        ax.set_ylabel("y (m)")
    ax.set_xlabel("x (m)")
    ax.set_title(component)


def plot_result(result, *, component=None, quantity="real"):
    from matplotlib.figure import Figure
    figure = Figure()
    ax = figure.subplots()
    _draw(ax, result, component or "potential", quantity)
    figure.tight_layout()
    return figure


def show_result(result, *, block=True):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons
    figure = plt.figure(figsize=(9, 6))
    selector_axes = figure.add_axes((.02, .25, .15, .5))
    choices = ("potential", "Ex", "E", "D", "mesh") if result.coordinates.shape[1] == 1 else ("potential", "Ex", "Ey", "E", "D", "mesh")
    buttons = RadioButtons(selector_axes, choices)
    def redraw(component):
        for axis in list(figure.axes):
            if axis is not selector_axes:
                figure.delaxes(axis)
        axis = figure.add_axes((.25, .15, .65, .75))
        _draw(axis, result, component, "real")
        figure.canvas.draw_idle()
    buttons.on_clicked(redraw)
    figure._cem_selector = buttons
    redraw("potential")
    plt.show(block=block)
    return figure
