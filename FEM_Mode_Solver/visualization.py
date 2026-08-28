"""Backend-neutral visualizers for :mod:`FEM_Mode_Solver` results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .results import Mode, ModeSet, SampledFields


@runtime_checkable
class SupportsModeVisualization(Protocol):
    """Minimal solver hook understood by :func:`visualize`.

    Solvers normally satisfy this protocol by storing the :class:`ModeSet`
    returned by ``solve()`` in a ``solution`` property.
    """

    @property
    def solution(self) -> ModeSet | None: ...


def _coerce_modes(source: Mode | ModeSet | SupportsModeVisualization | Any) -> tuple[Mode, ...]:
    if isinstance(source, Mode):
        return (source,)
    if isinstance(source, ModeSet):
        return source.modes

    for attribute in ("solution", "mode_set", "modes", "results"):
        if not hasattr(source, attribute):
            continue
        candidate = getattr(source, attribute)
        if candidate is None:
            continue
        if candidate is source:
            continue
        try:
            return _coerce_modes(candidate)
        except (TypeError, ValueError):
            pass
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        modes = tuple(source)
        if all(isinstance(mode, Mode) for mode in modes):
            return modes
    raise TypeError(
        "Expected a Mode, ModeSet, sequence of Mode objects, or a solved object "
        "with a ModeSet-valued '.solution' property."
    )


def _select_mode(modes: tuple[Mode, ...], mode: int | Mode) -> tuple[Mode, int]:
    if not modes:
        raise ValueError("There are no modes to visualize.")
    if isinstance(mode, Mode):
        return mode, modes.index(mode) + 1 if mode in modes else mode.index
    if isinstance(mode, (bool, np.bool_)):
        raise TypeError("mode must be a one-based integer or a Mode instance.")
    number = int(mode)
    if not 1 <= number <= len(modes):
        raise IndexError(f"mode must be between 1 and {len(modes)}; got {mode!r}.")
    return modes[number - 1], number


def _canonical_plot_quantity(quantity: str) -> str:
    aliases = {
        "real": "real",
        "re": "real",
        "imag": "imag",
        "imaginary": "imag",
        "im": "imag",
        "magnitude": "magnitude",
        "mag": "magnitude",
        "abs": "magnitude",
        "phase": "phase",
        "angle": "phase",
    }
    try:
        return aliases[str(quantity).strip().lower()]
    except KeyError as exc:
        raise ValueError("quantity must be 'real', 'imag', 'magnitude', or 'phase'.") from exc


def _canonical_plot_component(component: str) -> str:
    stripped = str(component).strip()
    lower = stripped.lower().replace("|", "")
    if lower in ("e", "h"):
        return lower.upper()
    if lower in ("ex", "ey", "ez", "hx", "hy", "hz"):
        return lower[0].upper() + lower[1]
    return stripped


def _components_from_arguments(
    fields: SampledFields,
    component: str | None,
    components: Sequence[str] | str | None,
    legacy_flags: Mapping[str, Any],
) -> tuple[str, ...]:
    unknown = set(legacy_flags) - {
        "ex",
        "ey",
        "ez",
        "hx",
        "hy",
        "hz",
        "eabs",
        "habs",
    }
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"Unexpected visualization option(s): {names}.")
    if component is not None and components is not None:
        raise ValueError("Pass component or components, not both.")

    selected: list[str] = []
    if component is not None:
        selected.append(_canonical_plot_component(component))
    elif components is not None:
        raw = (components,) if isinstance(components, str) else tuple(components)
        selected.extend(_canonical_plot_component(item) for item in raw)
    for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
        if bool(legacy_flags.get(name, False)):
            selected.append(_canonical_plot_component(name))
    if bool(legacy_flags.get("eabs", False)):
        selected.append("E")
    if bool(legacy_flags.get("habs", False)):
        selected.append("H")
    if not selected:
        selected.extend(fields.components)

    unique = tuple(dict.fromkeys(selected))
    for name in unique:
        if name in ("E", "H"):
            fields.vector_magnitude(name)
        else:
            fields.component(name)
    return unique


def _normalised(data: NDArray[np.float64], normalize: bool) -> tuple[NDArray[np.float64], str]:
    if not normalize:
        return data, ""
    scale = float(np.max(np.abs(data))) if data.size else 0.0
    if scale <= np.finfo(float).tiny:
        return data, ""
    return np.asarray(data / scale, dtype=np.float64), " (normalized)"


def _mode_title(mode: Mode, number: int, component: str, quantity: str) -> str:
    polarization = "" if mode.polarization is None else f" {mode.polarization}"
    rendered_quantity = "magnitude" if component in ("E", "H") else quantity
    return (
        f"Mode {number}{polarization}: {component} {rendered_quantity}\n"
        f"$n_{{eff}}$={mode.neff.real:.7g}{mode.neff.imag:+.3g}j"
    )


def _colour_limits(data: NDArray[np.float64], quantity: str) -> dict[str, float]:
    if quantity == "phase":
        return {"vmin": -np.pi, "vmax": np.pi}
    if quantity not in ("real", "imag") or data.size == 0:
        return {}
    limit = float(np.max(np.abs(data)))
    return {} if limit <= np.finfo(float).tiny else {"vmin": -limit, "vmax": limit}


def _material_levels(material: NDArray[Any]) -> NDArray[np.float64]:
    values = np.unique(np.round(np.abs(np.asarray(material, dtype=np.complex128)), 12))
    if values.size < 2:
        return np.empty(0, dtype=float)
    if values.size <= 12:
        return 0.5 * (values[:-1] + values[1:])
    return np.linspace(float(values.min()), float(values.max()), 8)[1:-1]


def _native_triangulation(fields: SampledFields) -> Any | None:
    """Return the supplied FEM triangulation, if this is a 2D mesh sample."""

    if (
        fields.dimension != 2
        or fields.mesh_points is None
        or fields.mesh_cells is None
    ):
        return None
    from matplotlib.tri import Triangulation

    points = np.asarray(fields.mesh_points, dtype=np.float64)
    triangles = np.asarray(fields.mesh_cells[:, :3], dtype=np.int64)
    return Triangulation(points[:, 0], points[:, 1], triangles)


def _native_sample_data(
    fields: SampledFields,
    data: NDArray[Any],
) -> tuple[str, NDArray[np.float64]] | None:
    """Map sampled values onto native FEM nodes or cells.

    Two-dimensional solver fields live at quadrature points, not vertices.
    ``sample_element_indices`` records which native triangle owns each flattened
    sample.  Reducing those samples per element preserves holes and non-convex
    boundaries during plotting.  A conservative scatter fallback is used by
    callers when no trustworthy ownership information is available.
    """

    if fields.mesh_points is None or fields.mesh_cells is None:
        return None
    flat = np.asarray(data, dtype=np.float64).reshape(-1)
    cells = np.asarray(fields.mesh_cells)
    cell_count = cells.shape[0]
    point_count = fields.mesh_points.shape[0]

    # Native nodal samples can use smooth Gouraud shading without any
    # interpolation outside the supplied connectivity.
    if flat.size == point_count:
        x, y = (np.asarray(axis).reshape(-1) for axis in fields.coordinates)
        points = np.asarray(fields.mesh_points)
        if (
            x.size == point_count
            and y.size == point_count
            and np.allclose(x, points[:, 0])
            and np.allclose(y, points[:, 1])
        ):
            return "node", flat

    if flat.size == cell_count:
        return "cell", flat

    owners_value = fields.metadata.get("sample_element_indices")
    if owners_value is not None:
        raw_owners = np.asarray(owners_value).reshape(-1)
        if raw_owners.size != flat.size or not np.all(np.isfinite(raw_owners)):
            return None
        owners = raw_owners.astype(np.int64)
        if (
            not np.array_equal(raw_owners, owners)
            or np.any(owners < 0)
            or np.any(owners >= cell_count)
        ):
            return None
        counts = np.bincount(owners, minlength=cell_count)
        if np.any(counts == 0):
            return None
        sums = np.bincount(owners, weights=flat, minlength=cell_count)
        return "cell", np.asarray(sums / counts, dtype=np.float64)

    # Backward compatibility for early FEM results which documented the
    # element-major quadrature layout but did not yet store explicit owners.
    if (
        fields.metadata.get("sampling") == "element-quadrature"
        and cell_count > 0
        and flat.size % cell_count == 0
    ):
        return "cell", np.mean(flat.reshape(cell_count, -1), axis=1)
    return None


def _cell_to_node(
    triangles: NDArray[np.int64],
    cell_values: NDArray[np.float64],
    point_count: int,
) -> NDArray[np.float64]:
    totals = np.zeros(point_count, dtype=np.float64)
    counts = np.zeros(point_count, dtype=np.int64)
    for local_vertex in range(3):
        vertices = triangles[:, local_vertex]
        np.add.at(totals, vertices, cell_values)
        np.add.at(counts, vertices, 1)
    result = np.zeros(point_count, dtype=np.float64)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def _draw_cell_material_interfaces(
    ax: Any,
    fields: SampledFields,
    values: NDArray[np.float64],
) -> None:
    """Draw exact native edges separating piecewise-constant materials."""

    from matplotlib.collections import LineCollection

    points = np.asarray(fields.mesh_points, dtype=np.float64)
    triangles = np.asarray(fields.mesh_cells[:, :3], dtype=np.int64)
    owners: dict[tuple[int, int], list[int]] = {}
    for cell_index, triangle in enumerate(triangles):
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            owners.setdefault(edge, []).append(cell_index)

    segments = [
        points[np.asarray(edge, dtype=np.int64)]
        for edge, adjacent in owners.items()
        if len(adjacent) == 2
        and not np.isclose(values[adjacent[0]], values[adjacent[1]], rtol=1e-9, atol=1e-12)
    ]
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors="white",
                linewidths=0.7,
                alpha=0.65,
                zorder=8,
            )
        )


def _draw_material(ax: Any, fields: SampledFields) -> None:
    material = fields.material
    if material is None:
        return
    values = np.abs(np.asarray(material))
    if fields.dimension == 1:
        maximum = float(np.max(values)) if values.size else 0.0
        if maximum > np.finfo(float).tiny:
            ax.fill_between(
                fields.x,
                0.0,
                values / maximum,
                transform=ax.get_xaxis_transform(),
                color="0.55",
                alpha=0.15,
                step="mid",
                linewidth=0.0,
                zorder=-10,
            )
        return

    levels = _material_levels(material)
    if levels.size == 0:
        return
    x, y = fields.coordinates
    try:
        if fields.layout == "structured":
            ax.contour(x, y, values, levels=levels, colors="white", linewidths=0.7, alpha=0.65)
        elif (triangulation := _native_triangulation(fields)) is not None:
            native = _native_sample_data(fields, values)
            if native is None:
                # The supplied mesh tells us a hole/non-convex topology exists,
                # but not which element owns each sample.  Skipping this
                # decorative overlay is safer than inventing global triangles.
                return
            location, native_values = native
            if location == "cell" and np.unique(np.round(native_values, 12)).size <= 12:
                _draw_cell_material_interfaces(ax, fields, native_values)
            else:
                if location == "cell":
                    native_values = _cell_to_node(
                        np.asarray(fields.mesh_cells[:, :3], dtype=np.int64),
                        native_values,
                        fields.mesh_points.shape[0],
                    )
                ax.tricontour(
                    triangulation,
                    native_values,
                    levels=levels,
                    colors="white",
                    linewidths=0.7,
                    alpha=0.65,
                )
        else:
            ax.tricontour(
                np.ravel(x),
                np.ravel(y),
                np.ravel(values),
                levels=levels,
                colors="white",
                linewidths=0.7,
                alpha=0.65,
            )
    except (RuntimeError, ValueError):
        # Material contours are decorative; degenerate point sets should not
        # prevent the modal field itself from being inspected.
        return


def _draw_mesh(ax: Any, fields: SampledFields) -> None:
    if fields.mesh_points is None or fields.mesh_cells is None:
        return
    points = fields.mesh_points
    cells = fields.mesh_cells
    if fields.dimension == 1:
        ax.vlines(
            points[:, 0],
            0.0,
            1.0,
            transform=ax.get_xaxis_transform(),
            color="0.35",
            alpha=0.22,
            linewidth=0.45,
            zorder=10,
        )
        return

    from matplotlib.tri import Triangulation

    triangles = np.asarray(cells[:, :3], dtype=np.int64)
    triangulation = Triangulation(points[:, 0], points[:, 1], triangles)
    ax.triplot(triangulation, color="k", alpha=0.28, linewidth=0.42, zorder=10)


def _draw_component(
    ax: Any,
    mode: Mode,
    number: int,
    component: str,
    quantity: str,
    *,
    cmap: str,
    normalize: bool,
    material: bool,
    mesh: bool,
) -> Any | None:
    fields = mode.fields
    selected_quantity = "magnitude" if component in ("E", "H") else quantity
    data = np.asarray(fields.quantity(component, selected_quantity), dtype=np.float64)
    data, normalization_label = _normalised(data, normalize)

    artist = None
    if fields.dimension == 1:
        ax.plot(fields.x, data, color="tab:blue", linewidth=1.7)
        ax.axhline(0.0, color="0.35", linewidth=0.65, alpha=0.5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel(f"{component} {selected_quantity}{normalization_label}")
        ax.grid(True, alpha=0.22)
    else:
        x, y = fields.coordinates
        if fields.layout in ("structured", "curvilinear"):
            limits = _colour_limits(data, selected_quantity)
            artist = ax.pcolormesh(x, y, data, shading="auto", cmap=cmap, **limits)
        elif (triangulation := _native_triangulation(fields)) is not None:
            native = _native_sample_data(fields, data)
            if native is None:
                # Do not globally triangulate samples when native connectivity
                # is present: that can fill PEC/PMC holes and concave cut-outs.
                limits = _colour_limits(data, selected_quantity)
                artist = ax.scatter(
                    np.ravel(x),
                    np.ravel(y),
                    c=np.ravel(data),
                    cmap=cmap,
                    s=12,
                    linewidths=0.0,
                    **limits,
                )
            else:
                location, native_values = native
                limits = _colour_limits(native_values, selected_quantity)
                if location == "node":
                    artist = ax.tripcolor(
                        triangulation,
                        native_values,
                        shading="gouraud",
                        cmap=cmap,
                        **limits,
                    )
                else:
                    artist = ax.tripcolor(
                        triangulation,
                        facecolors=native_values,
                        shading="flat",
                        cmap=cmap,
                        **limits,
                    )
        else:
            limits = _colour_limits(data, selected_quantity)
            try:
                artist = ax.tricontourf(
                    np.ravel(x),
                    np.ravel(y),
                    np.ravel(data),
                    levels=64,
                    cmap=cmap,
                    **limits,
                )
            except (RuntimeError, ValueError):
                artist = ax.scatter(
                    np.ravel(x),
                    np.ravel(y),
                    c=np.ravel(data),
                    cmap=cmap,
                    s=12,
                    linewidths=0.0,
                    **limits,
                )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
    if material:
        _draw_material(ax, fields)
    if mesh:
        _draw_mesh(ax, fields)
    ax.set_title(_mode_title(mode, number, component, selected_quantity))
    return artist


def visualize(
    source: Mode | ModeSet | SupportsModeVisualization | Any,
    mode: int | Mode = 1,
    component: str | None = None,
    *,
    components: Sequence[str] | str | None = None,
    quantity: str = "real",
    mesh: bool = False,
    mesh_overlay: bool | None = None,
    material: bool = True,
    normalize: bool = False,
    cmap: str | None = None,
    axes: Any | None = None,
    title: str | None = None,
    show: bool = False,
    **legacy_component_flags: Any,
) -> tuple[Any, NDArray[Any]]:
    """Plot one mode using common component and quantity controls.

    ``mode`` is one-based for compatibility with the original mode solvers.
    A single ``component`` or several ``components`` may be requested.  The
    legacy boolean flags ``ex=True`` through ``hz=True``, plus ``eabs`` and
    ``habs``, remain accepted by thin solver wrappers.

    Returns
    -------
    (figure, axes)
        ``axes`` is always a flat NumPy object array containing only the active
        field panels.  The function does not call ``show`` unless requested.
    """

    import matplotlib.pyplot as plt

    modes = _coerce_modes(source)
    selected_mode, number = _select_mode(modes, mode)
    selected_quantity = _canonical_plot_quantity(quantity)
    selected_components = _components_from_arguments(
        selected_mode.fields,
        component,
        components,
        legacy_component_flags,
    )
    mesh_enabled = bool(mesh if mesh_overlay is None else mesh_overlay)
    colour_map = cmap or (
        "twilight" if selected_quantity == "phase" else
        "RdBu_r" if selected_quantity in ("real", "imag") else
        "viridis"
    )

    if axes is None:
        columns = min(3, len(selected_components))
        rows = int(ceil(len(selected_components) / columns))
        figure, grid = plt.subplots(
            rows,
            columns,
            figsize=(5.0 * columns, 4.0 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        flat_axes = np.asarray(grid, dtype=object).ravel()
        for unused in flat_axes[len(selected_components):]:
            unused.set_visible(False)
    else:
        flat_axes = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
        if flat_axes.size < len(selected_components):
            raise ValueError(
                f"Need {len(selected_components)} axes, received {flat_axes.size}."
            )
        figure = flat_axes[0].figure

    active_axes = flat_axes[:len(selected_components)]
    for ax, selected_component in zip(active_axes, selected_components, strict=True):
        component_cmap = (
            "viridis"
            if cmap is None and selected_component in ("E", "H")
            else colour_map
        )
        artist = _draw_component(
            ax,
            selected_mode,
            number,
            selected_component,
            selected_quantity,
            cmap=component_cmap,
            normalize=normalize,
            material=material,
            mesh=mesh_enabled,
        )
        if artist is not None:
            figure.colorbar(artist, ax=ax, shrink=0.88)
    if title is not None:
        figure.suptitle(str(title))
    if show:
        plt.show()
    return figure, np.asarray(active_axes, dtype=object)


class ModeViewer:
    """Interactive Matplotlib controller returned by :func:`visualize_with_gui`."""

    def __init__(
        self,
        source: Mode | ModeSet | SupportsModeVisualization | Any,
        *,
        mode: int = 1,
        component: str | None = None,
        quantity: str = "real",
        mesh: bool = False,
        material: bool = True,
        normalize: bool = False,
        cmap: str | None = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons, RadioButtons

        self.modes = _coerce_modes(source)
        selected_mode, selected_number = _select_mode(self.modes, mode)
        self.mode_number = selected_number
        self.component_names = tuple(
            dict.fromkeys(
                component_name
                for item in self.modes
                for component_name in item.fields.components
            )
        )
        if any(any(name.startswith("E") for name in item.fields.components) for item in self.modes):
            self.component_names += ("E",)
        if any(any(name.startswith("H") for name in item.fields.components) for item in self.modes):
            self.component_names += ("H",)
        if not self.component_names:
            raise ValueError("No sampled field components are available.")
        self.component = (
            self.component_names[0]
            if component is None
            else _canonical_plot_component(component)
        )
        if self.component not in self.component_names:
            raise KeyError(f"Unknown initial component {self.component!r}.")
        self.quantity = _canonical_plot_quantity(quantity)
        self.mesh = bool(mesh)
        self.material = bool(material)
        self.normalize = bool(normalize)
        self.cmap = cmap
        self._colorbar: Any | None = None

        self.figure = plt.figure(figsize=(10.5, 6.5))
        # Keep the field and colorbar in independent, fixed axes.  Passing the
        # field axes to ``Figure.colorbar`` would make Matplotlib take space
        # from it on every redraw; removing that colorbar does not restore a
        # manually positioned axes, so repeated widget changes progressively
        # shrink the field panel.
        self.axes = self.figure.add_axes((0.08, 0.12, 0.60, 0.80))
        self._colorbar_axes = self.figure.add_axes((0.705, 0.17, 0.022, 0.70))
        self._colorbar_axes.set_visible(False)
        mode_axes = self.figure.add_axes((0.77, 0.67, 0.21, 0.25))
        component_axes = self.figure.add_axes((0.77, 0.39, 0.10, 0.24))
        quantity_axes = self.figure.add_axes((0.88, 0.39, 0.10, 0.24))
        options_axes = self.figure.add_axes((0.77, 0.22, 0.21, 0.11))

        self._mode_labels = tuple(
            f"{index + 1}: {item.polarization or 'mode'}"
            for index, item in enumerate(self.modes)
        )
        self.mode_control = RadioButtons(
            mode_axes,
            self._mode_labels,
            active=self.mode_number - 1,
        )
        self.component_control = RadioButtons(
            component_axes,
            self.component_names,
            active=self.component_names.index(self.component),
        )
        quantities = ("real", "imag", "magnitude", "phase")
        self.quantity_control = RadioButtons(
            quantity_axes,
            quantities,
            active=quantities.index(self.quantity),
        )
        self.options_control = CheckButtons(
            options_axes,
            ("mesh", "material", "normalize"),
            (self.mesh, self.material, self.normalize),
        )
        mode_axes.set_title("Mode", fontsize=10)
        component_axes.set_title("Component", fontsize=10)
        quantity_axes.set_title("Quantity", fontsize=10)
        options_axes.set_title("Overlays", fontsize=10)

        self.mode_control.on_clicked(self._set_mode)
        self.component_control.on_clicked(self._set_component)
        self.quantity_control.on_clicked(self._set_quantity)
        self.options_control.on_clicked(self._set_option)
        self._draw()

    @property
    def mode(self) -> Mode:
        return self.modes[self.mode_number - 1]

    def _set_mode(self, label: str) -> None:
        self.mode_number = self._mode_labels.index(label) + 1
        self._draw()

    def _set_component(self, label: str) -> None:
        self.component = label
        self._draw()

    def _set_quantity(self, label: str) -> None:
        self.quantity = label
        self._draw()

    def _set_option(self, _label: str) -> None:
        self.mesh, self.material, self.normalize = self.options_control.get_status()
        self._draw()

    def _draw(self) -> None:
        self.axes.clear()
        self._colorbar_axes.set_visible(False)
        if self.component not in ("E", "H") and self.component not in self.mode.fields.values:
            self.axes.text(
                0.5,
                0.5,
                f"{self.component} is not sampled for mode {self.mode_number}.",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
            )
            self.figure.canvas.draw_idle()
            return
        if self.component in ("E", "H"):
            available = any(
                name.startswith(self.component) for name in self.mode.fields.components
            )
            if not available:
                self.axes.text(
                    0.5,
                    0.5,
                    f"No {self.component}-field components are sampled for this mode.",
                    ha="center",
                    va="center",
                    transform=self.axes.transAxes,
                )
                self.figure.canvas.draw_idle()
                return
        selected_quantity = _canonical_plot_quantity(self.quantity)
        colour_map = self.cmap or (
            "viridis" if self.component in ("E", "H") else
            "twilight" if selected_quantity == "phase" else
            "RdBu_r" if selected_quantity in ("real", "imag") else
            "viridis"
        )
        artist = _draw_component(
            self.axes,
            self.mode,
            self.mode_number,
            self.component,
            selected_quantity,
            cmap=colour_map,
            normalize=self.normalize,
            material=self.material,
            mesh=self.mesh,
        )
        if artist is not None:
            self._colorbar_axes.set_visible(True)
            if self._colorbar is None:
                self._colorbar = self.figure.colorbar(
                    artist,
                    cax=self._colorbar_axes,
                )
            else:
                self._colorbar.update_normal(artist)
        self.figure.canvas.draw_idle()

    def show(self, *, block: bool | None = None) -> None:
        import matplotlib.pyplot as plt

        if block is None:
            plt.show()
        else:
            plt.show(block=bool(block))

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.figure)


def visualize_with_gui(
    source: Mode | ModeSet | SupportsModeVisualization | Any,
    *,
    mode: int = 1,
    component: str | None = None,
    quantity: str = "real",
    mesh: bool = False,
    mesh_overlay: bool | None = None,
    material: bool = True,
    normalize: bool = False,
    cmap: str | None = None,
    show: bool = True,
    block: bool | None = None,
) -> ModeViewer:
    """Create an interactive mode/component/quantity viewer.

    The GUI uses Matplotlib widgets, so it works with any interactive
    Matplotlib backend and does not require solver-specific Tk code.  The
    returned controller must be kept alive while the window is open.
    """

    viewer = ModeViewer(
        source,
        mode=mode,
        component=component,
        quantity=quantity,
        mesh=bool(mesh if mesh_overlay is None else mesh_overlay),
        material=material,
        normalize=normalize,
        cmap=cmap,
    )
    if show:
        viewer.show(block=block)
    return viewer


__all__ = [
    "ModeViewer",
    "SupportsModeVisualization",
    "visualize",
    "visualize_with_gui",
]
