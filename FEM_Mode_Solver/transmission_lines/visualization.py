"""Native-mesh vector-field plots for transmission-line solutions.

The mode solver samples fields at integration points inside each finite
element.  This module deliberately reduces those samples onto their owning
triangle instead of globally triangulating the sample coordinates.  The
result therefore preserves conductor holes and every other feature of the
actual FEM topology.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _checked_result(result: Any) -> tuple[Any, Any, FloatArray, NDArray[np.int64]]:
    mode = getattr(result, "mode", None)
    fields = getattr(mode, "fields", None)
    if fields is None:
        raise TypeError("result must expose a solved mode through result.mode.fields.")
    if getattr(fields, "dimension", None) != 2:
        raise ValueError("Transmission-line vector plots require two-dimensional fields.")
    if fields.mesh_points is None or fields.mesh_cells is None:
        raise ValueError("Transmission-line fields do not contain their native FEM mesh.")

    points = np.asarray(fields.mesh_points, dtype=np.float64)
    cells = np.asarray(fields.mesh_cells, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("The native FEM mesh points must have shape (N, 2).")
    if cells.ndim != 2 or cells.shape[1] < 3 or cells.shape[0] == 0:
        raise ValueError("The native FEM mesh must contain triangular cells.")
    missing = [name for name in ("Ex", "Ey", "Hx", "Hy") if name not in fields.values]
    if missing:
        raise ValueError(
            "The solved mode is missing transverse field component(s): "
            + ", ".join(missing)
        )
    return mode, fields, points, cells[:, :3]


def _element_average(
    fields: Any,
    values: Any,
    triangles: NDArray[np.int64],
) -> ComplexArray:
    """Average complex nodal or quadrature data on each native triangle."""

    flat = np.asarray(values, dtype=np.complex128).reshape(-1)
    element_count = triangles.shape[0]
    point_count = np.asarray(fields.mesh_points).shape[0]
    if flat.size == element_count:
        return np.asarray(flat, dtype=np.complex128)

    owners_value = fields.metadata.get("sample_element_indices")
    if owners_value is not None:
        raw_owners = np.asarray(owners_value).reshape(-1)
        if raw_owners.size != flat.size or not np.all(np.isfinite(raw_owners)):
            raise ValueError("Invalid FEM quadrature ownership metadata.")
        owners = raw_owners.astype(np.int64)
        if (
            not np.array_equal(raw_owners, owners)
            or np.any(owners < 0)
            or np.any(owners >= element_count)
        ):
            raise ValueError("FEM quadrature owners are outside the native mesh.")
        counts = np.bincount(owners, minlength=element_count)
        if np.any(counts == 0):
            raise ValueError("At least one native triangle has no field samples.")
        totals = np.zeros(element_count, dtype=np.complex128)
        np.add.at(totals, owners, flat)
        return totals / counts

    if (
        fields.metadata.get("sampling") == "element-quadrature"
        and element_count > 0
        and flat.size % element_count == 0
    ):
        return np.mean(flat.reshape(element_count, -1), axis=1)

    # Supporting nodal data makes the visualizer useful for imported solutions
    # while still respecting the supplied FEM connectivity.
    if flat.size == point_count:
        return np.mean(flat[triangles], axis=1)
    raise ValueError(
        "Field samples cannot be associated with the native triangles; "
        "sample_element_indices metadata is required for quadrature data."
    )


def _field_summary(result: Any) -> str:
    label = str(getattr(result, "label", "Transmission line"))
    neff = complex(getattr(result, "neff"))
    characteristic = complex(getattr(result, "characteristic_impedance"))
    wave = complex(getattr(result, "wave_impedance"))
    return (
        f"{label}   "
        f"n_eff={_format_complex(neff)}   "
        f"Zc={_format_complex(characteristic)} ohm   "
        f"Zw={_format_complex(wave)} ohm"
    )


def _format_complex(value: complex, precision: int = 6) -> str:
    threshold = 10.0 ** (-(precision - 1))
    if abs(value.imag) <= threshold * max(1.0, abs(value.real)):
        return f"{value.real:.{precision}g}"
    return f"{value.real:.{precision}g}{value.imag:+.{precision}g}j"


def _direction_arrow_indices(
    centroids: FloatArray,
    magnitude: FloatArray,
    *,
    maximum_arrows: int = 400,
    relative_cutoff: float = 1.0e-2,
) -> NDArray[np.int64]:
    """Choose strong, spatially distributed cells for direction glyphs."""

    if magnitude.size == 0 or maximum_arrows <= 0:
        return np.empty(0, dtype=np.int64)
    maximum = float(np.max(magnitude))
    if not np.isfinite(maximum) or maximum <= np.finfo(float).tiny:
        return np.empty(0, dtype=np.int64)
    candidates = np.flatnonzero(magnitude >= relative_cutoff * maximum)
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)

    candidate_points = centroids[candidates]
    spans = np.ptp(candidate_points, axis=0)
    aspect = float(
        np.clip(
            spans[0] / max(float(spans[1]), np.finfo(float).tiny),
            0.25,
            4.0,
        )
    )
    columns = max(1, int(round(np.sqrt(maximum_arrows * aspect))))
    rows = max(1, maximum_arrows // columns)
    lower = np.min(candidate_points, axis=0)
    safe_spans = np.maximum(spans, np.finfo(float).tiny)
    normalized = (candidate_points - lower) / safe_spans
    column_index = np.clip(
        np.floor(normalized[:, 0] * columns).astype(np.int64),
        0,
        columns - 1,
    )
    row_index = np.clip(
        np.floor(normalized[:, 1] * rows).astype(np.int64),
        0,
        rows - 1,
    )
    bin_index = row_index * columns + column_index

    # Select the strongest cell in every occupied spatial bin.  Locations are
    # therefore stable with phase and are not biased by Gmsh element ordering
    # or local boundary refinement density.
    order = np.lexsort((-magnitude[candidates], bin_index))
    ordered_bins = bin_index[order]
    first_in_bin = np.concatenate(
        (np.asarray([True]), ordered_bins[1:] != ordered_bins[:-1])
    )
    return np.asarray(candidates[order[first_in_bin]], dtype=np.int64)


def _draw_vector_panel(
    ax: Any,
    colorbar_ax: Any,
    *,
    points: FloatArray,
    triangles: NDArray[np.int64],
    first: ComplexArray,
    second: ComplexArray,
    family: str,
    phase: float,
    mesh: bool,
) -> Any:
    from matplotlib.colors import PowerNorm
    from matplotlib.tri import Triangulation

    triangulation = Triangulation(points[:, 0], points[:, 1], triangles)
    centroids = np.mean(points[triangles], axis=1)
    magnitude = np.sqrt(np.abs(first) ** 2 + np.abs(second) ** 2)
    maximum = float(np.max(magnitude)) if magnitude.size else 0.0
    colour_maximum = maximum if maximum > np.finfo(float).tiny else 1.0

    ax.clear()
    colorbar_ax.clear()
    colorbar_ax.set_visible(True)
    artist = ax.tripcolor(
        triangulation,
        facecolors=magnitude,
        shading="flat",
        cmap="viridis" if family == "E" else "magma",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=colour_maximum),
        zorder=1,
    )

    rotation = np.exp(1j * float(phase))
    first_phase = np.real(first * rotation)
    second_phase = np.real(second * rotation)
    instantaneous = np.hypot(first_phase, second_phase)
    arrow_indices = _direction_arrow_indices(
        centroids,
        np.asarray(magnitude, dtype=np.float64),
    )
    if arrow_indices.size:
        selected_norm = instantaneous[arrow_indices]
        # The colour surface is the magnitude encoding.  Arrows carry only
        # instantaneous direction, so every visible vector is normalized to
        # unit length.  Suppress a glyph only near its temporal zero crossing,
        # where direction would be dominated by round-off.
        phase_valid = selected_norm > 1.0e-3 * magnitude[arrow_indices]
        arrow_indices = arrow_indices[phase_valid]
        selected_norm = selected_norm[phase_valid]
    if arrow_indices.size:
        ax.quiver(
            centroids[arrow_indices, 0],
            centroids[arrow_indices, 1],
            first_phase[arrow_indices] / selected_norm,
            second_phase[arrow_indices] / selected_norm,
            angles="xy",
            scale_units="inches",
            scale=8.5,
            pivot="middle",
            color="white",
            edgecolor="black",
            linewidth=0.30,
            width=0.0032,
            headwidth=3.2,
            headlength=4.2,
            alpha=0.93,
            zorder=5,
        )
    if mesh:
        ax.triplot(
            triangulation,
            color="black",
            alpha=0.28,
            linewidth=0.42,
            zorder=8,
        )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        rf"$|{family}_t|$ colour; direction of "
        rf"$\mathrm{{Re}}({family}_t e^{{j\phi}})$, "
        rf"$\phi={float(phase):.3g}$ rad"
    )
    colorbar = ax.figure.colorbar(artist, cax=colorbar_ax)
    unit = "V/m" if family == "E" else "A/m"
    colorbar.set_label(rf"$|{family}_t|$ ({unit})")
    return colorbar


def _draw_transverse_fields(
    result: Any,
    axes: tuple[Any, Any] | NDArray[Any],
    colorbar_axes: tuple[Any, Any] | NDArray[Any],
    *,
    phase: float,
    mesh: bool,
    update_title: bool = True,
) -> tuple[Any, Any]:
    """Draw into existing fixed axes; used by both public GUIs."""

    if not np.isfinite(phase):
        raise ValueError("phase must be finite and expressed in radians.")
    mode, fields, points, triangles = _checked_result(result)
    del mode
    field_axes = np.asarray(axes, dtype=object).reshape(-1)
    bars = np.asarray(colorbar_axes, dtype=object).reshape(-1)
    if field_axes.size != 2 or bars.size != 2:
        raise ValueError("Exactly two field axes and two colorbar axes are required.")

    averages = {
        name: _element_average(fields, fields.component(name), triangles)
        for name in ("Ex", "Ey", "Hx", "Hy")
    }
    electric_bar = _draw_vector_panel(
        field_axes[0],
        bars[0],
        points=points,
        triangles=triangles,
        first=averages["Ex"],
        second=averages["Ey"],
        family="E",
        phase=phase,
        mesh=mesh,
    )
    magnetic_bar = _draw_vector_panel(
        field_axes[1],
        bars[1],
        points=points,
        triangles=triangles,
        first=averages["Hx"],
        second=averages["Hy"],
        family="H",
        phase=phase,
        mesh=mesh,
    )
    if update_title:
        field_axes[0].figure.suptitle(_field_summary(result), fontsize=11)
    return electric_bar, magnetic_bar


def _new_field_figure(*, interactive: bool) -> tuple[Any, NDArray[Any], NDArray[Any]]:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11.5, 6.3 if interactive else 5.6))
    bottom = 0.22 if interactive else 0.13
    height = 0.68 if interactive else 0.75
    axes = np.asarray(
        [
            figure.add_axes((0.065, bottom, 0.34, height)),
            figure.add_axes((0.545, bottom, 0.34, height)),
        ],
        dtype=object,
    )
    colorbar_axes = np.asarray(
        [
            figure.add_axes((0.420, bottom + 0.04, 0.018, height - 0.08)),
            figure.add_axes((0.900, bottom + 0.04, 0.018, height - 0.08)),
        ],
        dtype=object,
    )
    return figure, axes, colorbar_axes


def visualize_transmission_line(
    result: Any,
    *,
    phase: float = 0.0,
    mesh: bool = False,
    show: bool = False,
) -> tuple[Any, NDArray[Any]]:
    """Plot the transverse electric and magnetic fields on the native mesh.

    ``phase`` is the phasor phase in radians for the uniform-length direction
    arrows; the per-element background and its colorbar show phase-independent
    vector magnitude.  The returned axes array contains only the electric and
    magnetic field panels.
    """

    import matplotlib.pyplot as plt

    figure, axes, colorbar_axes = _new_field_figure(interactive=False)
    _draw_transverse_fields(
        result,
        axes,
        colorbar_axes,
        phase=float(phase),
        mesh=bool(mesh),
    )
    if show:
        plt.show()
    return figure, axes


class TransmissionLineFieldViewer:
    """Interactive phase and mesh controller for one transmission-line result."""

    def __init__(
        self,
        result: Any,
        *,
        phase: float = 0.0,
        mesh: bool = False,
    ) -> None:
        from matplotlib.widgets import CheckButtons, Slider

        if not np.isfinite(phase):
            raise ValueError("phase must be finite and expressed in radians.")
        # Validate before constructing a partially initialized GUI.
        _checked_result(result)
        self.result = result
        self.phase = float(phase) % (2.0 * np.pi)
        self.mesh = bool(mesh)
        self.figure, self.axes, self.colorbar_axes = _new_field_figure(interactive=True)
        self._colorbars: tuple[Any, Any] | None = None

        slider_axes = self.figure.add_axes((0.20, 0.075, 0.48, 0.035))
        options_axes = self.figure.add_axes((0.77, 0.045, 0.14, 0.095))
        self.phase_control = Slider(
            slider_axes,
            "Phase (rad)",
            0.0,
            2.0 * np.pi,
            valinit=self.phase,
            valfmt="%.3f",
        )
        self.mesh_control = CheckButtons(options_axes, ("mesh",), (self.mesh,))
        options_axes.set_title("Overlay", fontsize=10)
        self.phase_control.on_changed(self._set_phase)
        self.mesh_control.on_clicked(self._set_mesh)
        self._draw()

    def _set_phase(self, value: float) -> None:
        self.phase = float(value)
        self._draw()

    def _set_mesh(self, _label: str) -> None:
        self.mesh = bool(self.mesh_control.get_status()[0])
        self._draw()

    def _draw(self) -> None:
        self._colorbars = _draw_transverse_fields(
            self.result,
            self.axes,
            self.colorbar_axes,
            phase=self.phase,
            mesh=self.mesh,
        )
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


def visualize_transmission_line_with_gui(
    result: Any,
    *,
    phase: float = 0.0,
    mesh: bool = False,
    show: bool = True,
    block: bool | None = None,
) -> TransmissionLineFieldViewer:
    """Create a Matplotlib phase/mesh GUI without requiring a GUI toolkit API."""

    viewer = TransmissionLineFieldViewer(result, phase=phase, mesh=mesh)
    if show:
        viewer.show(block=block)
    return viewer


# Concise aliases let ``TransmissionLineResult.visualize(...)`` delegate to
# this module in exactly the same way as the general mode-result API.
visualize = visualize_transmission_line
visualize_with_gui = visualize_transmission_line_with_gui


__all__ = [
    "TransmissionLineFieldViewer",
    "visualize",
    "visualize_transmission_line",
    "visualize_with_gui",
    "visualize_transmission_line_with_gui",
]
