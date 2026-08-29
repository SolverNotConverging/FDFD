"""Backend-neutral plotting helpers for standalone WaveFEM result viewing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import SceneData


SKey = tuple[str, int, int]
SPlotQuantity = Literal["magnitude_db", "magnitude", "phase_deg", "real", "imag"]
FieldQuantity = Literal["abs", "real", "imag"]


@dataclass(frozen=True, slots=True)
class SParameterRow:
    """One normalized row for an S-parameter table."""

    side: str
    out_mode: int
    in_mode: int
    value: complex
    magnitude: float
    phase_deg: float


@dataclass(frozen=True, slots=True)
class SceneArtists:
    """Matplotlib artists created by :func:`plot_scene`."""

    material: Any
    lines: tuple[Any, ...]
    legend: Any


_LINE_STYLES: Mapping[str, Mapping[str, object]] = {
    "pec": {"color": "#f2c94c", "linestyle": "-", "label": "PEC"},
    "pmc": {"color": "#2f80ed", "linestyle": "-", "label": "PMC"},
    "wave_port": {"color": "#e53935", "linestyle": "-", "label": "Wave port"},
    "pml": {"color": "#27ae60", "linestyle": "--", "label": "PML interface"},
}


def _s_mapping(result_or_mapping: object) -> Mapping[object, object]:
    if isinstance(result_or_mapping, Mapping):
        return result_or_mapping
    value = getattr(result_or_mapping, "s_parameters", None)
    if not isinstance(value, Mapping):
        raise TypeError("Expected an S-parameter mapping or result.s_parameters mapping.")
    return value


def _normalize_s_key(key: object) -> SKey:
    if not isinstance(key, tuple) or len(key) != 3:
        raise ValueError("S-parameter keys must be (side, out_mode, in_mode) tuples.")
    side, out_mode, in_mode = key
    if not isinstance(side, str) or not side.strip():
        raise ValueError("S-parameter side must be a nonempty string.")
    for value, name in ((out_mode, "out_mode"), (in_mode, "in_mode")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"S-parameter {name} must be a nonnegative integer.")
    return side.strip().lower(), int(out_mode), int(in_mode)


def _side_order(side: str) -> tuple[int, str]:
    return ({"left": 0, "right": 1}.get(side, 2), side)


def _s_key_order(key: SKey) -> tuple[int, str, int, int]:
    side_rank, side = _side_order(key[0])
    return side_rank, side, key[1], key[2]


def s_parameter_label(key: SKey) -> str:
    """Return an unambiguous display label for one modal S-parameter key."""

    side, out_mode, in_mode = _normalize_s_key(key)
    conventional = ""
    if out_mode == 0 and in_mode == 0 and side in {"left", "right"}:
        conventional = ("S11" if side == "left" else "S21") + " · "
    return f"{conventional}{side}[out={out_mode}, in={in_mode}]"


def s_parameter_rows(result_or_mapping: object) -> tuple[SParameterRow, ...]:
    """Return normalized, sorted numeric rows for an S-parameter table."""

    normalized: dict[SKey, complex] = {}
    for raw_key, raw_value in _s_mapping(result_or_mapping).items():
        key = _normalize_s_key(raw_key)
        value_array = np.asarray(raw_value)
        if value_array.shape != ():
            raise ValueError(f"S-parameter {key!r} must be a scalar.")
        value = complex(value_array.item())
        if not np.isfinite((value.real, value.imag)).all():
            raise ValueError(f"S-parameter {key!r} must be finite.")
        if key in normalized:
            raise ValueError(f"Duplicate normalized S-parameter key {key!r}.")
        normalized[key] = value
    return tuple(
        SParameterRow(
            side=key[0],
            out_mode=key[1],
            in_mode=key[2],
            value=value,
            magnitude=float(abs(value)),
            phase_deg=float(np.degrees(np.angle(value))),
        )
        for key, value in sorted(normalized.items(), key=lambda item: _s_key_order(item[0]))
    )


def _s_plot_values(
    values: NDArray[np.complex128], quantity: str
) -> tuple[NDArray[np.float64], str]:
    if quantity in {"magnitude_db", "db"}:
        plotted = 20.0 * np.log10(np.maximum(np.abs(values), 1e-15))
        return np.asarray(plotted, dtype=float), "Magnitude (dB)"
    if quantity in {"magnitude", "abs"}:
        return np.asarray(np.abs(values), dtype=float), "Magnitude"
    if quantity in {"phase_deg", "phase"}:
        return np.asarray(np.degrees(np.angle(values)), dtype=float), "Phase (deg)"
    if quantity == "real":
        return np.asarray(values.real, dtype=float), "Real part"
    if quantity == "imag":
        return np.asarray(values.imag, dtype=float), "Imaginary part"
    raise ValueError(
        "quantity must be 'magnitude_db', 'magnitude', 'phase_deg', 'real', or 'imag'."
    )


def plot_s_parameters(
    ax: Any,
    frequencies_hz: ArrayLike,
    results: Sequence[object],
    *,
    quantity: SPlotQuantity | str = "magnitude_db",
    keys: Sequence[SKey] | None = None,
) -> tuple[Any, ...]:
    """Plot indexed S-parameters against ordinary frequency in hertz.

    Missing keys at an individual frequency become ``NaN`` gaps.  The return
    value is the tuple of created Matplotlib line artists.
    """

    frequency_raw = np.asarray(frequencies_hz)
    if np.iscomplexobj(frequency_raw) and np.any(np.imag(frequency_raw) != 0.0):
        raise ValueError("frequencies_hz must be real.")
    try:
        frequencies = np.asarray(np.real(frequency_raw), dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("frequencies_hz must be a one-dimensional real array.") from exc
    if frequencies.size != len(results):
        raise ValueError("frequencies_hz and results must have the same length.")
    if frequencies.size == 0 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz must contain finite values.")

    mappings: list[dict[SKey, complex]] = []
    union: set[SKey] = set()
    for result in results:
        mapping = {
            (row.side, row.out_mode, row.in_mode): row.value
            for row in s_parameter_rows(result)
        }
        mappings.append(mapping)
        union.update(mapping)
    selected = (
        tuple(sorted(union, key=_s_key_order))
        if keys is None
        else tuple(_normalize_s_key(key) for key in keys)
    )

    lines: list[Any] = []
    ylabel = _s_plot_values(np.ones(1, dtype=np.complex128), quantity)[1]
    for key in selected:
        values = np.asarray(
            [mapping.get(key, np.nan + 1j * np.nan) for mapping in mappings],
            dtype=np.complex128,
        )
        plotted, ylabel = _s_plot_values(values, quantity)
        (line,) = ax.plot(
            frequencies,
            plotted,
            marker="o" if frequencies.size == 1 else None,
            label=s_parameter_label(key),
        )
        lines.append(line)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title("Modal S-parameters")
    ax.grid(True, alpha=0.3)
    if lines:
        ax.legend(loc="best")
    return tuple(lines)


def plot_s_parameter_sweep(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Alias for :func:`plot_s_parameters`."""

    return plot_s_parameters(*args, **kwargs)


def _validated_field(
    x: ArrayLike, field: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    x_raw = np.asarray(x)
    if np.iscomplexobj(x_raw) and np.any(np.imag(x_raw) != 0.0):
        raise ValueError("x coordinates must be real.")
    try:
        x_values = np.asarray(np.real(x_raw), dtype=float).reshape(-1)
        field_values = np.asarray(field, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("x and field must contain numeric values.") from exc
    if x_values.size == 0 or field_values.shape != (3, x_values.size):
        raise ValueError(f"field must have shape (3, {x_values.size}).")
    if not np.isfinite(x_values).all() or not np.isfinite(field_values).all():
        raise ValueError("x and field values must be finite.")
    return x_values, field_values


def _component_index(component: str, field_name: str) -> int | None:
    normalized = str(component).strip().lower()
    prefix = str(field_name).strip().lower()
    if prefix and normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    if normalized in {"norm", "magnitude", "abs", ""}:
        return None
    if normalized not in {"x", "y", "z"}:
        raise ValueError("component must be 'x', 'y', 'z', or 'norm'.")
    return {"x": 0, "y": 1, "z": 2}[normalized]


def plot_modal_field(
    ax: Any,
    x: ArrayLike,
    field: ArrayLike,
    *,
    field_name: Literal["E", "H"] | str = "E",
    component: Literal["x", "y", "z", "norm"] | str = "norm",
    quantity: FieldQuantity | str = "abs",
) -> Any:
    """Plot one sampled modal E or H component on its transverse x grid."""

    x_values, field_values = _validated_field(x, field)
    index = _component_index(component, field_name)
    if quantity not in {"abs", "real", "imag"}:
        raise ValueError("quantity must be 'abs', 'real', or 'imag'.")
    if index is None:
        selected = {
            "abs": np.sqrt(np.sum(np.abs(field_values) ** 2, axis=0)),
            "real": np.sqrt(np.sum(field_values.real**2, axis=0)),
            "imag": np.sqrt(np.sum(field_values.imag**2, axis=0)),
        }[quantity]
        component_label = "norm"
    else:
        scalar = field_values[index]
        selected = {
            "abs": np.abs(scalar),
            "real": scalar.real,
            "imag": scalar.imag,
        }[quantity]
        component_label = "xyz"[index]
    (line,) = ax.plot(x_values, np.asarray(selected, dtype=float))
    ax.set_xlabel("x (m)")
    ax.set_ylabel(f"{quantity}({field_name}_{component_label})")
    ax.set_title(f"Modal {field_name} field")
    ax.grid(True, alpha=0.3)
    return line


def plot_scene(ax: Any, scene: SceneData) -> SceneArtists:
    """Draw a saved material mesh and its boundary/port overlays.

    The physical z coordinate is horizontal and x is vertical.  Triangle
    shading uses a grey-only map of ``Re(eps_r)``.  PEC is yellow, PMC blue,
    wave ports red, and PML interfaces are dashed green.
    """

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    dielectric_map = LinearSegmentedColormap.from_list(
        "wavefem_dielectric", ("#eeeeee", "#777777")
    )
    scalar = np.asarray(scene.eps_r.real, dtype=float)
    value_min = float(np.min(scalar))
    value_max = float(np.max(scalar))
    if np.isclose(value_min, value_max):
        value_min -= 0.5
        value_max += 0.5
    material = ax.tripcolor(
        scene.points[1],
        scene.points[0],
        triangles=scene.triangles.T,
        facecolors=scalar,
        shading="flat",
        cmap=dielectric_map,
        vmin=value_min,
        vmax=value_max,
        edgecolors="none",
        alpha=0.78,
        zorder=0,
    )

    line_artists: list[Any] = []
    present: list[str] = []
    for boundary in scene.lines:
        style = _LINE_STYLES[boundary.kind]
        endpoints = np.asarray(boundary.endpoints)
        (artist,) = ax.plot(
            endpoints[:, 1],
            endpoints[:, 0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            zorder=4,
        )
        line_artists.append(artist)
        if boundary.kind not in present:
            present.append(boundary.kind)

    handles: list[Any] = [
        Patch(facecolor="#a7a7a7", edgecolor="none", alpha=0.78, label="Dielectric")
    ]
    for kind in ("pec", "pmc", "wave_port", "pml"):
        if kind not in present:
            continue
        style = _LINE_STYLES[kind]
        handles.append(
            Line2D(
                (0,),
                (0,),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                label=style["label"],
            )
        )
    legend = ax.legend(handles=handles, loc="best")
    ax.set_xlim(float(scene.z_span[0]), float(scene.z_span[1]))
    ax.set_ylim(float(scene.x_span[0]), float(scene.x_span[1]))
    return SceneArtists(material=material, lines=tuple(line_artists), legend=legend)


def _validated_vector_samples(
    coordinates: ArrayLike, field: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    coordinates_raw = np.asarray(coordinates)
    if np.iscomplexobj(coordinates_raw) and np.any(np.imag(coordinates_raw) != 0.0):
        raise ValueError("coordinates must be real.")
    try:
        coordinate_values = np.asarray(np.real(coordinates_raw), dtype=float)
        field_values = np.asarray(field, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("coordinates and field must contain numeric values.") from exc
    if coordinate_values.ndim != 2 or coordinate_values.shape[0] != 2:
        raise ValueError("coordinates must have shape (2, npoints).")
    npoints = coordinate_values.shape[1]
    if npoints == 0 or field_values.shape != (3, npoints):
        raise ValueError(f"field must have shape (3, {npoints}).")
    if not np.isfinite(coordinate_values).all() or not np.isfinite(field_values).all():
        raise ValueError("coordinates and field values must be finite.")
    return coordinate_values, field_values


def plot_vector_field_2d(
    ax: Any,
    coordinates: ArrayLike,
    field: ArrayLike,
    *,
    field_name: Literal["E", "H"] | str = "E",
    quantity: Literal["real", "imag"] | str = "real",
    max_arrows: int = 900,
    scene: SceneData | None = None,
) -> Any:
    """Plot an E or H vector field with z horizontal and x vertical.

    Input coordinates remain in the solver's ``(x, z)`` row order.  Quiver's
    horizontal component is ``E_z``/``H_z`` and its vertical component is
    ``E_x``/``H_x``.  Duplicate sample coordinates are averaged, and the
    optional scene is rendered behind the arrows.
    """

    coordinate_values, field_values = _validated_vector_samples(coordinates, field)
    if quantity not in {"real", "imag"}:
        raise ValueError("quantity must be 'real' or 'imag'.")
    if (
        isinstance(max_arrows, bool)
        or not isinstance(max_arrows, (int, np.integer))
        or max_arrows < 1
    ):
        raise ValueError("max_arrows must be a positive integer.")
    if scene is not None:
        plot_scene(ax, scene)

    projected = field_values.real if quantity == "real" else field_values.imag
    points_xz = np.asarray(coordinate_values.T, dtype=float)
    unique_points, inverse = np.unique(points_xz, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    horizontal = np.bincount(inverse, weights=projected[2]) / counts
    vertical = np.bincount(inverse, weights=projected[0]) / counts
    if unique_points.shape[0] > int(max_arrows):
        keep = np.linspace(
            0, unique_points.shape[0] - 1, int(max_arrows), dtype=np.int64
        )
        unique_points = unique_points[keep]
        horizontal = horizontal[keep]
        vertical = vertical[keep]
    magnitude = np.hypot(horizontal, vertical)
    artist = ax.quiver(
        unique_points[:, 1],
        unique_points[:, 0],
        horizontal,
        vertical,
        magnitude,
        angles="xy",
        scale_units="xy",
        scale=None,
        cmap="viridis",
        zorder=3,
    )
    ax.set_xlabel("z (m)")
    ax.set_ylabel("x (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{quantity}({field_name}) in the z-x plane")
    if scene is not None:
        ax.set_xlim(float(scene.z_span[0]), float(scene.z_span[1]))
        ax.set_ylim(float(scene.x_span[0]), float(scene.x_span[1]))
    return artist


def plot_vector_field(*args: Any, **kwargs: Any) -> Any:
    """Alias for :func:`plot_vector_field_2d`."""

    return plot_vector_field_2d(*args, **kwargs)


__all__ = [
    "SParameterRow",
    "SceneArtists",
    "plot_modal_field",
    "plot_s_parameter_sweep",
    "plot_s_parameters",
    "plot_scene",
    "plot_vector_field",
    "plot_vector_field_2d",
    "s_parameter_label",
    "s_parameter_rows",
]
