"""Interactive HDF5 result viewer and backend-neutral plotting helpers.

Importing this module does not import Tkinter, select a Matplotlib backend, or
create a display.  The GUI runtime is loaded only by :class:`H5ViewerApp` or
:func:`main`; the plotting helpers operate on caller-provided Matplotlib axes
and are therefore usable with the non-interactive ``Agg`` backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


SKey = tuple[str, int, int]
SPlotQuantity = Literal["magnitude_db", "magnitude", "phase_deg", "real", "imag"]
FieldQuantity = Literal["abs", "real", "imag"]


@dataclass(frozen=True, slots=True)
class SParameterRow:
    """One S-parameter table row in the HDF5 viewer."""

    side: str
    out_mode: int
    in_mode: int
    value: complex
    magnitude: float
    phase_deg: float


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
    """Return an unambiguous label for one indexed modal S-parameter."""

    side, out_mode, in_mode = _normalize_s_key(key)
    conventional = ""
    if out_mode == 0 and in_mode == 0 and side in {"left", "right"}:
        conventional = "S11" if side == "left" else "S21"
        conventional += " · "
    return f"{conventional}{side}[out={out_mode}, in={in_mode}]"


def s_parameter_rows(result_or_mapping: object) -> tuple[SParameterRow, ...]:
    """Return sorted numeric rows for an S-parameter table."""

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


def _s_plot_values(values: NDArray[np.complex128], quantity: str) -> tuple[NDArray[np.float64], str]:
    if quantity in {"magnitude_db", "db"}:
        # A finite display floor keeps exact zeros useful on an ordinary axis.
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
    """Plot indexed S-parameters across one or more saved frequencies.

    Missing keys at individual frequencies are represented by ``NaN`` gaps.
    Frequencies remain in hertz so the plotted data exactly match the file.
    The created Matplotlib line objects are returned.
    """

    frequency_array = np.asarray(frequencies_hz)
    if np.iscomplexobj(frequency_array) and np.any(np.imag(frequency_array) != 0.0):
        raise ValueError("frequencies_hz must be real.")
    try:
        frequency_array = np.asarray(np.real(frequency_array), dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("frequencies_hz must be a one-dimensional real array.") from exc
    if frequency_array.size != len(results):
        raise ValueError("frequencies_hz and results must have the same length.")
    if frequency_array.size == 0 or not np.isfinite(frequency_array).all():
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
        complex_values = np.asarray(
            [mapping.get(key, np.nan + 1j * np.nan) for mapping in mappings],
            dtype=np.complex128,
        )
        plotted, ylabel = _s_plot_values(complex_values, quantity)
        (line,) = ax.plot(
            frequency_array,
            plotted,
            marker="o" if frequency_array.size == 1 else None,
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


def _validated_field(x: ArrayLike, field: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
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
        normalized = normalized[len(prefix):]
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
) -> Any:
    """Plot the x-z projection of a complex vector field with ``quiver``.

    Duplicate FEM sampling coordinates are averaged for display, then evenly
    subsampled to at most ``max_arrows``.  The stored HDF5 data are untouched.
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

    projected = field_values.real if quantity == "real" else field_values.imag
    points = np.asarray(coordinate_values.T, dtype=float)
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    u = np.bincount(inverse, weights=projected[0]) / counts
    v = np.bincount(inverse, weights=projected[2]) / counts
    if unique_points.shape[0] > int(max_arrows):
        keep = np.linspace(
            0, unique_points.shape[0] - 1, int(max_arrows), dtype=np.int64
        )
        unique_points = unique_points[keep]
        u = u[keep]
        v = v[keep]
    magnitude = np.hypot(u, v)
    artist = ax.quiver(
        unique_points[:, 0],
        unique_points[:, 1],
        u,
        v,
        magnitude,
        angles="xy",
        scale_units="xy",
        scale=None,
        cmap="viridis",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{quantity}({field_name}) in the x-z plane")
    return artist


def plot_vector_field(*args: Any, **kwargs: Any) -> Any:
    """Alias for :func:`plot_vector_field_2d`."""

    return plot_vector_field_2d(*args, **kwargs)


def _format_frequency(value: float) -> str:
    if np.isnan(value):
        return "unknown frequency"
    magnitude = abs(value)
    if magnitude >= 1e12:
        return f"{value / 1e12:.9g} THz"
    if magnitude >= 1e9:
        return f"{value / 1e9:.9g} GHz"
    if magnitude >= 1e6:
        return f"{value / 1e6:.9g} MHz"
    if magnitude >= 1e3:
        return f"{value / 1e3:.9g} kHz"
    return f"{value:.9g} Hz"


@dataclass(frozen=True, slots=True)
class _GuiRuntime:
    tk: Any
    ttk: Any
    filedialog: Any
    messagebox: Any
    Figure: Any
    FigureCanvasTkAgg: Any
    NavigationToolbar2Tk: Any
    load_h5: Any


def _load_gui_runtime() -> _GuiRuntime:
    """Import Tk/TkAgg and the HDF5 loader only when a GUI is requested."""

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    import matplotlib

    matplotlib.use("TkAgg", force=True)
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    from .hdf5 import load_h5

    return _GuiRuntime(
        tk=tk,
        ttk=ttk,
        filedialog=filedialog,
        messagebox=messagebox,
        Figure=Figure,
        FigureCanvasTkAgg=FigureCanvasTkAgg,
        NavigationToolbar2Tk=NavigationToolbar2Tk,
        load_h5=load_h5,
    )


class H5ViewerApp:
    """Tk application for browsing single-result and sweep WaveFEM HDF5 files."""

    def __init__(self, root: Any, *, _runtime: _GuiRuntime | None = None) -> None:
        self.runtime = _load_gui_runtime() if _runtime is None else _runtime
        self.root = root
        self.data: Any | None = None
        self.path: Path | None = None
        self.current_index = 0
        self._modal_tabs: dict[str, dict[str, Any]] = {}
        self._vector_tabs: dict[str, dict[str, Any]] = {}

        root.title("WaveFEM HDF5 Viewer")
        root.geometry("1280x820")
        self._build_ui()

    def _build_ui(self) -> None:
        ttk = self.runtime.ttk
        tk = self.runtime.tk

        controls = ttk.Frame(self.root, padding=6)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Open HDF5…", command=self.open_file).pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value="No file loaded")
        ttk.Label(controls, textvariable=self.path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Label(controls, text="Frequency:").pack(side=tk.LEFT, padx=(8, 3))
        self.frequency_var = tk.StringVar()
        self.frequency_combo = ttk.Combobox(
            controls,
            textvariable=self.frequency_var,
            state="disabled",
            width=28,
        )
        self.frequency_combo.pack(side=tk.LEFT)
        self.frequency_combo.bind("<<ComboboxSelected>>", self._on_frequency_selected)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self._build_s_tab()
        self._build_modal_tab("E")
        self._build_modal_tab("H")
        self._build_vector_tab("E")
        self._build_vector_tab("H")

    def _new_figure_panel(self, parent: Any) -> dict[str, Any]:
        tk = self.runtime.tk
        figure = self.runtime.Figure(figsize=(6.4, 4.8), dpi=100)
        axes = figure.add_subplot(111)
        canvas = self.runtime.FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = self.runtime.NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        return {"figure": figure, "axes": axes, "canvas": canvas, "toolbar": toolbar}

    def _build_s_tab(self) -> None:
        ttk = self.runtime.ttk
        tk = self.runtime.tk
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="S-parameters")

        top = ttk.Frame(tab, padding=4)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Plot:").pack(side=tk.LEFT)
        self.s_quantity_var = tk.StringVar(value="magnitude_db")
        quantity_combo = ttk.Combobox(
            top,
            textvariable=self.s_quantity_var,
            values=("magnitude_db", "magnitude", "phase_deg", "real", "imag"),
            state="readonly",
            width=16,
        )
        quantity_combo.pack(side=tk.LEFT, padx=4)
        quantity_combo.bind("<<ComboboxSelected>>", lambda _event: self._draw_s_tab())

        panes = ttk.Panedwindow(tab, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)
        table_frame = ttk.Frame(panes, padding=4)
        plot_frame = ttk.Frame(panes, padding=4)
        panes.add(table_frame, weight=1)
        panes.add(plot_frame, weight=3)

        columns = ("side", "out", "in", "complex", "magnitude", "phase")
        self.s_table = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "side": "Side",
            "out": "Out mode",
            "in": "In mode",
            "complex": "Complex amplitude",
            "magnitude": "|S|",
            "phase": "Phase (deg)",
        }
        for column in columns:
            self.s_table.heading(column, text=headings[column])
            self.s_table.column(column, width=105, anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.s_table.yview)
        self.s_table.configure(yscrollcommand=scrollbar.set)
        self.s_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.s_plot = self._new_figure_panel(plot_frame)

    def _build_modal_tab(self, field_name: str) -> None:
        ttk = self.runtime.ttk
        tk = self.runtime.tk
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=f"Modal {field_name}")
        controls = ttk.Frame(tab, padding=4)
        controls.pack(fill=tk.X)

        mode_var = tk.StringVar()
        component_var = tk.StringVar(value="norm")
        quantity_var = tk.StringVar(value="abs")
        ttk.Label(controls, text="Mode:").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(controls, textvariable=mode_var, state="disabled", width=42)
        mode_combo.pack(side=tk.LEFT, padx=4)
        ttk.Label(controls, text="Component:").pack(side=tk.LEFT, padx=(12, 0))
        component_combo = ttk.Combobox(
            controls,
            textvariable=component_var,
            values=("norm", "x", "y", "z"),
            state="readonly",
            width=8,
        )
        component_combo.pack(side=tk.LEFT, padx=4)
        ttk.Label(controls, text="Quantity:").pack(side=tk.LEFT, padx=(12, 0))
        quantity_combo = ttk.Combobox(
            controls,
            textvariable=quantity_var,
            values=("abs", "real", "imag"),
            state="readonly",
            width=8,
        )
        quantity_combo.pack(side=tk.LEFT, padx=4)

        plot_frame = ttk.Frame(tab, padding=4)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        state = {
            "mode_var": mode_var,
            "mode_combo": mode_combo,
            "component_var": component_var,
            "quantity_var": quantity_var,
            "plot": self._new_figure_panel(plot_frame),
        }
        self._modal_tabs[field_name] = state
        mode_combo.bind(
            "<<ComboboxSelected>>", lambda _event, name=field_name: self._draw_modal_tab(name)
        )
        component_combo.bind(
            "<<ComboboxSelected>>", lambda _event, name=field_name: self._draw_modal_tab(name)
        )
        quantity_combo.bind(
            "<<ComboboxSelected>>", lambda _event, name=field_name: self._draw_modal_tab(name)
        )

    def _build_vector_tab(self, field_name: str) -> None:
        ttk = self.runtime.ttk
        tk = self.runtime.tk
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=f"2D Vector {field_name}")
        controls = ttk.Frame(tab, padding=4)
        controls.pack(fill=tk.X)
        part_var = tk.StringVar(value="total")
        quantity_var = tk.StringVar(value="real")
        ttk.Label(controls, text="Field part:").pack(side=tk.LEFT)
        part_combo = ttk.Combobox(
            controls,
            textvariable=part_var,
            values=("total", "incident", "scattered"),
            state="readonly",
            width=12,
        )
        part_combo.pack(side=tk.LEFT, padx=4)
        ttk.Label(controls, text="Quantity:").pack(side=tk.LEFT, padx=(12, 0))
        quantity_combo = ttk.Combobox(
            controls,
            textvariable=quantity_var,
            values=("real", "imag"),
            state="readonly",
            width=8,
        )
        quantity_combo.pack(side=tk.LEFT, padx=4)
        plot_frame = ttk.Frame(tab, padding=4)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        state = {
            "part_var": part_var,
            "quantity_var": quantity_var,
            "plot": self._new_figure_panel(plot_frame),
        }
        self._vector_tabs[field_name] = state
        part_combo.bind(
            "<<ComboboxSelected>>", lambda _event, name=field_name: self._draw_vector_tab(name)
        )
        quantity_combo.bind(
            "<<ComboboxSelected>>", lambda _event, name=field_name: self._draw_vector_tab(name)
        )

    def open_file(self) -> None:
        path = self.runtime.filedialog.askopenfilename(
            title="Open WaveFEM HDF5 result",
            filetypes=(("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")),
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str | Path, *, show_error: bool = True) -> bool:
        """Load one HDF5 file and refresh every viewer tab."""

        try:
            loaded = self.runtime.load_h5(Path(path))
            frequencies = np.asarray(loaded.frequencies_hz, dtype=float).reshape(-1)
            results = tuple(loaded.results)
            if frequencies.size == 0 or frequencies.size != len(results):
                raise ValueError("The HDF5 file has inconsistent frequency/result counts.")
            if loaded.kind == "sweep" and not np.isfinite(frequencies).all():
                raise ValueError("The HDF5 file contains a non-finite frequency.")
            if loaded.kind == "single" and not np.all(
                np.isfinite(frequencies) | np.isnan(frequencies)
            ):
                raise ValueError("The HDF5 file contains an invalid frequency.")
        except Exception as exc:
            if show_error:
                self.runtime.messagebox.showerror("Could not open HDF5 file", str(exc))
            return False

        self.data = loaded
        self.path = Path(path)
        self.current_index = 0
        self.path_var.set(f"{self.path}  ({loaded.kind})")
        labels = tuple(
            f"{index}: {_format_frequency(float(frequency))}"
            for index, frequency in enumerate(frequencies)
        )
        self.frequency_combo.configure(values=labels, state="readonly")
        self.frequency_combo.current(0)
        self._refresh_all()
        return True

    def _on_frequency_selected(self, _event: object = None) -> None:
        selected = self.frequency_combo.current()
        if selected >= 0:
            self.current_index = selected
            self._refresh_all()

    def _current_result(self) -> Any | None:
        if self.data is None:
            return None
        return self.data.results[self.current_index]

    @staticmethod
    def _empty_axes(axes: Any, title: str, message: str) -> None:
        axes.clear()
        axes.set_title(title)
        axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()

    def _refresh_all(self) -> None:
        self._draw_s_tab()
        for field_name in ("E", "H"):
            self._refresh_mode_choices(field_name)
            self._draw_modal_tab(field_name)
            self._draw_vector_tab(field_name)

    def _draw_s_tab(self) -> None:
        axes = self.s_plot["axes"]
        axes.clear()
        for item in self.s_table.get_children():
            self.s_table.delete(item)
        result = self._current_result()
        if result is None or self.data is None:
            self._empty_axes(axes, "Modal S-parameters", "Open an HDF5 result file")
            self.s_plot["canvas"].draw_idle()
            return
        for row in s_parameter_rows(result):
            self.s_table.insert(
                "",
                "end",
                values=(
                    row.side,
                    row.out_mode,
                    row.in_mode,
                    f"{row.value.real:+.6e}{row.value.imag:+.6e}j",
                    f"{row.magnitude:.6e}",
                    f"{row.phase_deg:.4f}",
                ),
            )
        stored_frequencies = np.asarray(self.data.frequencies_hz, dtype=float)
        frequencies_known = np.isfinite(stored_frequencies).all()
        plotted_frequencies = (
            stored_frequencies
            if frequencies_known
            else np.arange(stored_frequencies.size, dtype=float)
        )
        plot_s_parameters(
            axes,
            plotted_frequencies,
            self.data.results,
            quantity=self.s_quantity_var.get(),
        )
        if not frequencies_known:
            axes.set_xlabel("Saved result (frequency unknown)")
            axes.set_xticks(plotted_frequencies, ("unknown",))
        axes.axvline(
            float(plotted_frequencies[self.current_index]),
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            label="selected",
        )
        axes.legend(loc="best")
        self.s_plot["figure"].tight_layout()
        self.s_plot["canvas"].draw_idle()

    @staticmethod
    def _mode_label(index: int, mode: object) -> str:
        metadata = getattr(mode, "metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}

        def modal_value(name: str) -> object | None:
            direct = getattr(mode, name, None)
            return direct if direct is not None else metadata.get(name)

        details = [f"mode {index}"]
        neff = modal_value("neff")
        if neff is not None:
            details.append(f"neff={neff:.7g}")
        for name in ("direction", "classification"):
            value = modal_value(name)
            if value is not None:
                details.append(str(value))
        return " · ".join(details)

    def _refresh_mode_choices(self, field_name: str) -> None:
        state = self._modal_tabs[field_name]
        result = self._current_result()
        modes = () if result is None else tuple(result.modes)
        previous = state["mode_combo"].current()
        labels = tuple(self._mode_label(index, mode) for index, mode in enumerate(modes))
        state["mode_combo"].configure(
            values=labels,
            state="readonly" if labels else "disabled",
        )
        if labels:
            state["mode_combo"].current(min(max(previous, 0), len(labels) - 1))
        else:
            state["mode_var"].set("")

    def _draw_modal_tab(self, field_name: str) -> None:
        state = self._modal_tabs[field_name]
        axes = state["plot"]["axes"]
        axes.clear()
        result = self._current_result()
        mode_index = state["mode_combo"].current()
        modes = () if result is None else tuple(result.modes)
        if mode_index < 0 or mode_index >= len(modes):
            self._empty_axes(axes, f"Modal {field_name} field", "No saved modes")
        else:
            mode = modes[mode_index]
            plot_modal_field(
                axes,
                mode.x,
                getattr(mode, field_name),
                field_name=field_name,
                component=state["component_var"].get(),
                quantity=state["quantity_var"].get(),
            )
            axes.set_title(f"{self._mode_label(mode_index, mode)} · Modal {field_name}")
        state["plot"]["figure"].tight_layout()
        state["plot"]["canvas"].draw_idle()

    def _draw_vector_tab(self, field_name: str) -> None:
        state = self._vector_tabs[field_name]
        axes = state["plot"]["axes"]
        axes.clear()
        result = self._current_result()
        if result is None:
            self._empty_axes(axes, f"2D Vector {field_name}", "Open an HDF5 result file")
        else:
            part = state["part_var"].get()
            attribute = f"{field_name}_{part}"
            plot_vector_field_2d(
                axes,
                result.coordinates,
                getattr(result, attribute),
                field_name=field_name,
                quantity=state["quantity_var"].get(),
            )
            axes.set_title(f"{part} {field_name} · {state['quantity_var'].get()}")
        state["plot"]["figure"].tight_layout()
        state["plot"]["canvas"].draw_idle()


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the Tk HDF5 viewer, optionally opening one path immediately."""

    import argparse

    parser = argparse.ArgumentParser(description="View WaveFEM HDF5 result files.")
    parser.add_argument("path", nargs="?", type=Path, help="optional .h5/.hdf5 file")
    arguments = parser.parse_args(argv)

    runtime = _load_gui_runtime()
    root = runtime.tk.Tk()
    viewer = H5ViewerApp(root, _runtime=runtime)
    if arguments.path is not None:
        viewer.load_path(arguments.path)
    root.mainloop()
    return 0


__all__ = [
    "H5ViewerApp",
    "SParameterRow",
    "main",
    "plot_modal_field",
    "plot_s_parameter_sweep",
    "plot_s_parameters",
    "plot_vector_field",
    "plot_vector_field_2d",
    "s_parameter_label",
    "s_parameter_rows",
]


if __name__ == "__main__":  # pragma: no cover - manual GUI entry point
    raise SystemExit(main())
