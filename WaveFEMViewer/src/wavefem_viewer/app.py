"""Tk desktop application for browsing WaveFEM HDF5 files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .plotting import (
    plot_modal_field,
    plot_s_parameters,
    plot_vector_field_2d,
    s_parameter_rows,
)
from .reader import load_h5


def _format_frequency(value: float) -> str:
    if np.isnan(value):
        return "unknown frequency"
    magnitude = abs(value)
    for threshold, divisor, suffix in (
        (1e12, 1e12, "THz"),
        (1e9, 1e9, "GHz"),
        (1e6, 1e6, "MHz"),
        (1e3, 1e3, "kHz"),
    ):
        if magnitude >= threshold:
            return f"{value / divisor:.9g} {suffix}"
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


def _load_gui_runtime() -> _GuiRuntime:
    """Load Tk and TkAgg only when a desktop window is requested."""

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    import matplotlib

    matplotlib.use("TkAgg", force=True)
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    return _GuiRuntime(
        tk=tk,
        ttk=ttk,
        filedialog=filedialog,
        messagebox=messagebox,
        Figure=Figure,
        FigureCanvasTkAgg=FigureCanvasTkAgg,
        NavigationToolbar2Tk=NavigationToolbar2Tk,
    )


class H5ViewerApp:
    """Interactive Tk application for single-result and frequency-sweep files."""

    def __init__(self, root: Any, *, _runtime: _GuiRuntime | None = None) -> None:
        self.runtime = _load_gui_runtime() if _runtime is None else _runtime
        self.root = root
        self.data: Any | None = None
        self.path: Path | None = None
        self.current_index = 0
        self._modal_tabs: dict[str, dict[str, Any]] = {}
        self._vector_tabs: dict[str, dict[str, Any]] = {}

        root.title("WaveFEM Viewer")
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
        for field_name in ("E", "H"):
            self._build_modal_tab(field_name)
        for field_name in ("E", "H"):
            self._build_vector_tab(field_name)

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
        controls = ttk.Frame(tab, padding=4)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Plot:").pack(side=tk.LEFT)
        self.s_quantity_var = tk.StringVar(value="magnitude_db")
        quantity_combo = ttk.Combobox(
            controls,
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
        for combo in (mode_combo, component_combo, quantity_combo):
            combo.bind(
                "<<ComboboxSelected>>",
                lambda _event, name=field_name: self._draw_modal_tab(name),
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
        for combo in (part_combo, quantity_combo):
            combo.bind(
                "<<ComboboxSelected>>",
                lambda _event, name=field_name: self._draw_vector_tab(name),
            )

    def open_file(self) -> None:
        """Show a file picker and load the selected WaveFEM HDF5 file."""

        path = self.runtime.filedialog.askopenfilename(
            title="Open WaveFEM HDF5 result",
            filetypes=(("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")),
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str | Path, *, show_error: bool = True) -> bool:
        """Load a file, refresh every tab, and return whether loading succeeded."""

        try:
            loaded = load_h5(path)
            frequencies = np.asarray(loaded.frequencies_hz, dtype=float).reshape(-1)
            if frequencies.size == 0 or frequencies.size != len(loaded.results):
                raise ValueError("The file has inconsistent frequency/result counts.")
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
        axes.set_axis_on()
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
        stored = np.asarray(self.data.frequencies_hz, dtype=float)
        known = np.isfinite(stored).all()
        plotted = stored if known else np.arange(stored.size, dtype=float)
        plot_s_parameters(
            axes, plotted, self.data.results, quantity=self.s_quantity_var.get()
        )
        if not known:
            axes.set_xlabel("Saved result (frequency unknown)")
            axes.set_xticks(plotted, ("unknown",))
        axes.axvline(
            float(plotted[self.current_index]),
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
        details = [f"mode {index}"]
        neff = metadata.get("neff")
        if neff is not None:
            details.append(f"neff={neff:.7g}")
        for name in ("direction", "classification"):
            if metadata.get(name) is not None:
                details.append(str(metadata[name]))
        return " · ".join(details)

    def _refresh_mode_choices(self, field_name: str) -> None:
        state = self._modal_tabs[field_name]
        result = self._current_result()
        modes = () if result is None else tuple(result.modes)
        previous = state["mode_combo"].current()
        labels = tuple(self._mode_label(index, mode) for index, mode in enumerate(modes))
        state["mode_combo"].configure(
            values=labels, state="readonly" if labels else "disabled"
        )
        if labels:
            state["mode_combo"].current(min(max(previous, 0), len(labels) - 1))
        else:
            state["mode_var"].set("")

    def _draw_modal_tab(self, field_name: str) -> None:
        state = self._modal_tabs[field_name]
        axes = state["plot"]["axes"]
        axes.clear()
        axes.set_axis_on()
        result = self._current_result()
        modes = () if result is None else tuple(result.modes)
        mode_index = state["mode_combo"].current()
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
        axes.set_axis_on()
        result = self._current_result()
        if result is None:
            self._empty_axes(axes, f"2D Vector {field_name}", "Open an HDF5 result file")
        else:
            part = state["part_var"].get()
            plot_vector_field_2d(
                axes,
                result.coordinates,
                getattr(result, f"{field_name}_{part}"),
                field_name=field_name,
                quantity=state["quantity_var"].get(),
                scene=result.scene,
            )
            axes.set_title(f"{part} {field_name} · {state['quantity_var'].get()}")
        state["plot"]["figure"].tight_layout()
        state["plot"]["canvas"].draw_idle()


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the viewer and optionally open an HDF5 path immediately."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="wavefem-viewer",
        description="Open and visualize WaveFEM HDF5 result files.",
    )
    parser.add_argument("path", nargs="?", type=Path, help="optional .h5/.hdf5 file")
    arguments = parser.parse_args(argv)
    runtime = _load_gui_runtime()
    root = runtime.tk.Tk()
    viewer = H5ViewerApp(root, _runtime=runtime)
    if arguments.path is not None:
        viewer.load_path(arguments.path)
    root.mainloop()
    return 0


__all__ = ["H5ViewerApp", "main"]
