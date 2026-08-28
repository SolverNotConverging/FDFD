"""Matplotlib-widget FEM transmission-line calculator.

No window is created when this module is imported.  Construct
``TransmissionLineCalculatorGUI`` (or call the launch helper) only when an
interactive calculator is wanted.  The same object works under a headless Agg
backend for examples and automated tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .visualization import _draw_transverse_fields


@dataclass(frozen=True, slots=True)
class _EntrySpec:
    key: str
    label: str
    default: float
    si_scale: float
    strictly_positive: bool = True


_GHZ = 1.0e9
_MM = 1.0e-3
_UM = 1.0e-6

_COMMON_PREFIX = (_EntrySpec("frequency", "Frequency (GHz)", 10.0, _GHZ),)
_COMMON_SUFFIX = (
    _EntrySpec("epsilon_r", "Relative permittivity", 3.55, 1.0),
    _EntrySpec("loss_tangent", "Loss tangent", 0.0027, 1.0, False),
    _EntrySpec("max_element_size", "Mesh size (mm)", 0.12, _MM),
)

_LINE_ENTRIES: dict[str, tuple[str, tuple[_EntrySpec, ...]]] = {
    "Coaxial": (
        "coaxial",
        _COMMON_PREFIX
        + (
            _EntrySpec("inner_radius", "Inner radius (mm)", 0.5, _MM),
            _EntrySpec("outer_radius", "Outer radius (mm)", 1.67, _MM),
            _EntrySpec(
                "outer_conductor_thickness",
                "Outer metal (um)",
                150.0,
                _UM,
            ),
        )
        + (
            _EntrySpec("epsilon_r", "Relative permittivity", 2.1, 1.0),
            _EntrySpec("loss_tangent", "Loss tangent", 0.0002, 1.0, False),
            _EntrySpec("max_element_size", "Mesh size (mm)", 0.12, _MM),
        ),
    ),
    "Microstrip": (
        "microstrip",
        _COMMON_PREFIX
        + (
            _EntrySpec("trace_width", "Trace width (mm)", 3.0, _MM),
            _EntrySpec("substrate_height", "Substrate h (mm)", 1.524, _MM),
            _EntrySpec("conductor_thickness", "Metal thick. (um)", 35.0, _UM),
            _EntrySpec("domain_padding_factor", "Domain padding (x)", 1.0, 1.0),
        )
        + _COMMON_SUFFIX,
    ),
    "Stripline": (
        "stripline",
        _COMMON_PREFIX
        + (
            _EntrySpec("trace_width", "Trace width (mm)", 0.8, _MM),
            _EntrySpec("ground_spacing", "Ground gap (mm)", 1.524, _MM),
            _EntrySpec("conductor_thickness", "Metal thick. (um)", 35.0, _UM),
            _EntrySpec("domain_padding_factor", "Domain padding (x)", 1.0, 1.0),
        )
        + _COMMON_SUFFIX,
    ),
    "CPW odd (signal to tied grounds)": (
        "coplanar_waveguide",
        _COMMON_PREFIX
        + (
            _EntrySpec("center_width", "Signal width (mm)", 0.6, _MM),
            _EntrySpec("gap", "Slot gap (mm)", 0.25, _MM),
            _EntrySpec("ground_width", "Ground width (mm)", 1.5, _MM),
            _EntrySpec("substrate_height", "Substrate h (mm)", 0.8, _MM),
            _EntrySpec("conductor_thickness", "Metal thick. (um)", 35.0, _UM),
            _EntrySpec("domain_padding_factor", "Domain padding (x)", 1.0, 1.0),
        )
        + _COMMON_SUFFIX,
    ),
}


def _format_complex(value: Any, digits: int = 7) -> str:
    number = complex(value)
    threshold = 10.0 ** (-(digits - 1)) * max(1.0, abs(number.real))
    if abs(number.imag) <= threshold:
        return f"{number.real:.{digits}g}"
    return f"{number.real:.{digits}g}{number.imag:+.{digits}g}j"


def _format_real_or_complex(value: Any, unit: str) -> str:
    number = complex(value)
    return f"{_format_complex(number)} {unit}".rstrip()


class TransmissionLineCalculatorGUI:
    """Interactive finite-element transmission-line calculator.

    Dimensions are entered in the units printed next to each field and are
    converted to SI before being passed to ``TransmissionLineCalculator``.
    CPW is intentionally restricted to its odd quasi-TEM mode: the center
    signal is driven against the two tied ground conductors.
    """

    line_labels = tuple(_LINE_ENTRIES)

    def __init__(self, *, show: bool = False) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, TextBox

        self.figure = plt.figure(figsize=(14.0, 8.0))
        self.calculator: Any | None = None
        self.result: Any | None = None
        self.line_label = self.line_labels[0]
        self.line_kind = _LINE_ENTRIES[self.line_label][0]
        self._active_specs: tuple[_EntrySpec, ...] = ()
        self._suspend_entry_callbacks = False

        selector_axes = self.figure.add_axes((0.015, 0.725, 0.225, 0.215))
        self.line_control = RadioButtons(
            selector_axes,
            self.line_labels,
            active=0,
        )
        selector_axes.set_title("Transmission line", fontsize=10)

        maximum_entries = max(len(entry_specs) for _, entry_specs in _LINE_ENTRIES.values())
        self._entry_rows: list[tuple[Any, Any]] = []
        for index in range(maximum_entries):
            entry_axes = self.figure.add_axes((0.135, 0.682 - 0.052 * index, 0.102, 0.034))
            box = TextBox(entry_axes, "", initial="")
            box.label.set_fontsize(8.5)
            box.on_text_change(
                lambda _text, edited_box=box: self._parameter_edited(edited_box)
            )
            self._entry_rows.append((entry_axes, box))
        self.parameter_boxes: dict[str, Any] = {}

        calculate_axes = self.figure.add_axes((0.018, 0.115, 0.102, 0.048))
        refine_axes = self.figure.add_axes((0.137, 0.115, 0.102, 0.048))
        self.calculate_button = Button(calculate_axes, "Calculate FEM")
        self.refine_button = Button(refine_axes, "Refine x2")

        self.axes = np.asarray(
            [
                self.figure.add_axes((0.285, 0.285, 0.240, 0.60)),
                self.figure.add_axes((0.665, 0.285, 0.240, 0.60)),
            ],
            dtype=object,
        )
        self.colorbar_axes = np.asarray(
            [
                self.figure.add_axes((0.540, 0.335, 0.014, 0.50)),
                self.figure.add_axes((0.920, 0.335, 0.014, 0.50)),
            ],
            dtype=object,
        )
        for colorbar_axes in self.colorbar_axes:
            colorbar_axes.set_visible(False)
        for axes, family in zip(self.axes, ("E", "H"), strict=True):
            axes.text(
                0.5,
                0.5,
                f"{family}-field appears after calculation",
                ha="center",
                va="center",
                color="0.45",
                transform=axes.transAxes,
            )
            axes.set_xticks(())
            axes.set_yticks(())

        self.status_text = self.figure.text(
            0.018,
            0.080,
            "Ready",
            fontsize=9,
            color="0.25",
            va="top",
            wrap=True,
        )
        self.results_text = self.figure.text(
            0.285,
            0.185,
            "Enter the line dimensions, then calculate.",
            fontsize=10,
            family="monospace",
            va="top",
        )

        self.line_control.on_clicked(self._select_line)
        self.calculate_button.on_clicked(self._calculate_clicked)
        self.refine_button.on_clicked(self._refine_clicked)
        self._configure_entries(self.line_label)
        self._update_heading()
        if show:
            self.show()

    def _update_heading(self) -> None:
        self.figure.suptitle(f"FEM transmission-line calculator — {self.line_label}")

    def _configure_entries(self, label: str) -> None:
        callbacks_were_suspended = self._suspend_entry_callbacks
        self._suspend_entry_callbacks = True
        try:
            self.line_label = label
            self.line_kind, self._active_specs = _LINE_ENTRIES[label]
            self.parameter_boxes = {}
            for index, (entry_axes, box) in enumerate(self._entry_rows):
                if index >= len(self._active_specs):
                    entry_axes.set_visible(False)
                    continue
                spec = self._active_specs[index]
                entry_axes.set_visible(True)
                box.label.set_text(spec.label)
                box.set_val(f"{spec.default:g}")
                self.parameter_boxes[spec.key] = box
        finally:
            self._suspend_entry_callbacks = callbacks_were_suspended

    def _select_line(self, label: str) -> None:
        if label == self.line_label:
            return
        self._configure_entries(label)
        self._update_heading()
        self._invalidate_solution("Line changed; calculate the new geometry.")

    def _parameter_edited(self, edited_box: Any) -> None:
        if self._suspend_entry_callbacks:
            return
        if not any(edited_box is box for box in self.parameter_boxes.values()):
            return
        self._invalidate_solution("Parameters changed; calculate again.")

    def _invalidate_solution(self, status: str) -> None:
        """Discard every artifact tied to the previous input values."""

        self.calculator = None
        self.result = None
        self.results_text.set_text("Enter the line dimensions, then calculate.")
        for axes, family in zip(self.axes, ("E", "H"), strict=True):
            axes.clear()
            axes.text(
                0.5,
                0.5,
                f"{family}-field appears after calculation",
                ha="center",
                va="center",
                color="0.45",
                transform=axes.transAxes,
            )
            axes.set_xticks(())
            axes.set_yticks(())
        for colorbar_axes in self.colorbar_axes:
            colorbar_axes.clear()
            colorbar_axes.set_visible(False)
        self._set_status(status)
        self.figure.canvas.draw_idle()

    def _read_inputs(self) -> tuple[float, float, dict[str, float]]:
        values: dict[str, float] = {}
        for spec in self._active_specs:
            raw = self.parameter_boxes[spec.key].text.strip()
            try:
                displayed = float(raw)
            except ValueError as exc:
                raise ValueError(f"{spec.label} must be a number; received {raw!r}.") from exc
            if not np.isfinite(displayed):
                raise ValueError(f"{spec.label} must be finite.")
            if spec.strictly_positive and displayed <= 0.0:
                raise ValueError(f"{spec.label} must be greater than zero.")
            if not spec.strictly_positive and displayed < 0.0:
                raise ValueError(f"{spec.label} must not be negative.")
            values[spec.key] = displayed * spec.si_scale

        frequency = values.pop("frequency")
        mesh_size = values.pop("max_element_size")
        return frequency, mesh_size, values

    @staticmethod
    def _calculator_class() -> Any:
        # Kept lazy so importing this GUI cannot initialize a solver backend or
        # make optional native meshing libraries a GUI-import requirement.
        from .calculator import TransmissionLineCalculator

        return TransmissionLineCalculator

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self.status_text.set_text(str(message))
        self.status_text.set_color("crimson" if error else "0.25")

    def calculate(self) -> Any | None:
        """Build, discretize, and synchronously solve the selected line."""

        try:
            frequency, mesh_size, parameters = self._read_inputs()
            self._set_status("Meshing and solving ...")
            self.figure.canvas.draw_idle()
            calculator_type = self._calculator_class()
            calculator = calculator_type.from_type(
                self.line_kind,
                frequency=frequency,
                **parameters,
            )
            calculator.discretize(
                max_element_size=mesh_size,
                boundary_refinement=0.4,
            )
            solved = calculator.solve()
            result = solved if solved is not None else calculator.solution
            if result is None:
                raise RuntimeError("The FEM calculator completed without a solution.")
            self.calculator = calculator
            self.result = result
            self._show_result(result)
            self._set_status("Solved. Refine x2 halves the current FEM element size.")
            self.figure.canvas.draw_idle()
            return result
        except Exception as exc:  # GUI callbacks must report, not terminate, the event loop.
            self._set_status(f"Error: {exc}", error=True)
            self.figure.canvas.draw_idle()
            return None

    def refine(self) -> Any | None:
        """Refine the current calculator by two and solve it again."""

        if self.calculator is None:
            self._set_status("Calculate a line before requesting refinement.", error=True)
            self.figure.canvas.draw_idle()
            return None
        try:
            self._set_status("Refining mesh x2 and solving ...")
            self.figure.canvas.draw_idle()
            self.calculator.refine(2.0)
            solved = self.calculator.solve()
            result = solved if solved is not None else self.calculator.solution
            if result is None:
                raise RuntimeError("The refined FEM solve completed without a solution.")
            self.result = result
            self._show_result(result)
            self._set_status("Refined FEM solution complete.")
            self.figure.canvas.draw_idle()
            return result
        except Exception as exc:  # keep callback failures inside the calculator window
            self._set_status(f"Error: {exc}", error=True)
            self.figure.canvas.draw_idle()
            return None

    def _calculate_clicked(self, _event: Any) -> None:
        self.calculate()

    def _refine_clicked(self, _event: Any) -> None:
        self.refine()

    def _show_result(self, result: Any) -> None:
        capacitance = _format_real_or_complex(result.capacitance_per_length, "F/m")
        inductance = _format_real_or_complex(result.inductance_per_length, "H/m")
        power = _format_real_or_complex(result.power, "W")
        self.results_text.set_text(
            "  ".join(
                (
                    f"n_eff = {_format_complex(result.neff)}",
                    f"Zc = {_format_complex(result.characteristic_impedance)} ohm",
                    f"Zwave = {_format_complex(result.wave_impedance)} ohm",
                )
            )
            + "\n"
            + "  ".join((f"C' = {capacitance}", f"L' = {inductance}", f"P = {power}"))
        )
        _draw_transverse_fields(
            result,
            self.axes,
            self.colorbar_axes,
            phase=0.0,
            mesh=False,
            update_title=False,
        )

    def show(self, *, block: bool | None = None) -> None:
        import matplotlib.pyplot as plt

        if block is None:
            plt.show()
        else:
            plt.show(block=bool(block))

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.figure)


def launch_transmission_line_calculator(
    *,
    show: bool = True,
) -> TransmissionLineCalculatorGUI:
    """Construct and optionally show the FEM transmission-line calculator."""

    return TransmissionLineCalculatorGUI(show=show)


def main() -> None:
    """Console-script entry point for the interactive calculator."""

    launch_transmission_line_calculator(show=True)


__all__ = [
    "TransmissionLineCalculatorGUI",
    "launch_transmission_line_calculator",
    "main",
]
