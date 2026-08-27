"""Headless tests for the HDF5 viewer's backend-neutral plot helpers."""

from __future__ import annotations

from types import SimpleNamespace
import subprocess
import sys

import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from wavefem.gui import (
    H5ViewerApp,
    plot_modal_field,
    plot_s_parameters,
    plot_vector_field_2d,
    s_parameter_rows,
)


def test_gui_module_import_does_not_load_tk_or_matplotlib() -> None:
    code = """
import sys
import wavefem.gui
assert 'tkinter' not in sys.modules
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_s_parameter_rows_are_numeric_normalized_and_sorted() -> None:
    rows = s_parameter_rows(
        {
            ("RIGHT", 1, 0): 0.5j,
            ("left", 0, 0): -1.0j,
            ("right", 0, 0): 0.8 + 0.6j,
        }
    )

    assert [(row.side, row.out_mode, row.in_mode) for row in rows] == [
        ("left", 0, 0),
        ("right", 0, 0),
        ("right", 1, 0),
    ]
    assert rows[0].magnitude == pytest.approx(1.0)
    assert rows[0].phase_deg == pytest.approx(-90.0)
    assert rows[1].value == pytest.approx(0.8 + 0.6j)


def test_s_parameter_sweep_plot_preserves_hz_and_marks_missing_samples() -> None:
    frequencies = np.asarray((1.0e9, 2.0e9, 3.0e9))
    results = (
        SimpleNamespace(
            s_parameters={
                ("left", 0, 0): 0.1,
                ("right", 0, 0): 1.0,
            }
        ),
        SimpleNamespace(s_parameters={("right", 0, 0): 0.5}),
        SimpleNamespace(
            s_parameters={
                ("left", 0, 0): 0.2j,
                ("right", 0, 0): 0.25,
            }
        ),
    )
    figure, axes = plt.subplots()
    try:
        lines = plot_s_parameters(
            axes,
            frequencies,
            results,
            quantity="magnitude_db",
        )

        assert len(lines) == 2
        np.testing.assert_array_equal(lines[0].get_xdata(), frequencies)
        np.testing.assert_allclose(
            lines[1].get_ydata(),
            20.0 * np.log10((1.0, 0.5, 0.25)),
        )
        assert np.isnan(lines[0].get_ydata()[1])
        assert lines[0].get_label().startswith("S11")
        assert lines[1].get_label().startswith("S21")
        assert axes.get_xlabel() == "Frequency (Hz)"
        assert axes.get_ylabel() == "Magnitude (dB)"
    finally:
        plt.close(figure)


def test_modal_plot_selects_complex_h_component() -> None:
    x = np.asarray((-1.0e-6, 0.0, 1.0e-6))
    field = np.asarray(
        (
            (1.0 + 2.0j, 2.0 + 3.0j, 3.0 + 4.0j),
            (4.0 + 5.0j, 5.0 + 6.0j, 6.0 + 7.0j),
            (7.0 + 8.0j, 8.0 + 9.0j, 9.0 + 10.0j),
        )
    )
    figure, axes = plt.subplots()
    try:
        line = plot_modal_field(
            axes,
            x,
            field,
            field_name="H",
            component="Hy",
            quantity="imag",
        )

        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), (5.0, 6.0, 7.0))
        assert axes.get_xlabel() == "x (m)"
        assert axes.get_ylabel() == "imag(H_y)"
        assert axes.get_title() == "Modal H field"
    finally:
        plt.close(figure)


def test_mode_label_reads_portable_h5_metadata() -> None:
    mode = SimpleNamespace(
        metadata={
            "neff": 1.75 + 0.0j,
            "direction": "forward",
            "classification": "propagating",
        }
    )

    label = H5ViewerApp._mode_label(2, mode)

    assert label.startswith("mode 2 · neff=1.75+0j")
    assert label.endswith("forward · propagating")


def test_vector_plot_uses_xz_components_and_averages_duplicate_points() -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    field = np.zeros((3, 4), dtype=np.complex128)
    field[0] = 1j * np.asarray((1.0, 3.0, 5.0, 7.0))
    field[1] = 1j * 100.0
    field[2] = 1j * np.asarray((2.0, 4.0, 6.0, 8.0))
    figure, axes = plt.subplots()
    try:
        quiver = plot_vector_field_2d(
            axes,
            coordinates,
            field,
            field_name="E",
            quantity="imag",
        )

        np.testing.assert_array_equal(quiver.X, (0.0, 1.0, 1.0))
        np.testing.assert_array_equal(quiver.Y, (0.0, 0.0, 1.0))
        np.testing.assert_allclose(quiver.U, (2.0, 5.0, 7.0))
        np.testing.assert_allclose(quiver.V, (3.0, 6.0, 8.0))
        np.testing.assert_allclose(quiver.get_array(), np.hypot(quiver.U, quiver.V))
        assert axes.get_xlabel() == "x (m)"
        assert axes.get_ylabel() == "z (m)"
        assert axes.get_title() == "imag(E) in the x-z plane"
    finally:
        plt.close(figure)


def test_plot_helpers_reject_inconsistent_data_shapes() -> None:
    figure, axes = plt.subplots()
    try:
        with pytest.raises(ValueError, match="same length"):
            plot_s_parameters(axes, (1.0, 2.0), (SimpleNamespace(s_parameters={}),))
        with pytest.raises(ValueError, match=r"shape \(3, 2\)"):
            plot_modal_field(axes, (0.0, 1.0), np.ones((2, 2)))
        with pytest.raises(ValueError, match="coordinates must have shape"):
            plot_vector_field_2d(axes, np.ones((3, 2)), np.ones((3, 2)))
    finally:
        plt.close(figure)
