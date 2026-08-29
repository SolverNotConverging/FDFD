from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
from matplotlib.colors import to_rgba


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from wavefem_viewer.model import SceneData, SceneLine
from wavefem_viewer.plotting import plot_scene, plot_s_parameters, plot_vector_field_2d


def make_scene() -> SceneData:
    points = np.asarray(
        (
            (-1.0, 1.0, 1.0, -1.0),
            (0.0, 0.0, 4.0, 4.0),
        )
    )
    triangles = np.asarray(((0, 0), (1, 2), (2, 3)), dtype=np.int64)
    return SceneData(
        points=points,
        triangles=triangles,
        eps_r=np.asarray((1.0, 4.0), dtype=np.complex128),
        x_span=np.asarray((-1.0, 1.0)),
        z_span=np.asarray((0.0, 4.0)),
        lines=(
            SceneLine("pec", np.asarray(((-1.0, 0.0), (-1.0, 4.0))), "PEC"),
            SceneLine("pmc", np.asarray(((1.0, 0.0), (1.0, 4.0))), "PMC"),
            SceneLine("wave_port", np.asarray(((-1.0, 1.0), (1.0, 1.0))), "port"),
            SceneLine("pml", np.asarray(((-1.0, 3.0), (1.0, 3.0))), "PML"),
        ),
    )


def test_vector_plot_uses_z_horizontal_x_vertical_and_swaps_components() -> None:
    coordinates = np.asarray(((10.0, 20.0), (100.0, 200.0)))
    field = np.zeros((3, 2), dtype=np.complex128)
    field[0] = (1.0, 2.0)
    field[1] = (90.0, 90.0)
    field[2] = (3.0, 4.0)
    figure, axes = plt.subplots()
    try:
        quiver = plot_vector_field_2d(axes, coordinates, field)
        np.testing.assert_array_equal(quiver.X, (100.0, 200.0))
        np.testing.assert_array_equal(quiver.Y, (10.0, 20.0))
        np.testing.assert_array_equal(quiver.U, (3.0, 4.0))
        np.testing.assert_array_equal(quiver.V, (1.0, 2.0))
        assert axes.get_xlabel() == "z (m)"
        assert axes.get_ylabel() == "x (m)"
        assert axes.get_title() == "real(E) in the z-x plane"
    finally:
        plt.close(figure)


def test_vector_plot_averages_duplicate_samples_after_axis_swap() -> None:
    coordinates = np.asarray(((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)))
    field = np.zeros((3, 4), dtype=np.complex128)
    field[0] = 1j * np.asarray((1.0, 3.0, 5.0, 7.0))
    field[2] = 1j * np.asarray((2.0, 4.0, 6.0, 8.0))
    figure, axes = plt.subplots()
    try:
        quiver = plot_vector_field_2d(axes, coordinates, field, quantity="imag")
        np.testing.assert_array_equal(quiver.X, (0.0, 0.0, 1.0))
        np.testing.assert_array_equal(quiver.Y, (0.0, 1.0, 1.0))
        np.testing.assert_allclose(quiver.U, (3.0, 6.0, 8.0))
        np.testing.assert_allclose(quiver.V, (2.0, 5.0, 7.0))
    finally:
        plt.close(figure)


def test_scene_draws_grey_material_styles_unique_legend_and_full_domain() -> None:
    scene = make_scene()
    figure, axes = plt.subplots()
    try:
        artists = plot_scene(axes, scene)
        assert artists.material.get_cmap().name == "wavefem_dielectric"
        assert len(artists.lines) == 4
        assert to_rgba(artists.lines[0].get_color()) == to_rgba("#f2c94c")
        assert to_rgba(artists.lines[1].get_color()) == to_rgba("#2f80ed")
        assert to_rgba(artists.lines[2].get_color()) == to_rgba("#e53935")
        assert to_rgba(artists.lines[3].get_color()) == to_rgba("#27ae60")
        assert artists.lines[3].get_linestyle() == "--"
        labels = [text.get_text() for text in artists.legend.get_texts()]
        assert labels == ["Dielectric", "PEC", "PMC", "Wave port", "PML interface"]
        assert len(labels) == len(set(labels))
        np.testing.assert_allclose(axes.get_xlim(), scene.z_span)
        np.testing.assert_allclose(axes.get_ylim(), scene.x_span)

        # A constant-z wave port is vertical in the displayed z-horizontal view.
        np.testing.assert_allclose(artists.lines[2].get_xdata(), (1.0, 1.0))
        np.testing.assert_allclose(artists.lines[2].get_ydata(), (-1.0, 1.0))
        # A constant-x boundary is horizontal.
        np.testing.assert_allclose(artists.lines[0].get_xdata(), (0.0, 4.0))
        np.testing.assert_allclose(artists.lines[0].get_ydata(), (-1.0, -1.0))
    finally:
        plt.close(figure)


def test_s_parameter_plot_uses_frequency_hz() -> None:
    frequencies = np.asarray((1.0e9, 2.0e9))
    results = (
        SimpleNamespace(s_parameters={("left", 0, 0): 0.5}),
        SimpleNamespace(s_parameters={("left", 0, 0): 0.25}),
    )
    figure, axes = plt.subplots()
    try:
        (line,) = plot_s_parameters(axes, frequencies, results)
        np.testing.assert_array_equal(line.get_xdata(), frequencies)
        np.testing.assert_allclose(line.get_ydata(), 20.0 * np.log10((0.5, 0.25)))
        assert axes.get_xlabel() == "Frequency (Hz)"
    finally:
        plt.close(figure)
