"""Regression tests for topology-aware 2D FEM field rendering."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from FEM_Mode_Solver.results import Mode, SampledFields
from FEM_Mode_Solver.constants import C_0
from FEM_Mode_Solver.solver_2d import ModeSolver2D
from FEM_Mode_Solver.visualization import visualize, visualize_with_gui


def _mode_on_square_ring() -> Mode:
    # Eight native triangles cover a square annulus.  A global Delaunay of the
    # element samples would invent triangles across the central square hole.
    points = np.asarray(
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.8, 0.8),
            (1.2, 0.8),
            (1.2, 1.2),
            (0.8, 1.2),
        ]
    )
    cells = np.asarray(
        [
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ],
        dtype=np.int64,
    )
    centroids = points[cells].mean(axis=1)
    sample_points = np.repeat(centroids, 2, axis=0)
    sample_points[0::2, 0] -= 0.025
    sample_points[1::2, 0] += 0.025
    owners = np.repeat(np.arange(cells.shape[0]), 2)
    values = np.repeat(np.linspace(-1.0, 1.0, cells.shape[0]), 2)
    fields = SampledFields(
        sample_points,
        {"Ez": values.astype(np.complex128)},
        dimension=2,
        mesh_points=points,
        mesh_cells=cells,
        material=np.repeat(np.arange(cells.shape[0]) % 2 + 1.0, 2),
        metadata={
            "sampling": "element-quadrature",
            "sample_element_indices": owners,
        },
    )
    return Mode(neff=1.0, beta=1.0, fields=fields, index=1)


def test_quadrature_samples_render_on_native_cells_without_filling_hole() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    mode = _mode_on_square_ring()
    figure, axes = visualize(
        mode,
        component="Ez",
        quantity="real",
        material=False,
        show=False,
    )
    figure.canvas.draw()

    collections = [item for item in axes[0].collections if isinstance(item, PolyCollection)]
    assert collections
    field_collection = collections[0]
    assert len(field_collection.get_paths()) == mode.fields.mesh_cells.shape[0]

    # Every rendered polygon is one supplied mesh triangle; none has a
    # centroid in the intentionally empty central square.
    for path in field_collection.get_paths():
        centroid = path.vertices[:-1].mean(axis=0)
        assert not (0.8 < centroid[0] < 1.2 and 0.8 < centroid[1] < 1.2)

    plt.close(figure)


def test_material_mode_uses_native_interface_edges_without_field_layer() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    mode = _mode_on_square_ring()
    figure, axes = visualize(
        mode,
        component="Ez",
        material=True,
        show=False,
    )
    figure.canvas.draw()

    assert any(isinstance(item, LineCollection) for item in axes[0].collections)
    assert axes[0].get_title().startswith("Mode 1: material")
    plt.close(figure)


def test_gui_field_changes_keep_plot_and_colorbar_geometry() -> None:
    viewer = visualize_with_gui(
        _mode_on_square_ring(),
        component="Ez",
        material=False,
        show=False,
    )
    try:
        viewer.figure.canvas.draw()
        assert viewer._colorbar is not None
        axes_count = len(viewer.figure.axes)
        plot_original = np.asarray(
            viewer.axes.get_position(original=True).bounds,
            dtype=float,
        )
        plot_active = np.asarray(viewer.axes.get_position().bounds, dtype=float)
        colorbar_original = np.asarray(
            viewer._colorbar.ax.get_position(original=True).bounds,
            dtype=float,
        )
        colorbar_active = np.asarray(
            viewer._colorbar.ax.get_position().bounds,
            dtype=float,
        )

        changes = (
            (viewer.quantity_control, 2),  # magnitude
            (viewer.component_control, 1),  # E
            (viewer.component_control, 0),  # Ez
            (viewer.quantity_control, 1),  # imag
            (viewer.quantity_control, 3),  # phase
            (viewer.quantity_control, 0),  # real
        )
        for control, active_index in changes:
            control.set_active(active_index)
            viewer.figure.canvas.draw()
            assert len(viewer.figure.axes) == axes_count
            np.testing.assert_allclose(
                viewer.axes.get_position(original=True).bounds,
                plot_original,
            )
            np.testing.assert_allclose(
                viewer.axes.get_position().bounds,
                plot_active,
            )
            np.testing.assert_allclose(
                viewer._colorbar.ax.get_position(original=True).bounds,
                colorbar_original,
            )
            np.testing.assert_allclose(
                viewer._colorbar.ax.get_position().bounds,
                colorbar_active,
            )
    finally:
        viewer.close()


def test_gui_field_and_material_modes_are_exclusive_and_replace_colorbar() -> None:
    from matplotlib.collections import LineCollection

    viewer = visualize_with_gui(
        _mode_on_square_ring(),
        component="Ez",
        show=False,
    )
    try:
        assert viewer.field
        assert not viewer.material
        viewer.figure.canvas.draw()
        assert not any(
            isinstance(item, LineCollection) for item in viewer.axes.collections
        )
        assert viewer._colorbar is not None
        colorbar = viewer._colorbar
        axes_count = len(viewer.figure.axes)

        viewer.options_control.set_active(2)  # material on, field off
        viewer.figure.canvas.draw()

        assert not viewer.field
        assert viewer.material
        assert viewer._colorbar is colorbar
        assert viewer._colorbar_axes.get_visible()
        assert viewer._colorbar.mappable.cmap.name == "coolwarm"
        assert len(viewer.figure.axes) == axes_count

        viewer.options_control.set_active(0)  # field on, material off
        viewer.figure.canvas.draw()
        assert viewer.field
        assert not viewer.material
        assert viewer._colorbar is colorbar
        assert viewer._colorbar.mappable.cmap.name == "RdBu_r"
        assert not any(
            isinstance(item, LineCollection) for item in viewer.axes.collections
        )
        assert len(viewer.figure.axes) == axes_count
    finally:
        viewer.close()


@pytest.mark.gmsh
def test_2d_solver_records_element_ownership_for_every_sample() -> None:
    solver = ModeSolver2D(
        C_0,
        x_range=1.0,
        y_range=0.5,
        num_modes=1,
        guess=np.sqrt(0.75),
    )
    solver.discretize(resolution=(5, 3))
    fields = solver.solve()[0].fields

    owners = np.asarray(fields.metadata["sample_element_indices"], dtype=np.int64)
    assert owners.shape == fields.component("Ey").shape
    assert owners.min() == 0
    assert owners.max() == fields.mesh_cells.shape[0] - 1
    assert np.all(np.bincount(owners) > 0)
