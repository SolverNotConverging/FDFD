"""Focused tests for transmission-line field-plot annotations."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from FEM_Mode_Solver.results import SampledFields
from FEM_Mode_Solver.transmission_lines.calculator import TransmissionLineCalculator
from FEM_Mode_Solver.transmission_lines.specs import (
    Coaxial,
    CoplanarWaveguide,
    Microstrip,
    Stripline,
)
from FEM_Mode_Solver.transmission_lines.visualization import (
    TransmissionLineFieldViewer,
    _METAL_EDGE_COLOR,
    _METAL_FACE_COLOR,
    _adaptive_magnitude_cutoff,
    _direction_arrow_indices,
    _element_average,
    visualize_transmission_line,
)


def _plot_result(spec: object) -> SimpleNamespace:
    points = np.asarray(
        [
            (-5.0e-3, -3.0e-3),
            (5.0e-3, -3.0e-3),
            (5.0e-3, 4.0e-3),
            (-5.0e-3, 4.0e-3),
        ]
    )
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    coordinates = np.mean(points[cells], axis=1)
    values = {
        "Ex": np.asarray((1.0, 0.4), dtype=np.complex128),
        "Ey": np.asarray((0.2, 0.8), dtype=np.complex128),
        "Hx": np.asarray((-0.2, -0.8), dtype=np.complex128),
        "Hy": np.asarray((1.0, 0.4), dtype=np.complex128),
    }
    fields = SampledFields(
        coordinates,
        values,
        dimension=2,
        mesh_points=points,
        mesh_cells=cells,
    )
    return SimpleNamespace(
        spec=spec,
        label=type(spec).__name__,
        neff=1.5 + 0.0j,
        characteristic_impedance=50.0 + 0.0j,
        wave_impedance=200.0 + 0.0j,
        mode=SimpleNamespace(fields=fields),
    )


@pytest.mark.parametrize(
    "spec, expected_patch_names",
    (
        (Coaxial(), ("Wedge", "Circle", "Wedge")),
        (Microstrip(), ("Rectangle", "Rectangle", "Rectangle")),
        (
            Stripline(),
            ("PathPatch", "Rectangle", "Rectangle", "Rectangle"),
        ),
        (
            CoplanarWaveguide(),
            ("Rectangle", "Rectangle", "Rectangle", "Rectangle"),
        ),
    ),
)
def test_every_line_plot_has_exact_dielectric_and_metal_shapes(
    spec: object,
    expected_patch_names: tuple[str, ...],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = visualize_transmission_line(_plot_result(spec), show=False)
    try:
        for axis in axes:
            assert tuple(type(patch).__name__ for patch in axis.patches) == (
                expected_patch_names
            )
            assert axis.get_legend() is not None
            assert [text.get_text() for text in axis.get_legend().get_texts()] == [
                rf"dielectric $\epsilon_r={float(spec.epsilon_r):g}$",
                "metal",
            ]
            assert axis.get_legend().get_zorder() > max(
                patch.get_zorder() for patch in axis.patches
            )
            for patch in axis.patches[1:]:
                np.testing.assert_allclose(patch.get_facecolor(), _METAL_FACE_COLOR)
                np.testing.assert_allclose(patch.get_edgecolor(), _METAL_EDGE_COLOR)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "spec",
    (Coaxial(), Microstrip(), Stripline(), CoplanarWaveguide()),
)
def test_metal_patch_bounds_match_the_template_geometry(spec: object) -> None:
    import matplotlib.pyplot as plt

    figure, axes = visualize_transmission_line(_plot_result(spec), show=False)
    try:
        metals = axes[0].patches[1:]
        x_minimum, x_maximum = -5.0e-3, 5.0e-3
        full_width = x_maximum - x_minimum

        if isinstance(spec, Coaxial):
            inner, outer = metals
            assert inner.center == pytest.approx((0.0, 0.0))
            assert inner.radius == pytest.approx(spec.inner_radius)
            assert outer.center == pytest.approx((0.0, 0.0))
            assert outer.r == pytest.approx(
                spec.outer_radius + spec.outer_conductor_thickness
            )
            assert outer.width == pytest.approx(spec.outer_conductor_thickness)
        elif isinstance(spec, Microstrip):
            ground, signal = metals
            assert ground.get_bbox().bounds == pytest.approx(
                (
                    x_minimum,
                    -spec.conductor_thickness,
                    full_width,
                    spec.conductor_thickness,
                )
            )
            assert signal.get_bbox().bounds == pytest.approx(
                (
                    -0.5 * spec.trace_width,
                    spec.substrate_height,
                    spec.trace_width,
                    spec.conductor_thickness,
                )
            )
        elif isinstance(spec, Stripline):
            lower, upper, signal = metals
            half_spacing = 0.5 * spec.ground_spacing
            assert lower.get_bbox().bounds == pytest.approx(
                (
                    x_minimum,
                    -half_spacing - spec.conductor_thickness,
                    full_width,
                    spec.conductor_thickness,
                )
            )
            assert upper.get_bbox().bounds == pytest.approx(
                (
                    x_minimum,
                    half_spacing,
                    full_width,
                    spec.conductor_thickness,
                )
            )
            assert signal.get_bbox().bounds == pytest.approx(
                (
                    -0.5 * spec.trace_width,
                    -0.5 * spec.conductor_thickness,
                    spec.trace_width,
                    spec.conductor_thickness,
                )
            )
        else:
            assert isinstance(spec, CoplanarWaveguide)
            signal, left, right = metals
            signal_edge = 0.5 * spec.center_width
            ground_inner_edge = signal_edge + spec.gap
            metal_half_width = ground_inner_edge + spec.ground_width
            assert signal.get_bbox().bounds == pytest.approx(
                (-signal_edge, 0.0, spec.center_width, spec.conductor_thickness)
            )
            assert left.get_bbox().bounds == pytest.approx(
                (
                    -metal_half_width,
                    0.0,
                    spec.ground_width,
                    spec.conductor_thickness,
                )
            )
            assert right.get_bbox().bounds == pytest.approx(
                (
                    ground_inner_edge,
                    0.0,
                    spec.ground_width,
                    spec.conductor_thickness,
                )
            )
    finally:
        plt.close(figure)


def test_stripline_dielectric_shape_preserves_the_signal_hole() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.path import Path

    figure, axes = visualize_transmission_line(_plot_result(Stripline()), show=False)
    try:
        path = axes[0].patches[0].get_path()
        starts = np.flatnonzero(path.codes == Path.MOVETO)
        assert starts.tolist() == [0, 5]

        exterior = path.vertices[:5]
        hole = path.vertices[5:]

        def signed_area(vertices: np.ndarray) -> float:
            return 0.5 * float(
                np.sum(
                    vertices[:-1, 0] * vertices[1:, 1]
                    - vertices[1:, 0] * vertices[:-1, 1]
                )
            )

        assert signed_area(exterior) > 0.0
        assert signed_area(hole) < 0.0
    finally:
        plt.close(figure)


def test_phase_redraw_replaces_instead_of_accumulating_cross_section_shapes() -> None:
    viewer = TransmissionLineFieldViewer(_plot_result(Microstrip()))
    try:
        assert [len(axis.patches) for axis in viewer.axes] == [3, 3]
        viewer.phase_control.set_val(np.pi / 3.0)
        assert [len(axis.patches) for axis in viewer.axes] == [3, 3]
        assert all(axis.get_legend() is not None for axis in viewer.axes)
    finally:
        viewer.close()


def test_adaptive_arrow_floor_ignores_weak_regions_and_resists_one_outlier() -> None:
    magnitude = np.concatenate((np.ones(25), np.full(75, 0.01)))
    areas = np.ones_like(magnitude)
    cutoff = _adaptive_magnitude_cutoff(magnitude, areas)
    assert 0.01 < cutoff <= 1.0

    coordinates = np.column_stack(
        (np.linspace(0.0, 1.0, 100), np.linspace(0.0, 0.5, 100))
    )
    selected = _direction_arrow_indices(
        coordinates,
        magnitude,
        cell_areas=areas,
        maximum_arrows=100,
    )
    assert selected.size > 0
    assert np.all(selected < 25)

    mostly_uniform = np.ones(100)
    mostly_uniform[-1] = 1.0e6
    outlier_cutoff = _adaptive_magnitude_cutoff(mostly_uniform, areas)
    assert outlier_cutoff == pytest.approx(1.0)


@pytest.mark.gmsh
def test_default_microstrip_arrows_suppress_weak_domain_without_losing_direction(
) -> None:
    from matplotlib.quiver import Quiver

    calculator = TransmissionLineCalculator.microstrip(
        frequency=10.0e9,
        metal_conductivity=58.0e6,
    )
    calculator.discretize(
        max_element_size=1.0e-3,
        boundary_refinement=0.4,
    )
    result = calculator.solve()
    figure, axes = result.visualize(show=False)
    try:
        arrow_counts = [
            next(
                collection
                for collection in axis.collections
                if isinstance(collection, Quiver)
            ).U.size
            for axis in axes
        ]
        assert all(80 <= count <= 160 for count in arrow_counts)

        fields = result.fields
        points = np.asarray(fields.mesh_points, dtype=np.float64)
        triangles = np.asarray(fields.mesh_cells, dtype=np.int64)[:, :3]
        triangle_points = points[triangles]
        first_edges = triangle_points[:, 1] - triangle_points[:, 0]
        second_edges = triangle_points[:, 2] - triangle_points[:, 0]
        cell_areas = 0.5 * np.abs(
            first_edges[:, 0] * second_edges[:, 1]
            - first_edges[:, 1] * second_edges[:, 0]
        )

        for family in ("E", "H"):
            first = _element_average(
                fields,
                fields.component(f"{family}x"),
                triangles,
            )
            second = _element_average(
                fields,
                fields.component(f"{family}y"),
                triangles,
            )
            magnitude = np.hypot(np.abs(first), np.abs(second))
            cutoff = _adaptive_magnitude_cutoff(magnitude, cell_areas)
            significant = magnitude >= cutoff
            selected_area_fraction = float(
                np.sum(cell_areas[significant]) / np.sum(cell_areas)
            )
            retained_energy_fraction = float(
                np.sum(cell_areas[significant] * magnitude[significant] ** 2)
                / np.sum(cell_areas * magnitude**2)
            )
            assert cutoff >= 0.05 * float(np.max(magnitude))
            assert 0.20 < selected_area_fraction < 0.56
            assert retained_energy_fraction >= 0.95
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_direction_arrow_limit_is_a_strict_cap() -> None:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
        dtype=np.float64,
    )
    selected = _direction_arrow_indices(
        coordinates,
        np.ones(4),
        cell_areas=np.ones(4),
        maximum_arrows=1,
    )
    assert selected.size == 1


def test_mesh_overlay_can_be_enabled_and_removed_on_redraw() -> None:
    from matplotlib.quiver import Quiver

    viewer = TransmissionLineFieldViewer(_plot_result(Microstrip()))
    try:
        assert all(not axis.lines for axis in viewer.axes)
        viewer.mesh_control.set_active(0)
        assert all(axis.lines for axis in viewer.axes)
        for axis in viewer.axes:
            quiver = next(
                collection
                for collection in axis.collections
                if isinstance(collection, Quiver)
            )
            assert axis.patches[0].get_zorder() < axis.lines[0].get_zorder()
            assert axis.lines[0].get_zorder() < axis.patches[1].get_zorder()
            assert axis.patches[1].get_zorder() < quiver.get_zorder()
            assert axis.lines[0].get_zorder() < quiver.get_zorder()
            assert axis.lines[0].get_zorder() < axis.get_legend().get_zorder()
        viewer.mesh_control.set_active(0)
        assert all(not axis.lines for axis in viewer.axes)
    finally:
        viewer.close()
