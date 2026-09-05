from __future__ import annotations

import numpy as np
import pytest

from fem_waveguide_modes.exceptions import GeometryError, MeshError
from fem_waveguide_modes.geometry import (
    Circle,
    GeometryModel1D,
    GeometryModel2D,
    PMLSpec,
    Rectangle,
)
from fem_waveguide_modes.materials import Material
from fem_waveguide_modes.meshing import discretize_2d
from fem_waveguide_modes.solver_1d import ModeSolver1D


def test_annular_boundary_removes_ring_and_names_both_circular_walls() -> None:
    model = GeometryModel2D(1.0, 1.0, Material())
    ring = model.add_boundary(
        Circle((0.5, 0.5), radius=0.25, inner_radius=0.14),
        "pec",
        name="metal_ring",
    )

    discrete = discretize_2d(model, max_element_size=0.075)
    centroids = discrete.nodes[discrete.elements].mean(axis=1)
    centroid_radii = np.linalg.norm(centroids - np.asarray((0.5, 0.5)), axis=1)

    assert not np.any(ring.shape.contains(centroids[:, 0], centroids[:, 1]))
    assert np.any(centroid_radii < 0.12), "the dielectric island inside the ring was removed"

    ring_facets = discrete.mesh.facets[:, discrete.boundary_facets["metal_ring"]].T
    endpoint_radii = np.linalg.norm(
        discrete.nodes[ring_facets] - np.asarray((0.5, 0.5)), axis=2
    )
    assert np.any(np.all(np.isclose(endpoint_radii, 0.14, atol=1e-10), axis=1))
    assert np.any(np.all(np.isclose(endpoint_radii, 0.25, atol=1e-10), axis=1))


def test_annular_material_tags_only_the_ring() -> None:
    model = GeometryModel2D(1.0, 1.0, Material())
    ring = model.add_region(
        Circle((0.5, 0.5), radius=0.27, inner_radius=0.13),
        Material(3.0),
        name="dielectric_ring",
    )

    discrete = discretize_2d(model, max_element_size=0.075)
    centroids = discrete.nodes[discrete.elements].mean(axis=1)
    inside = ring.shape.contains(centroids[:, 0], centroids[:, 1])

    np.testing.assert_array_equal(discrete.element_tags == 2, inside)
    assert np.any(discrete.element_tags == 1)
    assert np.any(discrete.element_tags == 2)


def test_curved_fragment_tags_preserve_later_region_precedence() -> None:
    model = GeometryModel2D(1.0, 1.0, Material())
    ring = model.add_region(
        Circle((0.5, 0.5), radius=0.30, inner_radius=0.15),
        Material(2.0),
        name="ring",
    )
    disk = model.add_region(
        Circle((0.5, 0.5), radius=0.20),
        Material(4.0),
        name="later_disk",
    )

    discrete = discretize_2d(model, max_element_size=0.08)
    centroids = discrete.nodes[discrete.elements].mean(axis=1)
    expected = np.ones(discrete.info.elements, dtype=np.int32)
    expected[ring.shape.contains(centroids[:, 0], centroids[:, 1])] = 2
    expected[disk.shape.contains(centroids[:, 0], centroids[:, 1])] = 3

    np.testing.assert_array_equal(discrete.element_tags, expected)
    assert set(discrete.element_tags) == {1, 2, 3}


@pytest.mark.parametrize(
    ("first", "second"),
    (("x-", "x-"), ("x", "x+"), ("y+", "all"), ("all", "y")),
)
def test_2d_pml_rejects_repeated_side_coverage(first: str, second: str) -> None:
    model = GeometryModel2D(1.0, 1.0, Material())
    model.add_pml(PMLSpec(0.1, direction=first))

    with pytest.raises(GeometryError, match="already covers"):
        model.add_pml(PMLSpec(0.1, direction=second))


def test_2d_pml_accepts_disjoint_sides_and_checks_combined_width() -> None:
    model = GeometryModel2D((0.0, 1.0), (0.0, 2.0), Material())
    for direction in ("x-", "x+", "y-", "y+"):
        model.add_pml(PMLSpec(0.1, direction=direction))

    assert [pml.direction for pml in model.pmls] == ["x-", "x+", "y-", "y+"]

    too_wide = GeometryModel2D(1.0, 1.0, Material())
    too_wide.add_pml(PMLSpec(0.55, direction="x-"))
    with pytest.raises(GeometryError, match="leave no interior"):
        too_wide.add_pml(PMLSpec(0.45, direction="x+"))


def test_1d_pml_rejects_repeated_sides_but_accepts_opposing_sides() -> None:
    model = GeometryModel1D(1.0, Material())
    model.add_pml(PMLSpec(0.2, direction="x-"))
    model.add_pml(PMLSpec(0.2, direction="x+"))

    with pytest.raises(GeometryError, match="already covers"):
        model.add_pml(PMLSpec(0.1, direction="x+"))

    too_wide = GeometryModel1D(1.0, Material())
    too_wide.add_pml(PMLSpec(0.6, direction="x-"))
    with pytest.raises(GeometryError, match="no non-PML"):
        too_wide.add_pml(PMLSpec(0.4, direction="x+"))


def test_1d_solver_facade_accepts_distinct_pml_sides() -> None:
    solver = ModeSolver1D(frequency=10000000000.0, x_range=1.0)
    solver.add_pml(thickness=0.1, direction='left')
    solver.add_pml(thickness=0.2, direction='right')

    assert [pml.direction for pml in solver.geometry.pmls] == ["x-", "x+"]
    with pytest.raises(GeometryError, match="already covers"):
        solver.add_pml(thickness=0.1, direction='x-')


def test_boundary_geometry_rejects_clipping_and_reserved_mesh_tag_names() -> None:
    model = GeometryModel2D(1.0, 1.0, Material())

    with pytest.raises(GeometryError, match="outside"):
        model.add_boundary(
            Rectangle((-0.1, 0.2), (0.2, 0.4)),
            "impedance",
            impedance=50.0,
        )
    with pytest.raises(GeometryError, match="reserved"):
        model.add_boundary(
            Rectangle((0.1, 0.2), (0.2, 0.4)),
            "impedance",
            impedance=50.0,
            name="pec",
        )
    assert model.boundaries == []


@pytest.mark.gmsh
def test_overlapping_conductors_reject_different_boundary_conditions() -> None:
    model = GeometryModel2D(1.0, 1.0, Material())
    model.add_boundary(
        Rectangle((0.2, 0.55), (0.3, 0.7)),
        "impedance",
        impedance=50.0,
        name="first_metal",
    )
    model.add_boundary(
        Rectangle((0.45, 0.8), (0.3, 0.7)),
        "impedance",
        impedance=75.0,
        name="second_metal",
    )

    with pytest.raises(MeshError, match="Overlapping conductor"):
        discretize_2d(model, max_element_size=0.12)

    same_condition = GeometryModel2D(1.0, 1.0, Material())
    same_condition.add_boundary(
        Rectangle((0.2, 0.55), (0.3, 0.7)),
        "impedance",
        impedance=50.0,
        name="first_same_metal",
    )
    same_condition.add_boundary(
        Rectangle((0.45, 0.8), (0.3, 0.7)),
        "impedance",
        impedance=50.0,
        name="second_same_metal",
    )
    with pytest.raises(MeshError, match="Overlapping conductor"):
        discretize_2d(same_condition, max_element_size=0.12)
