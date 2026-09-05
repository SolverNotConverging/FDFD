from __future__ import annotations

import numpy as np
import pytest

from fem_waveguide_modes.geometry import (
    Circle,
    GeometryModel1D,
    GeometryModel2D,
    Interval,
    PMLSpec,
    Polygon,
    Rectangle,
)
from fem_waveguide_modes.materials import Material
from fem_waveguide_modes.meshing import discretize_1d, discretize_2d


def test_material_normalizes_scalar_and_diagonal_values() -> None:
    isotropic = Material(2.25, 1.0)
    diagonal = Material((2.0, 3.0, 4.0), (1.0, 1.1, 1.2))

    assert isotropic.eps_r == (2.25 + 0j,) * 3
    assert isotropic.isotropic
    assert diagonal.eps_r == (2.0 + 0j, 3.0 + 0j, 4.0 + 0j)
    assert not diagonal.isotropic


def test_geometry_is_continuous_and_later_regions_win() -> None:
    model = GeometryModel2D(1.0, 1.0, Material())
    model.add_region(Rectangle((0.1, 0.9), (0.1, 0.9)), Material(2.0), name="slab")
    model.add_region(Circle((0.5, 0.5), 0.2), Material(4.0), name="core")

    eps, _ = model.material_at(np.array([0.05, 0.25, 0.5]), np.array([0.05, 0.25, 0.5]))
    np.testing.assert_allclose(eps[0], [1.0, 2.0, 4.0])
    assert model.revision == 2


def test_polygon_contains_vectorized_points() -> None:
    triangle = Polygon(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    inside = triangle.contains(np.array([0.1, 0.8]), np.array([0.1, 0.8]))
    np.testing.assert_array_equal(inside, [True, False])


def test_1d_discretization_inserts_material_and_pml_interfaces() -> None:
    model = GeometryModel1D((0.0, 1.0), Material())
    model.add_region(Interval((0.3, 0.6)), Material(2.0), name="layer")
    model.add_pml(PMLSpec(0.1, direction="x"))

    discrete = discretize_1d(model, resolution=10)

    for coordinate in (0.1, 0.3, 0.6, 0.9):
        assert np.min(np.abs(discrete.nodes - coordinate)) < 1e-14
    assert discrete.geometry_revision == model.revision


@pytest.mark.gmsh
def test_2d_mesh_conforms_to_material_and_excludes_pec_object() -> None:
    model = GeometryModel2D(1.0, 1.0, Material())
    model.add_region(Circle((0.25, 0.5), 0.12), Material(2.5), name="core")
    metal = model.add_boundary(
        Rectangle((0.60, 0.78), (0.40, 0.60)), "pec", name="metal"
    )

    discrete = discretize_2d(model, max_element_size=0.12)
    centroids = discrete.nodes[discrete.elements].mean(axis=1)

    assert set(discrete.element_tags) == {1, 2}
    assert not np.any(metal.shape.contains(centroids[:, 0], centroids[:, 1]))
    assert discrete.boundary_facets["metal"].size > 0
    assert discrete.boundary_facets["outer_pec"].size > 0
