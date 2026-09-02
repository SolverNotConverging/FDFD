"""Local sizing and remeshing tests for the 2D Gmsh backend."""

from __future__ import annotations

import numpy as np
import pytest

from FEM_Mode_Solver.exceptions import MeshError
from FEM_Mode_Solver.geometry import Circle, GeometryModel2D, Rectangle
from FEM_Mode_Solver.materials import Material
from FEM_Mode_Solver.meshing import FEMMesh2D, discretize_2d
from FEM_Mode_Solver.solver_2d import ModeSolver2D


def _element_maximum_edges(mesh: FEMMesh2D) -> np.ndarray:
    points = mesh.nodes
    triangles = mesh.elements
    return np.stack(
        (
            np.linalg.norm(points[triangles[:, 0]] - points[triangles[:, 1]], axis=1),
            np.linalg.norm(points[triangles[:, 1]] - points[triangles[:, 2]], axis=1),
            np.linalg.norm(points[triangles[:, 2]] - points[triangles[:, 0]], axis=1),
        ),
        axis=1,
    ).max(axis=1)


@pytest.mark.parametrize(
    "resolution",
    (
        (2.9, 3.0),
        (3, 3, 99),
        (True, 3),
        (3, np.inf),
        "3,3",
    ),
)
def test_resolution_requires_exactly_two_finite_integers(resolution: object) -> None:
    geometry = GeometryModel2D(1.0, 1.0, Material())
    with pytest.raises(MeshError, match="resolution"):
        discretize_2d(geometry, resolution=resolution)  # type: ignore[arg-type]


@pytest.mark.gmsh
def test_high_dk_material_gets_smaller_elements() -> None:
    geometry = GeometryModel2D(1.0, 1.0, Material())
    geometry.add_region(
        Rectangle((0.5, 1.0), (0.0, 1.0)),
        Material(16.0),
        name="high_dk",
    )

    mesh = discretize_2d(
        geometry,
        max_element_size=0.15,
        material_aware=True,
        boundary_refinement=None,
    )
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    edges = _element_maximum_edges(mesh)
    low_dk = edges[centroids[:, 0] < 0.35]
    high_dk = edges[centroids[:, 0] > 0.65]

    assert mesh.info.material_aware
    assert np.median(high_dk) < 0.4 * np.median(low_dk)


@pytest.mark.gmsh
def test_material_grading_preserves_index_contrast_below_unity() -> None:
    geometry = GeometryModel2D(1.0, 1.0, Material(0.25))
    geometry.add_region(
        Rectangle((0.5, 1.0), (0.0, 1.0)),
        Material(1.0),
        name="higher_index",
    )
    mesh = discretize_2d(
        geometry,
        max_element_size=0.15,
        material_aware=True,
        boundary_refinement=None,
    )
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    edges = _element_maximum_edges(mesh)

    low_index = edges[centroids[:, 0] < 0.35]
    higher_index = edges[centroids[:, 0] > 0.65]
    assert np.median(higher_index) < 0.65 * np.median(low_index)


@pytest.mark.gmsh
def test_explicit_refinement_shape_changes_mesh_not_material() -> None:
    geometry = GeometryModel2D(1.0, 1.0, Material())
    refinement = geometry.add_mesh_refinement(
        Circle((0.5, 0.5), 0.2),
        0.025,
        transition_width=0.15,
        name="field_hotspot",
    )

    mesh = discretize_2d(
        geometry,
        max_element_size=0.14,
        material_aware=False,
        boundary_refinement=None,
    )
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    radius = np.linalg.norm(centroids - np.asarray((0.5, 0.5)), axis=1)
    edges = _element_maximum_edges(mesh)

    assert refinement.name == "field_hotspot"
    assert set(mesh.element_tags) == {1}
    assert mesh.info.refinement_regions == 1
    assert np.median(edges[radius < 0.15]) < 0.35 * np.median(edges[radius > 0.4])


@pytest.mark.gmsh
def test_material_interface_refinement_is_optional() -> None:
    geometry = GeometryModel2D(1.0, 1.0, Material())
    geometry.add_region(
        Rectangle((0.45, 0.55), (0.0, 1.0)), Material(4.0), name="strip"
    )
    refined = discretize_2d(
        geometry,
        max_element_size=0.13,
        material_aware=False,
        interface_refinement=0.35,
        interface_refinement_width=0.08,
        boundary_refinement=None,
    )
    ordinary = discretize_2d(
        geometry,
        max_element_size=0.13,
        material_aware=False,
        interface_refinement=None,
        boundary_refinement=None,
    )

    def near_interface_size(mesh: FEMMesh2D) -> float:
        centers = mesh.nodes[mesh.elements].mean(axis=1)
        distance = np.minimum(abs(centers[:, 0] - 0.45), abs(centers[:, 0] - 0.55))
        return float(np.median(_element_maximum_edges(mesh)[distance < 0.025]))

    assert refined.info.interface_refinement == pytest.approx(0.35)
    assert ordinary.info.interface_refinement is None
    assert near_interface_size(refined) < 0.65 * near_interface_size(ordinary)


@pytest.mark.gmsh
def test_boundary_refinement_densifies_walls_and_can_be_disabled() -> None:
    geometry = GeometryModel2D(1.0, 1.0, Material())
    refined = discretize_2d(
        geometry,
        max_element_size=0.14,
        material_aware=False,
        boundary_refinement=0.4,
    )
    ordinary = discretize_2d(
        geometry,
        max_element_size=0.14,
        material_aware=False,
        boundary_refinement=None,
    )

    def wall_and_bulk_sizes(mesh: FEMMesh2D) -> tuple[float, float]:
        centers = mesh.nodes[mesh.elements].mean(axis=1)
        wall_distance = np.min(
            np.stack(
                (
                    centers[:, 0],
                    1.0 - centers[:, 0],
                    centers[:, 1],
                    1.0 - centers[:, 1],
                ),
                axis=1,
            ),
            axis=1,
        )
        edges = _element_maximum_edges(mesh)
        return (
            float(np.median(edges[wall_distance < 0.06])),
            float(np.median(edges[wall_distance > 0.30])),
        )

    refined_wall, refined_bulk = wall_and_bulk_sizes(refined)
    ordinary_wall, _ = wall_and_bulk_sizes(ordinary)
    assert refined.info.boundary_refinement == pytest.approx(0.4)
    assert ordinary.info.boundary_refinement is None
    assert refined_wall < 0.6 * refined_bulk
    assert refined_wall < 0.6 * ordinary_wall


@pytest.mark.gmsh
def test_internal_pec_and_sibc_use_smooth_distance_grading() -> None:
    geometry = GeometryModel2D(1.0, 1.0, Material())
    pec = Rectangle((0.22, 0.32), (0.45, 0.55))
    sibc = Rectangle((0.68, 0.78), (0.45, 0.55))
    geometry.add_boundary(pec, "pec", name="pec_insert")
    geometry.add_boundary(
        sibc, "impedance", impedance=50.0, name="sibc_insert"
    )

    mesh = discretize_2d(
        geometry,
        max_element_size=0.14,
        material_aware=False,
        boundary_refinement=0.3,
        boundary_refinement_width=0.16,
    )
    centers = mesh.nodes[mesh.elements].mean(axis=1)
    edges = _element_maximum_edges(mesh)

    def distance_to_rectangle(rectangle: Rectangle) -> np.ndarray:
        dx = np.maximum.reduce(
            (
                rectangle.x[0] - centers[:, 0],
                centers[:, 0] - rectangle.x[1],
                np.zeros(len(centers)),
            )
        )
        dy = np.maximum.reduce(
            (
                rectangle.y[0] - centers[:, 1],
                centers[:, 1] - rectangle.y[1],
                np.zeros(len(centers)),
            )
        )
        return np.hypot(dx, dy)

    pec_distance = distance_to_rectangle(pec)
    sibc_distance = distance_to_rectangle(sibc)
    outer_distance = np.minimum.reduce(
        (
            centers[:, 0],
            1.0 - centers[:, 0],
            centers[:, 1],
            1.0 - centers[:, 1],
        )
    )
    bulk = edges[
        (pec_distance > 0.20)
        & (sibc_distance > 0.20)
        & (outer_distance > 0.20)
    ]
    pec_near = edges[pec_distance < 0.045]
    sibc_near = edges[sibc_distance < 0.045]

    assert bulk.size > 0 and pec_near.size > 0 and sibc_near.size > 0
    assert np.median(pec_near) < 0.55 * np.median(bulk)
    assert np.median(sibc_near) < 0.55 * np.median(bulk)


@pytest.mark.gmsh
def test_meshing_restores_options_in_an_existing_gmsh_session() -> None:
    import gmsh

    owned = not bool(gmsh.isInitialized())
    if owned:
        gmsh.initialize()
    names = (
        "General.Terminal",
        "Mesh.MeshSizeMax",
        "Mesh.MeshSizeMin",
        "Mesh.MeshSizeExtendFromBoundary",
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.Algorithm",
    )
    original = {name: float(gmsh.option.getNumber(name)) for name in names}
    requested = {
        "General.Terminal": 1.0,
        "Mesh.MeshSizeMax": 0.321,
        "Mesh.MeshSizeMin": 0.0123,
        "Mesh.MeshSizeExtendFromBoundary": 1.0,
        "Mesh.MeshSizeFromPoints": 1.0,
        "Mesh.MeshSizeFromCurvature": 1.0,
        "Mesh.Algorithm": 6.0,
    }
    try:
        for name, value in requested.items():
            gmsh.option.setNumber(name, value)
        geometry = GeometryModel2D(1.0, 1.0, Material())
        discretize_2d(geometry, max_element_size=0.15)
        for name, value in requested.items():
            assert gmsh.option.getNumber(name) == pytest.approx(value)
    finally:
        for name, value in original.items():
            gmsh.option.setNumber(name, value)
        if owned:
            gmsh.finalize()


@pytest.mark.gmsh
def test_internal_wall_kinds_keep_provenance_with_boundary_sizing() -> None:
    geometry = GeometryModel2D(1.0, 1.0, Material())
    geometry.add_boundary(
        Rectangle((0.15, 0.25), (0.40, 0.60)), "pec", name="pec_insert"
    )
    geometry.add_boundary(
        Rectangle((0.45, 0.55), (0.40, 0.60)), "pmc", name="pmc_insert"
    )
    geometry.add_boundary(
        Rectangle((0.75, 0.85), (0.40, 0.60)),
        "impedance",
        impedance=50.0,
        name="sibc_insert",
    )

    mesh = discretize_2d(
        geometry,
        max_element_size=0.12,
        boundary_refinement=0.5,
    )

    for name in ("pec_insert", "pmc_insert", "sibc_insert", "pec", "pmc", "impedance"):
        assert mesh.boundary_facets[name].size > 0


@pytest.mark.gmsh
def test_solver_refine_remeshes_and_reassembles() -> None:
    solver = ModeSolver2D(1.0e6, 1.0, 1.0, num_modes=1)
    solver.add_mesh_refinement(
        Rectangle((0.35, 0.65), (0.35, 0.65)),
        max_element_size=0.06,
        name="center",
    )
    coarse = solver.discretize(
        max_element_size=0.18,
        wavelength_elements=4,
        material_aware=False,
        boundary_refinement=None,
    )
    coarse_system = solver.system
    fine = solver.refine(2.0)

    assert fine.info.requested_maximum_edge == pytest.approx(
        0.5 * coarse.info.requested_maximum_edge
    )
    assert fine.info.elements > coarse.info.elements
    assert solver.system is not coarse_system
    assert solver.solution is None
