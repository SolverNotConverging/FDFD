import numpy as np
import pytest

from wavefem.exceptions import ConfigurationError, MeshError
from wavefem.geometry import GeometryModel
from wavefem.materials import Material
from wavefem.mesh import generate_mesh
from wavefem.scattering import Scattering2D


def _triangle_maximum_edges(mesh: object) -> np.ndarray:
    triangles = mesh.mesh.p.T[mesh.mesh.t.T]
    return np.max(
        np.stack(
            (
                np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
                np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            ),
            axis=1,
        ),
        axis=1,
    )


@pytest.mark.gmsh
def test_gmsh_mesh_has_conforming_material_regions() -> None:
    geometry = GeometryModel((-1.0, 1.0), (-1.0, 1.0), Material(eps_r=1.0))
    geometry.add_rectangle(
        x=(-0.25, 0.25), z="all", material=4.0, background=True, name="core"
    )
    geometry.add_circle(center=(0.0, 0.0), radius=0.15, material=5.0, name="defect")
    mesh = generate_mesh(
        geometry,
        max_element_size=0.2,
        x_partitions=(-0.8, 0.8),
        z_partitions=(-0.8, 0.8),
    )
    assert mesh.info.nodes > 10
    assert mesh.info.elements > 10
    assert mesh.elements_in("core").size > 0
    assert mesh.elements_in("defect").size > 0
    assert set(mesh.element_tags) == {1, 2, 3}


@pytest.mark.gmsh
def test_partition_lines_conform_with_overlapping_full_span_materials() -> None:
    geometry = GeometryModel((-1.0, 1.0), (-2.0, 2.0), Material(eps_r=1.0))
    geometry.add_rectangle(
        x=(-0.35, 0.35), z="all", material=4.0, background=True, name="core"
    )
    geometry.add_rectangle(
        x=(-1.0, 1.0), z=(-0.45, 0.45), material=5.0, name="full_width_step"
    )
    # Include coordinates coincident with material interfaces as well as lines
    # that cut across them; both cases must remain conforming and duplicate-free.
    x_partitions = (-0.8, -0.35, 0.0, 0.35, 0.8)
    z_partitions = (-1.5, -0.9, -0.45, 0.45, 0.9, 1.5)

    mesh = generate_mesh(
        geometry,
        max_element_size=0.25,
        x_partitions=x_partitions,
        z_partitions=z_partitions,
    )

    points = mesh.mesh.p.T
    triangles = points[mesh.mesh.t.T]
    tolerance = 64.0 * np.finfo(float).eps
    for axis, partitions in ((0, x_partitions), (1, z_partitions)):
        minimum = triangles[:, :, axis].min(axis=1)
        maximum = triangles[:, :, axis].max(axis=1)
        for coordinate in partitions:
            crosses = (minimum < coordinate - tolerance) & (
                maximum > coordinate + tolerance
            )
            assert not np.any(crosses)

    assert mesh.elements_in("core").size > 0
    assert mesh.elements_in("full_width_step").size > 0
    assert set(mesh.element_tags) == {1, 2, 3}


@pytest.mark.gmsh
def test_multiple_disjoint_layers_and_monitor_partitions_are_conforming() -> None:
    geometry = GeometryModel((0.0, 0.8e-6), (-2.8e-6, 2.8e-6), Material())
    geometry.add_rectangle(
        x=(0.0, 0.8e-6),
        z=(-0.50e-6, -0.16e-6),
        material=1.14,
        name="first",
    )
    geometry.add_rectangle(
        x=(0.0, 0.8e-6),
        z=(0.04e-6, 0.34e-6),
        material=1.035,
        name="second",
    )
    partitions = (-2.1e-6, -1.25e-6, 1.25e-6, 2.1e-6)

    mesh = generate_mesh(
        geometry,
        max_element_size=0.20e-6,
        z_partitions=partitions,
    )

    triangles = mesh.mesh.p.T[mesh.mesh.t.T]
    tolerance = 64.0 * np.finfo(float).eps
    z_minimum = triangles[:, :, 1].min(axis=1)
    z_maximum = triangles[:, :, 1].max(axis=1)
    for coordinate in partitions:
        crosses = (z_minimum < coordinate - tolerance) & (
            z_maximum > coordinate + tolerance
        )
        assert not np.any(crosses)
    assert mesh.elements_in("first").size > 0
    assert mesh.elements_in("second").size > 0


@pytest.mark.gmsh
def test_background_pec_sheet_and_slot_are_conforming_mesh_facets() -> None:
    geometry = GeometryModel((-1.0, 1.0), (-2.0, 2.0), Material())
    ground = geometry.add_pec(
        x=0.15,
        background=True,
        name="ground",
    )
    slot = geometry.add_slot(
        pec=ground,
        z=(-0.45, 0.35),
        name="aperture",
    )

    mesh = generate_mesh(geometry, max_element_size=0.25)

    assert mesh.background_pec_facets.size > mesh.actual_pec_facets.size > 0
    assert mesh.released_pec_facets.size > 0
    np.testing.assert_array_equal(
        mesh.facets_in_slot(slot.name),
        mesh.released_pec_facets,
    )
    np.testing.assert_array_equal(
        mesh.pec_facets("background"),
        np.union1d(mesh.actual_pec_facets, mesh.released_pec_facets),
    )
    assert not np.intersect1d(
        mesh.actual_pec_facets, mesh.released_pec_facets
    ).size

    facet_points = mesh.mesh.p[:, mesh.mesh.facets]
    released_points = facet_points[:, :, mesh.released_pec_facets]
    np.testing.assert_allclose(released_points[0], ground.x, atol=1e-13)
    assert released_points[1].min() >= slot.z[0] - 1e-13
    assert released_points[1].max() <= slot.z[1] + 1e-13
    assert np.all(mesh.mesh.f2t[1, mesh.released_pec_facets] >= 0)

    triangles = mesh.mesh.p.T[mesh.mesh.t.T]
    tolerance = 1e-13
    for axis, coordinate in (
        (0, ground.x),
        (1, slot.z[0]),
        (1, slot.z[1]),
    ):
        minimum = triangles[:, :, axis].min(axis=1)
        maximum = triangles[:, :, axis].max(axis=1)
        assert not np.any(
            (minimum < coordinate - tolerance)
            & (maximum > coordinate + tolerance)
        )


@pytest.mark.gmsh
def test_multiple_named_pec_slots_have_disjoint_released_facet_mappings() -> None:
    geometry = GeometryModel((-1.0, 1.0), (-3.0, 3.0), Material())
    ground = geometry.add_pec(x=0.0, background=True, name="ground")
    first = geometry.add_slot(pec=ground, z=(-1.2, -0.7), name="first")
    second = geometry.add_slot(pec=ground, z=(0.4, 1.1), name="second")

    mesh = generate_mesh(geometry, max_element_size=0.35)

    first_facets = mesh.facets_in_slot(first.name)
    second_facets = mesh.facets_in_slot(second.name)
    assert first_facets.size > 0
    assert second_facets.size > 0
    assert not np.intersect1d(first_facets, second_facets).size
    np.testing.assert_array_equal(
        mesh.released_pec_facets,
        np.union1d(first_facets, second_facets),
    )


@pytest.mark.gmsh
def test_dielectric_local_wavelength_field_refines_high_index_region() -> None:
    geometry = GeometryModel((-2.0, 2.0), (-2.0, 2.0), Material(eps_r=1.0))
    geometry.add_rectangle(
        x=(0.0, 2.0),
        z="all",
        material=4.0,
        background=True,
        name="dielectric",
    )

    mesh = generate_mesh(geometry, max_element_size=0.4)
    sizes = _triangle_maximum_edges(mesh)
    centroids = mesh.mesh.p.T[mesh.mesh.t.T].mean(axis=1)
    exterior = (mesh.element_tags == 1) & (centroids[:, 0] < -0.5)
    dielectric = (mesh.element_tags == 2) & (centroids[:, 0] > 0.5)

    assert exterior.sum() > 20
    assert dielectric.sum() > 20
    assert np.median(sizes[dielectric]) < 0.7 * np.median(sizes[exterior])


@pytest.mark.gmsh
def test_pec_distance_field_refines_elements_near_actual_sheet() -> None:
    geometry = GeometryModel((-2.0, 2.0), (-2.0, 2.0), Material())
    geometry.add_pec(x=0.0, background=True, name="ground")

    mesh = generate_mesh(
        geometry,
        max_element_size=0.4,
        pec_refinement_factor=0.35,
        pec_refinement_distance=0.7,
    )
    sizes = _triangle_maximum_edges(mesh)
    centroids = mesh.mesh.p.T[mesh.mesh.t.T].mean(axis=1)
    near = np.abs(centroids[:, 0]) < 0.2
    far = np.abs(centroids[:, 0]) > 1.2
    pec_points = mesh.mesh.p[:, mesh.mesh.facets[:, mesh.actual_pec_facets]]
    pec_edge_lengths = np.linalg.norm(pec_points[:, 1] - pec_points[:, 0], axis=0)

    assert near.sum() > 20
    assert far.sum() > 20
    assert np.median(sizes[near]) < 0.65 * np.median(sizes[far])
    assert np.max(pec_edge_lengths) < 0.65 * mesh.info.requested_maximum_edge


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("refine_dielectrics", "yes", "refine_dielectrics"),
        ("refine_pec", 1, "refine_pec"),
        ("dielectric_refinement_factor", 0.0, "dielectric_refinement_factor"),
        ("dielectric_refinement_factor", 1.1, "dielectric_refinement_factor"),
        ("pec_refinement_factor", 0.0, "pec_refinement_factor"),
        ("pec_refinement_factor", 1.1, "pec_refinement_factor"),
        ("pec_refinement_distance", -1.0, "pec_refinement_distance"),
    ),
)
def test_mesh_refinement_controls_are_validated(
    keyword: str,
    value: object,
    message: str,
) -> None:
    geometry = GeometryModel((-1.0, 1.0), (-1.0, 1.0), Material())
    with pytest.raises(MeshError, match=message):
        generate_mesh(geometry, max_element_size=0.25, **{keyword: value})


@pytest.mark.gmsh
def test_scattering_mesh_preserves_global_cap_and_adds_dielectric_target() -> None:
    simulation = Scattering2D(
        frequency=1.0e9,
        x_span=(-0.3, 0.3),
        z_span=(-0.5, 0.5),
    )
    simulation.add_rectangle(
        x=(0.0, 0.3),
        z="all",
        eps=4.0,
        background=True,
        name="dielectric",
    )

    mesh = simulation.mesh(wavelength_elements=8)

    assert mesh.info.requested_maximum_edge == pytest.approx(
        simulation.frequency.wavelength / 16.0
    )
    assert simulation._mesh_size == pytest.approx(
        simulation.frequency.wavelength / 64.0
    )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("dielectric_refinement_factor", False, "dielectric_refinement_factor"),
        ("dielectric_refinement_factor", np.nan, "dielectric_refinement_factor"),
        ("pec_refinement_factor", False, "pec_refinement_factor"),
        ("pec_refinement_factor", np.nan, "pec_refinement_factor"),
        ("pec_refinement_distance", True, "pec_refinement_distance"),
        ("pec_refinement_distance", np.inf, "pec_refinement_distance"),
    ),
)
def test_scattering_mesh_refinement_controls_are_validated(
    keyword: str,
    value: object,
    message: str,
) -> None:
    simulation = Scattering2D(
        frequency=1.0e9,
        x_span=(-0.3, 0.3),
        z_span=(-0.5, 0.5),
    )
    with pytest.raises(ConfigurationError, match=message):
        simulation.mesh(**{keyword: value})
