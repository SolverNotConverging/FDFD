import numpy as np
import pytest

from wavefem.geometry import GeometryModel
from wavefem.materials import Material
from wavefem.mesh import generate_mesh


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
