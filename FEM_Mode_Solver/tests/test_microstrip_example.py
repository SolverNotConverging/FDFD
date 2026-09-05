"""Focused checks for the copper-microstrip SIBC example."""

from __future__ import annotations

import numpy as np
import pytest

from FEM_Mode_Solver import Rectangle, good_conductor_surface_impedance
from FEM_Mode_Solver.examples.microstrip_sibc import (
    COPPER_THICKNESS,
    DOMAIN_X,
    FREQUENCY,
    MESH_OPTIONS,
    STRIP_WIDTH,
    SUBSTRATE_EPSILON,
    SUBSTRATE_HEIGHT,
    build_solver,
)


def test_microstrip_builder_places_dielectric_and_copper() -> None:
    solver = build_solver()

    assert solver.frequency == FREQUENCY
    assert solver.x_span == DOMAIN_X
    assert len(solver.geometry.regions) == 1
    substrate = solver.geometry.regions[0]
    assert substrate.name == "substrate"
    assert substrate.material.eps_r == (complex(SUBSTRATE_EPSILON),) * 3

    boundaries = {item.name: item for item in solver.geometry.boundaries}
    assert set(boundaries) == {"copper_ground", "copper_strip"}
    expected_impedance = good_conductor_surface_impedance("Cu", FREQUENCY)
    for boundary in boundaries.values():
        assert boundary.kind == "impedance"
        assert boundary.impedance == pytest.approx(expected_impedance)
        assert isinstance(boundary.shape, Rectangle)

    strip = boundaries["copper_strip"].shape
    assert strip.x == pytest.approx((-0.5 * STRIP_WIDTH, 0.5 * STRIP_WIDTH))
    assert strip.y == pytest.approx(
        (SUBSTRATE_HEIGHT, SUBSTRATE_HEIGHT + COPPER_THICKNESS)
    )

    # PEC/SIBC sizing is generated directly from conductor curves.  Keeping a
    # rectangular constant-size overlay here would mask the smooth transition.
    assert solver.geometry.refinements == []


@pytest.mark.gmsh
def test_microstrip_mesh_keeps_material_and_named_sibc_facets() -> None:
    solver = build_solver()
    mesh = solver.discretize(**MESH_OPTIONS)

    assert set(mesh.element_tags) == {1, 2}
    assert mesh.physical_names == {1: "background", 2: "substrate"}
    assert mesh.boundary_facets["copper_ground"].size > 0
    assert mesh.boundary_facets["copper_strip"].size > 0
    assert mesh.boundary_facets["impedance"].size > 0
    assert mesh.boundary_facets.get("pec", np.empty(0, dtype=np.int64)).size == 0
    named_impedance = np.unique(
        np.concatenate(
            (
                mesh.boundary_facets["copper_ground"],
                mesh.boundary_facets["copper_strip"],
            )
        )
    )
    np.testing.assert_array_equal(
        named_impedance,
        np.unique(mesh.boundary_facets["impedance"]),
    )
    assert mesh.info.material_aware
    assert mesh.info.boundary_refinement == pytest.approx(0.5)
    assert mesh.info.refinement_regions == 0

    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    for conductor in solver.geometry.boundaries:
        assert not np.any(conductor.shape.contains(centroids[:, 0], centroids[:, 1]))


@pytest.mark.gmsh
@pytest.mark.slow
def test_microstrip_example_solves_a_passive_quasi_tem_mode() -> None:
    solver = build_solver()
    solver.discretize(**MESH_OPTIONS)
    mode = solver.solve(
        max_refinements=0,
        residual_tolerance=1e-7,
        divergence_tolerance=2e-5,
    )[0]

    assert 1.0 < mode.neff.real < np.sqrt(abs(SUBSTRATE_EPSILON))
    assert mode.neff.imag < 0.0
    assert mode.alpha > 0.0
    assert mode.residual is not None and mode.residual < 1e-7
    assert mode.divergence_residual is not None
    assert mode.divergence_residual < 2e-5
