from __future__ import annotations

import numpy as np
import pytest

from fem_electrostatics import ElectrostaticSolver, Rectangle


pytestmark = pytest.mark.gmsh


def _element_edge_mean(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    points = nodes[elements]
    return (
        np.linalg.norm(points[:, 1] - points[:, 0], axis=1)
        + np.linalg.norm(points[:, 2] - points[:, 1], axis=1)
        + np.linalg.norm(points[:, 0] - points[:, 2], axis=1)
    ) / 3.0


def test_high_dk_region_receives_smaller_gmsh_elements() -> None:
    solver = ElectrostaticSolver(dim=2, x_range=((0.0, 2.0), (0.0, 1.0))[0], y_range=((0.0, 2.0), (0.0, 1.0))[1])
    solver.add_object(region=Rectangle((1.0, 2.0), (0.0, 1.0)), name='high_dk', epsilon=((16.0, 0.0), (0.0, 1.0)))
    mesh = solver.mesh(max_element_size=0.16, interface_refinement=None, boundary_refinement=None)
    centers = mesh.nodes[mesh.elements].mean(axis=1)
    edge = _element_edge_mean(mesh.nodes, mesh.elements)

    low = edge[(centers[:, 0] > 0.25) & (centers[:, 0] < 0.75)]
    high = edge[(centers[:, 0] > 1.25) & (centers[:, 0] < 1.75)]
    assert np.median(high) < 0.55 * np.median(low)
    assert mesh.info.material_element_sizes["high_dk"] == pytest.approx(0.04)


def test_dirichlet_boundary_refinement_reduces_near_boundary_size() -> None:
    solver = ElectrostaticSolver(dim=2, x_range=((0.0, 1.0), (0.0, 1.0))[0], y_range=((0.0, 1.0), (0.0, 1.0))[1])
    solver.set_potential(region=Rectangle((0.45, 0.55), (0.2, 0.8)), potential=1.0, name='electrode')
    mesh = solver.mesh(max_element_size=0.14, material_aware=False, interface_refinement=None, boundary_refinement=0.35, boundary_refinement_width=0.12)
    centers = mesh.nodes[mesh.elements].mean(axis=1)
    edge = _element_edge_mean(mesh.nodes, mesh.elements)
    near = edge[(np.abs(centers[:, 0] - 0.45) < 0.06) & (centers[:, 1] > 0.25) & (centers[:, 1] < 0.75)]
    bulk = edge[(centers[:, 0] > 0.7) & (centers[:, 0] < 0.85) & (centers[:, 1] > 0.35) & (centers[:, 1] < 0.65)]

    assert len(near) and len(bulk)
    assert np.median(near) < 0.75 * np.median(bulk)
