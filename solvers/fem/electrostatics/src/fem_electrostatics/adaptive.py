"""Flux-jump indicators and conforming local refinement for P1 electrostatics."""

from dataclasses import replace

import numpy as np

from .meshing import FEMMesh, _mesh_extrema


def flux_indicators(mesh: FEMMesh, displacement: np.ndarray, charge: np.ndarray,
                    fixed: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate discretization error from div(D)=rho and normal-flux jumps.

    The relative indicator is normalized by the elementwise L2 norm of D.
    It is a refinement heuristic, not a certified bound on the potential.
    Prescribed-potential faces carry conductor charge and are excluded.
    """
    points = mesh.nodes[mesh.elements]
    native = mesh.mesh
    if mesh.nodes.shape[1] == 1:
        measures = np.abs(points[:, 1, 0] - points[:, 0, 0])
        h = measures
    else:
        measures = 0.5 * np.abs(np.linalg.det(points[:, 1:] - points[:, :1]))
        h = np.maximum.reduce([
            np.linalg.norm(points[:, i] - points[:, j], axis=1)
            for i, j in ((0, 1), (1, 2), (2, 0))
        ])
    squared = h**2 * measures * charge**2
    adjacent = native.f2t
    first, second = adjacent
    facets = native.facets.T
    # A face is a Dirichlet face only if all its nodal trace DOFs are fixed.
    active = ~np.all(fixed[facets], axis=1)
    interior = (second >= 0) & active
    boundary = (second < 0) & active
    if mesh.nodes.shape[1] == 1:
        normals = np.ones((len(facets), 1))
        face_measure = np.ones(len(facets))
        face_h = h[first].copy()
        face_h[interior] = 0.5 * (h[first[interior]] + h[second[interior]])
    else:
        edges = mesh.nodes[facets[:, 1]] - mesh.nodes[facets[:, 0]]
        face_measure = np.linalg.norm(edges, axis=1)
        face_h = face_measure
        normals = np.column_stack((-edges[:, 1], edges[:, 0])) / face_measure[:, None]
    jump = np.zeros(len(facets))
    jump[interior] = np.sum(
        (displacement[first[interior]] - displacement[second[interior]]) * normals[interior], axis=1
    )
    # Unconstrained exterior faces have homogeneous natural flux conditions.
    jump[boundary] = np.sum(displacement[first[boundary]] * normals[boundary], axis=1)
    contribution = face_h * face_measure * jump**2
    np.add.at(squared, first[interior], 0.5 * contribution[interior])
    np.add.at(squared, second[interior], 0.5 * contribution[interior])
    np.add.at(squared, first[boundary], contribution[boundary])
    scale = float(np.sum(measures * np.sum(displacement**2, axis=1)))
    relative = np.sqrt(float(np.sum(squared)) / max(scale, np.finfo(float).tiny))
    return squared, float(relative)


def refine_marked(mesh: FEMMesh, marked: np.ndarray) -> FEMMesh:
    """Bisect marked cells and close hanging nodes, preserving parent materials."""
    native = mesh.mesh.refined(marked)
    nodes = np.asarray(native.p.T, dtype=float)
    elements = np.asarray(native.t.T, dtype=np.int64)
    centers = nodes[elements].mean(axis=1)
    parents = mesh.mesh.element_finder()(*centers.T)
    minimum, maximum = _mesh_extrema(nodes, elements)
    return replace(
        mesh, mesh=native, nodes=nodes, elements=elements,
        element_tags=mesh.element_tags[parents],
        info=replace(mesh.info, nodes=len(nodes), elements=len(elements),
                     minimum_edge=minimum, maximum_edge=maximum),
    )
