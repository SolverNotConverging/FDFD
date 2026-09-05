"""Shared solve/estimate/refine policy and Maxwell interface residuals."""

from dataclasses import replace
from math import factorial

import numpy as np
from skfem import InteriorFacetBasis


def validate_controls(max_refinements, adaptive_tolerance, error_type=ValueError):
    try:
        valid = (not isinstance(max_refinements, (bool, np.bool_, str, bytes))
                 and np.isscalar(max_refinements)
                 and int(max_refinements) == max_refinements and max_refinements >= 0)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise error_type("max_refinements must be a nonnegative integer.")
    try:
        valid = (not isinstance(adaptive_tolerance, (bool, np.bool_, str, bytes))
                 and np.isscalar(adaptive_tolerance)
                 and np.isrealobj(adaptive_tolerance)
                 and np.isfinite(adaptive_tolerance) and adaptive_tolerance > 0)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise error_type("adaptive_tolerance must be finite and positive.")
    return int(max_refinements), float(adaptive_tolerance)


def run_adaptive(owner, solve_once, estimate, refine, *, max_refinements,
                 adaptive_tolerance, error_type=ValueError, retry_errors=()):
    """Perform one initial solve and at most max_refinements mesh updates.

    An exhausted budget returns the last solution with converged=False.  A
    failed eigensolve may retry on a finer mesh, but is never counted as a
    converged pass.  Failed remeshing restores the last usable owner state.
    """
    limit, tolerance = validate_controls(max_refinements, adaptive_tolerance, error_type)
    history = []
    for iteration in range(limit + 1):
        try:
            result = solve_once()
        except retry_errors as exc:
            history.append({"refinement": iteration, "residual": None,
                            "status": "solve_failed", "message": str(exc)})
            if iteration == limit:
                raise
            saved = owner.__dict__.copy()
            try:
                refine(None)
            except Exception:
                owner.__dict__.clear()
                owner.__dict__.update(saved)
                raise
            continue
        indicators, residual = estimate(result)
        if not np.isfinite(residual):
            raise error_type("Adaptive discretization residual is non-finite.")
        converged = residual <= tolerance
        status = "tolerance" if converged else "refinement_limit" if iteration == limit else "refine"
        history.append({"refinement": iteration, "elements": len(indicators),
                        "residual": float(residual), "status": status})
        if converged or iteration == limit:
            key = "metadata" if hasattr(result, "metadata") else "solve_info"
            updated = replace(result, **{key: {
                **getattr(result, key), "adaptive_history": tuple(history),
                "adaptive_residual": float(residual), "adaptive_converged": bool(converged),
                "max_refinements": limit, "adaptive_tolerance": tolerance,
            }})
            for name, value in tuple(vars(owner).items()):
                if value is result:
                    setattr(owner, name, updated)
            return updated
        saved = owner.__dict__.copy()
        try:
            refine(indicators)
        except Exception:
            owner.__dict__.clear()
            owner.__dict__.update(saved)
            raise


def bulk_mark(indicators, fraction=0.5):
    indices = np.argsort(-np.asarray(indicators), kind="stable")
    count = np.searchsorted(np.cumsum(indicators[indices]), fraction * np.sum(indicators)) + 1
    return indices[:count]


def cell_geometry(mesh):
    vertices = mesh.p[:, mesh.t].transpose(2, 1, 0)
    dim = mesh.dim()
    measure = np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / factorial(dim)
    h = np.maximum.reduce([np.linalg.norm(vertices[:, i] - vertices[:, j], axis=1)
                           for i in range(dim + 1) for j in range(i)])
    return h, measure


def maxwell_jump_residual(basis, fields, material, *, axes, excluded_facets=(),
                          periodic_node_pairs=None):
    """Normalized normal-D/tangential-H interface residual, using FE traces.

    `fields(basis, material)` returns E and impedance-scaled H, each shaped
    (3, cells/facets, quadrature points). Coordinates are the basis's own
    computational coordinates. Material values are sampled toward each cell
    interior to distinguish the two sides of a dielectric interface.
    Periodic envelope traces are compared across corresponding seam facets.
    This estimator is not an algebraic residual or a certified error bound.
    """
    mesh = basis.mesh
    h, measure = cell_geometry(mesh)
    electric, magnetic = fields(basis, material)
    eps, _ = material(*basis.global_coordinates())
    dfield = eps * electric
    scale = float(np.sum(basis.dx * (np.sum(abs(dfield)**2, axis=0)
                                     + np.sum(abs(magnetic)**2, axis=0))))
    indicators = np.zeros(mesh.nelements)
    facets = np.setdiff1d(np.flatnonzero(mesh.f2t[1] >= 0), excluded_facets)

    def trace(facet_ids, side):
        fb = InteriorFacetBasis(mesh, basis.elem, facets=facet_ids, side=side,
                                dofs=basis.dofs, intorder=max(4, basis.elem.maxdeg * 2))
        coords = np.asarray(fb.global_coordinates())
        centers = mesh.p[:, mesh.t[:, mesh.f2t[side, facet_ids]]].mean(axis=1)
        inside = coords + 1e-8 * (centers[:, :, None] - coords)

        def side_material(*unused):
            return material(*inside)

        e, hfield = fields(fb, side_material)
        epsilon, _ = side_material()
        return fb, epsilon * e, hfield

    if len(facets):
        fb, d0, h0 = trace(facets, 0)
        _, d1, h1 = trace(facets, 1)
        normal = np.zeros((3, *fb.normals.shape[1:]))
        normal[list(axes)] = fb.normals
        dh = h0 - h1
        normal_d = np.sum(normal * (d0 - d1), axis=0)
        tangent_h = dh - normal * np.sum(normal * dh, axis=0)
        left, right = mesh.f2t[:, facets]
        contribution = 0.5 * (h[left] + h[right]) * np.sum(
            fb.dx * (abs(normal_d)**2 + np.sum(abs(tangent_h)**2, axis=0)), axis=1)
        np.add.at(indicators, left, contribution / 2)
        np.add.at(indicators, right, contribution / 2)

    if periodic_node_pairs is not None and len(periodic_node_pairs):
        # Match boundary facets by their periodic vertex representatives.
        representative = np.arange(mesh.nvertices)
        pairs = np.asarray(periodic_node_pairs, dtype=int)
        representative[pairs[:, 0]] = pairs[:, 1]
        seam_nodes = set(pairs.ravel())
        groups = {}
        for facet in mesh.boundary_facets():
            nodes = mesh.facets[:, facet]
            if all(int(node) in seam_nodes for node in nodes):
                groups.setdefault(tuple(sorted(representative[nodes])), []).append(int(facet))
        # Facet quadrature may be permuted at the seam. Compare cell-constant
        # mean flux traces; interior jumps above retain full quadrature.
        for pair in groups.values():
            if len(pair) != 2:
                continue  # transverse walls can touch the seam at corners
            fa, da, ha = trace(np.array(pair[:1]), 0)
            fb, db, hb = trace(np.array(pair[1:]), 0)
            normal = np.zeros(3)
            normal[list(axes)] = np.mean(fa.normals[:, 0], axis=1)
            wa, wb = fa.dx[0] / fa.dx[0].sum(), fb.dx[0] / fb.dx[0].sum()
            dd = da[:, 0] @ wa - db[:, 0] @ wb
            dh = ha[:, 0] @ wa - hb[:, 0] @ wb
            cells = mesh.f2t[0, pair]
            value = float(np.mean(h[cells]) * np.sum(fa.dx) * (
                abs(normal @ dd)**2 + np.sum(abs(dh - normal * (normal @ dh))**2)))
            indicators[cells] += value / 2
    residual = float(np.sqrt(indicators.sum() / max(scale, np.finfo(float).tiny)))
    return indicators, residual


def mixed_mode_fields(basis, coefficients, neff, material):
    et, ez = basis.interpolate(coefficients)
    eps, mu = material(*basis.global_coordinates())
    electric = np.stack((et[0], et[1], np.asarray(ez)))
    curl = np.stack((ez.grad[1] + 1j * neff * et[1],
                     -ez.grad[0] - 1j * neff * et[0], et.curl))
    return electric, 1j * curl / mu
