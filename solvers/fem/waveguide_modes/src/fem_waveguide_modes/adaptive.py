"""Discretization residuals and adaptive entry points for waveguide modes."""

from dataclasses import replace
import numpy as np
from skfem import MeshLine

from fem_adaptivity import bulk_mark, maxwell_jump_residual, mixed_mode_fields, run_adaptive, validate_controls
from .exceptions import ConfigurationError, SolverError


def estimate_1d(solver, result):
    nodes = solver.mesh_data.nodes
    widths = solver.k0 * np.diff(nodes)
    centers = (nodes[:-1] + nodes[1:]) / 2
    eps, mu = solver.geometry.transformed_material_at(centers)
    combined = np.zeros(len(widths))
    maximum = 0.0
    for mode in result:
        u = np.asarray(mode.metadata["nodal_primary"])
        active = np.asarray(mode.fields.metadata["active_elements"], dtype=bool)
        if mode.polarization == "TE":
            a, b, stiffness = eps[1], 1 / mu[0], 1 / mu[2]
        else:
            a, b, stiffness = mu[1], 1 / eps[0], 1 / eps[2]
        primary = (u[:-1] + u[1:]) / 2
        derivative = np.diff(u) / widths
        flux = stiffness * derivative
        # Within a PML, include the derivative of its spatially varying
        # coefficient rather than treating it as elementwise constant.
        offset = np.diff(nodes) * 1e-5
        e0, m0 = solver.geometry.transformed_material_at(centers - offset)
        e1, m1 = solver.geometry.transformed_material_at(centers + offset)
        s0 = 1 / (m0[2] if mode.polarization == "TE" else e0[2])
        s1 = 1 / (m1[2] if mode.polarization == "TE" else e1[2])
        strong = (s1 - s0) / (2 * solver.k0 * offset) * derivative + (a - mode.neff**2 * b) * primary
        squared = widths**3 * abs(strong)**2 * active
        # Use one-sided material values at interfaces for physical flux jumps.
        el, ml = solver.geometry.transformed_material_at(nodes[1:-1] - np.minimum(np.diff(nodes)[:-1], np.diff(nodes)[1:]) * 1e-8)
        er, mr = solver.geometry.transformed_material_at(nodes[1:-1] + np.minimum(np.diff(nodes)[:-1], np.diff(nodes)[1:]) * 1e-8)
        sl = 1 / (ml[2] if mode.polarization == "TE" else el[2])
        sr = 1 / (mr[2] if mode.polarization == "TE" else er[2])
        jumps = abs(sl * derivative[:-1] - sr * derivative[1:])**2
        jumps *= active[:-1] & active[1:]
        squared[:-1] += widths[:-1] * jumps / 2
        squared[1:] += widths[1:] * jumps / 2
        scale = float(np.sum(widths * (abs(flux)**2 + abs(b * primary)**2) * active))
        normalized = squared / max(scale, np.finfo(float).tiny)
        combined = np.maximum(combined, normalized)
        maximum = max(maximum, float(np.sqrt(normalized.sum())))
    return combined, maximum


def solve_1d(solver, args, options, max_refinements, adaptive_tolerance):
    validate_controls(max_refinements, adaptive_tolerance, ConfigurationError)
    if not solver.discretized:
        solver.mesh(**getattr(solver, "_mesh_settings", {"resolution": 24, "wavelength_elements": 4}))

    def refine(indicators):
        old = solver.mesh_data
        marked = np.arange(len(old.nodes) - 1) if indicators is None else bulk_mark(indicators)
        nodes = np.sort(np.concatenate((old.nodes, (old.nodes[marked] + old.nodes[marked + 1]) / 2)))
        info = replace(old.info, nodes=len(nodes), elements=len(nodes) - 1,
                       minimum_edge=float(np.diff(nodes).min()), maximum_edge=float(np.diff(nodes).max()))
        solver.mesh_data = replace(old, mesh=MeshLine(nodes), nodes=nodes, info=info)
        # A later explicit refine() must not rebuild from the original coarse
        # resolution and discard the smaller intervals created by adaptation.
        solver._discretization_settings = {
            **solver._discretization_settings, "resolution": None,
            "max_element_size": info.minimum_edge,
        }
        solver._clear_result_views()

    return run_adaptive(solver, lambda: solver._solve_once(*args, **options),
                        lambda result: estimate_1d(solver, result), refine,
                        max_refinements=max_refinements, adaptive_tolerance=adaptive_tolerance,
                        error_type=ConfigurationError, retry_errors=(SolverError,))


def solve_2d(solver, args, options, max_refinements, adaptive_tolerance):
    validate_controls(max_refinements, adaptive_tolerance, ConfigurationError)
    if not solver.discretized:
        solver.mesh(**getattr(solver, "_mesh_settings", {"wavelength_elements": 4}))

    def estimate(result):
        system = solver.system
        combined = np.zeros(system.physical_mesh.nelements)
        maximum = 0.0
        # All exterior conditions (PEC, PMC, SIBC) are already imposed by
        # the weak/essential forms. Only interior continuity is estimated.
        material = lambda x, y: system.material_at(x / solver.k0, y / solver.k0)
        for index, mode in enumerate(result):
            full = solver.coefficients[:, index]
            fields = lambda basis, coefficient: mixed_mode_fields(basis, full, mode.neff, coefficient)
            indicator, residual = maxwell_jump_residual(system.basis, fields, material, axes=(0, 1))
            combined = np.maximum(combined, indicator / max(indicator.sum(), np.finfo(float).tiny) * residual**2)
            maximum = max(maximum, residual)
        return combined, maximum

    return run_adaptive(solver, lambda: solver._solve_once(*args, **options), estimate,
                        lambda indicators: solver.refine(1.5),
                        max_refinements=max_refinements, adaptive_tolerance=adaptive_tolerance,
                        error_type=ConfigurationError, retry_errors=(SolverError,))
