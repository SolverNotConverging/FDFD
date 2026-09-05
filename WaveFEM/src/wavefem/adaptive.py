"""Adaptive mode and scattering workflows, separate from algebraic tolerances."""

from dataclasses import replace
import numpy as np
from fem_adaptivity import bulk_mark, maxwell_jump_residual, run_adaptive, validate_controls
from .constants import ETA_0
from .exceptions import ConfigurationError, ModeSolverError, SolverError
from .fem import evaluate_diagonal_coefficient
from .operators import electric_field_vector, modified_curl


def solve_modes(solver, options, max_refinements, adaptive_tolerance):
    validate_controls(max_refinements, adaptive_tolerance, ConfigurationError)

    def estimate(result):
        nodes = result.system.x_nodes
        widths = np.diff(nodes)
        centers = (nodes[:-1] + nodes[1:]) / 2
        delta = np.minimum(widths[:-1], widths[1:]) * 1e-8
        left, right = nodes[1:-1] - delta, nodes[1:-1] + delta
        material = lambda x: solver.cross_section.diagonal_material_at(x, k_reference=solver.frequency.k0)
        el, _ = material(left)
        er, _ = material(right)
        em, _ = material(centers)
        combined = np.zeros(len(widths))
        maximum = 0.0
        for mode in result:
            normal_d = el[0] * mode.sample_E(left)[0] - er[0] * mode.sample_E(right)[0]
            tangent_h = ETA_0 * (mode.sample_H(left)[1:] - mode.sample_H(right)[1:])
            jump = abs(normal_d)**2 + np.sum(abs(tangent_h)**2, axis=0)
            for sheet in solver.cross_section.pec_boundaries:
                jump[np.isclose(nodes[1:-1], sheet.x, rtol=0, atol=1e-10 * np.ptp(nodes))] = 0
            squared = np.zeros(len(widths))
            squared[:-1] += widths[:-1] * jump / 2
            squared[1:] += widths[1:] * jump / 2
            scale = float(np.sum(widths * (np.sum(abs(em * mode.sample_E(centers))**2, axis=0)
                                          + ETA_0**2 * np.sum(abs(mode.sample_H(centers))**2, axis=0))))
            normalized = squared / max(scale, np.finfo(float).tiny)
            combined = np.maximum(combined, normalized)
            maximum = max(maximum, float(np.sqrt(normalized.sum())))
        return combined, maximum

    def refine(indicators):
        system = solver.assemble()
        nodes = system.x_nodes
        marked = np.arange(len(nodes) - 1) if indicators is None else bulk_mark(indicators)
        solver._adaptive_nodes = np.sort(np.concatenate((nodes, (nodes[marked] + nodes[marked + 1]) / 2)))
        solver._adaptive_interfaces = solver.cross_section.interfaces

    return run_adaptive(solver, lambda: solver._solve_once(**options), estimate, refine,
                        max_refinements=max_refinements, adaptive_tolerance=adaptive_tolerance,
                        error_type=ConfigurationError, retry_errors=(ModeSolverError,))


def solve_scattering(simulation, h5_path, max_refinements, adaptive_tolerance):
    controls = simulation.solver_options
    limit = controls.max_refinements if max_refinements is None else max_refinements
    tolerance = controls.adaptive_tolerance if adaptive_tolerance is None else adaptive_tolerance
    validate_controls(limit, tolerance, ConfigurationError)
    if simulation.mesh_data is None:
        simulation.mesh(wavelength_elements=4)
    if simulation.modes is None:
        simulation.solve_modes(num_modes=1, max_refinements=limit, adaptive_tolerance=tolerance)
    if simulation.incident is None:
        simulation.set_incident_mode(0)

    def estimate(result):
        system = simulation._adaptive_system
        full = simulation._adaptive_coefficients
        length = system.length_scale

        def material(x, z):
            return (
                evaluate_diagonal_coefficient(simulation._eps_actual, length * x, length * z, name="eps_r"),
                evaluate_diagonal_coefficient(simulation._mu_actual, length * x, length * z, name="mu_r"),
            )

        def fields(basis, coefficient):
            et, ey = basis.interpolate(full)
            eps, mu = coefficient(*basis.global_coordinates())
            electric = electric_field_vector(et, ey)
            magnetic = modified_curl(et, ey, simulation.ky * length) / (1j * simulation.frequency.k0 * length * mu)
            x, z = basis.global_coordinates() * length
            inc_e, inc_h = simulation.incident.fields(x, z)
            return electric + inc_e, magnetic + ETA_0 * inc_h

        return maxwell_jump_residual(system.basis, fields, material, axes=(0, 2),
                                      excluded_facets=simulation.mesh_data.actual_pec_facets)

    def refine(indicators):
        # Regenerate monitor/PEC/PML facets using the original geometry, and
        # retain the independently adapted lead modes and launch definition.
        names = ("modes", "incident", "_incident_mode_index", "_angle_mode_request", "_angle_modes_resolved", "ky")
        saved = {name: getattr(simulation, name) for name in names}
        settings = dict(simulation._mesh_settings)
        settings["max_element_size"] = simulation.mesh_data.info.requested_maximum_edge / 1.5
        simulation.mesh(**settings)
        for name, value in saved.items():
            setattr(simulation, name, value)

    result = run_adaptive(simulation, simulation._solve_once, estimate, refine,
                          max_refinements=limit, adaptive_tolerance=tolerance,
                          error_type=ConfigurationError, retry_errors=(SolverError,))
    if h5_path is not None:
        result = replace(result, h5_path=result.save_h5(h5_path))
    return result
