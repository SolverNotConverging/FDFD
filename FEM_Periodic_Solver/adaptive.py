"""Residual-controlled remeshing with regenerated periodic constraints."""

import numpy as np
from fem_adaptivity import maxwell_jump_residual, run_adaptive, validate_controls
from .exceptions import ConfigurationError, SolverError


def solve_periodic(solver, dimension, args, options, max_refinements, adaptive_tolerance):
    validate_controls(max_refinements, adaptive_tolerance, ConfigurationError)
    ready = solver.discretized if dimension == 2 else solver.system is not None
    if not ready:
        solver.discretize(wavelength_elements=4)

    def estimate(result):
        combined = np.zeros(solver.mesh_data.info.elements)
        maximum = 0.0
        for mode in result:
            system = solver._systems[mode.polarization] if dimension == 2 else solver.system
            material = lambda *coordinates: solver.geometry.transformed_material_at(
                *(coordinate / solver.k0 for coordinate in coordinates))

            def fields(basis, coefficient):
                field = basis.interpolate(mode.coefficients)
                eps, mu = coefficient(*basis.global_coordinates())
                if dimension == 3:
                    electric = np.asarray(field)
                    cross = np.stack((-electric[1], electric[0], np.zeros_like(electric[0])))
                    return electric, 1j * (field.curl - 1j * mode.neff * cross) / mu
                primary = np.asarray(field)
                derivative = field.grad
                zero = np.zeros_like(primary)
                longitudinal = derivative[1] - 1j * mode.neff * primary
                if system.polarization == "TE":
                    return np.stack((zero, primary, zero)), np.stack(
                        (-1j * longitudinal / mu[0], zero, 1j * derivative[0] / mu[2]))
                return np.stack((1j * longitudinal / eps[0], zero, -1j * derivative[0] / eps[2])), np.stack((zero, primary, zero))

            indicator, residual = maxwell_jump_residual(
                system.basis, fields, material, axes=(0, 2) if dimension == 2 else (0, 1, 2),
                periodic_node_pairs=(np.column_stack((solver.mesh_data.slave_nodes, solver.mesh_data.master_nodes))
                                     if dimension == 2 else solver.mesh_data.periodic_node_pairs),
            )
            combined = np.maximum(combined, indicator / max(indicator.sum(), np.finfo(float).tiny) * residual**2)
            maximum = max(maximum, residual)
        return combined, maximum

    return run_adaptive(solver, lambda: solver._solve_once(*args, **options), estimate,
                        lambda indicators: solver.refine(1.5),
                        max_refinements=max_refinements, adaptive_tolerance=adaptive_tolerance,
                        error_type=ConfigurationError, retry_errors=(SolverError,))
