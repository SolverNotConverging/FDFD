"""Mesh-conforming closed contour configuration and field capture."""
import numpy as np

from .exceptions import ConfigurationError
from .farfield import ClosedContourFields, LayeredExterior
from .monitors import sample_horizontal_monitor, sample_vertical_monitor


def exterior_from_solver(solver):
    if solver._material_background is not None:
        raise ConfigurationError("Callback backgrounds need an explicit LayeredExterior for NF2FF.")
    section = solver._cross_section()
    cuts = sorted({v for layer in section.layers for v in layer.x
                   if solver.x_span[0] < v < solver.x_span[1]}
                  | {sheet.x for sheet in section.pec_boundaries})
    edges = np.array([solver.x_span[0], *cuts, solver.x_span[1]])
    eps, mu = section.material_at((edges[:-1]+edges[1:])/2)
    sheets = {sheet.x for sheet in section.pec_boundaries}
    return LayeredExterior(np.array(cuts), eps, mu, np.array([v in sheets for v in cuts]))


def validate_contour(solver):
    """Resolve bounds and require all actual/background differences inside."""
    request = solver._nf2ff_request
    if request is None:
        return None
    (ix0, ix1), (iz0, iz1) = solver._interior_spans()
    if solver.pml.x is None:
        raise ConfigurationError("NF2FF requires an open transverse exterior with x PML.")
    xr = request[0] or (ix0, ix1)
    zr = request[1] or (solver.left_monitor, solver.right_monitor)
    if not (ix0 <= xr[0] < xr[1] <= ix1 and iz0 < zr[0] < zr[1] < iz1):
        raise ConfigurationError("Closed NF2FF contour must lie before the PML and inside the z boundaries.")
    exterior = request[2] or exterior_from_solver(solver)
    tol = 1e-10 * (ix1-ix0)
    if any(np.any(abs(exterior.interfaces-value) < tol) for value in xr):
        raise ConfigurationError("NF2FF x sides must not coincide with a material interface or PEC sheet.")
    if solver._material_actual is None:
        from .scattering import _shape_bounds
        for region in solver.geometry.perturbations:
            x, z = _shape_bounds(region)
            if not (xr[0] < x[0] <= x[1] < xr[1] and zr[0] < z[0] <= z[1] < zr[1]):
                raise ConfigurationError("The closed NF2FF contour must strictly enclose every material perturbation.")
        for slot in solver.geometry.pec_slots:
            sheet = next(s for s in solver.geometry.pec_sheets if s.name == slot.sheet_name)
            if not (xr[0] < sheet.x < xr[1] and zr[0] < slot.z[0] < slot.z[1] < zr[1]):
                raise ConfigurationError("The closed NF2FF contour must enclose every PEC slot.")
        for sheet in solver.geometry.pec_sheets:
            if not sheet.background and not (xr[0] < sheet.x < xr[1] and zr[0] < sheet.z[0] < sheet.z[1] < zr[1]):
                raise ConfigurationError("The closed NF2FF contour must enclose every inserted PEC sheet.")
    return xr, zr, exterior


def capture_contour(solver, system, coefficients):
    configuration = validate_contour(solver)
    if configuration is None:
        return None
    xr, zr, exterior = configuration
    # Verify the supplied kernel against the actual physical exterior on the
    # FEM quadrature (including callback-backed configurations).
    x, z = system.physical_coordinates()
    outside = ((x <= xr[0]) | (x >= xr[1]) | (z <= zr[0]) | (z >= zr[1]))
    eps, mu = solver._physical_material(x, z, profile="actual")
    layer = np.searchsorted(exterior.interfaces, x, side="right")
    if not (np.allclose(eps[outside], exterior.epsilon[layer[outside]], rtol=1e-10, atol=1e-12)
            and np.allclose(mu[outside], exterior.mu[layer[outside]], rtol=1e-10, atol=1e-12)):
        raise ConfigurationError("Actual material outside the contour does not match the layered radiation kernel.")
    sheets = np.array([s.x for s in solver.geometry.pec_sheets if s.background])
    if not np.array_equal(np.sort(sheets), exterior.interfaces[exterior.pec]):
        raise ConfigurationError("Background PEC sheets must match the layered radiation kernel.")
    kwargs = dict(ky=solver.ky, omega=solver.omega, mu_r=solver._mu_actual,
                  length_scale=system.length_scale, intorder=system.quadrature_order)
    points, normals, weights, electric, magnetic = [], [], [], [], []
    for value, normal in ((xr[0], (-1,0)), (xr[1], (1,0))):
        samples = sample_horizontal_monitor(system.basis, coefficients, x=value, **kwargs)
        mask = (samples.z > zr[0]) & (samples.z < zr[1])
        points.append(np.vstack((np.full(mask.sum(), value), samples.z[mask])))
        normals.append(np.tile(normal, (mask.sum(),1)).T)
        weights.append(samples.weights[mask])
        electric.append(samples.E[:,mask])
        magnetic.append(samples.H[:,mask])
    for value, normal in ((zr[0], (0,-1)), (zr[1], (0,1))):
        samples = sample_vertical_monitor(system.basis, coefficients, z=value, **kwargs)
        mask = (samples.x > xr[0]) & (samples.x < xr[1])
        points.append(np.vstack((samples.x[mask], np.full(mask.sum(), value))))
        normals.append(np.tile(normal, (mask.sum(),1)).T)
        weights.append(samples.weights[mask])
        electric.append(samples.E[:,mask])
        magnetic.append(samples.H[:,mask])
    return ClosedContourFields(xr, zr, np.concatenate(points, axis=1),
        np.concatenate(normals, axis=1), np.concatenate(weights),
        np.concatenate(electric, axis=1), np.concatenate(magnetic, axis=1),
        solver.frequency, solver.ky, exterior)
