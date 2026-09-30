"""Modal electric-to-magnetic boundary maps with an impedance complement.

The map matches every retained outgoing mode. The complement is a local
normal-incidence impedance approximation, not an exact radiation-continuum
DtN map. Increasing the retained mode set and moving the port into a uniform
lead are independent convergence controls.
"""
from dataclasses import replace

import numpy as np
from scipy.sparse import coo_matrix
from skfem import FacetBasis, BilinearForm, LinearForm, asm

from .constants import ETA_0
from .exceptions import ConfigurationError
from .fem import evaluate_diagonal_coefficient
from .operators import electric_field_vector


@BilinearForm(dtype=np.complex128)
def _impedance(et, ey, vt, vy, w):
    e, v = electric_field_vector(et, ey), electric_field_vector(vt, vy)
    return w.factor * np.sum(np.conj(v[:2]) * w.admittance * e[:2], axis=0)


@LinearForm(dtype=np.complex128)
def _trace(vt, vy, w):
    return np.sum(np.conj(electric_field_vector(vt, vy)[:2]) * w["values"], axis=0)


def apply_matched_ports(system, modes, *, condition_limit=1e12):
    """Replace both outer z PEC boundaries by retained-mode matched ports.

    This acts on the scattered field; the analytic incident mode and compact
    contrast/PEC sources remain the excitation. No second launch term is added.
    """
    if not modes:
        raise ConfigurationError("Matched ports require at least one lead mode.")
    mesh = system.basis.mesh
    ends = (float(mesh.p[1].min()), float(mesh.p[1].max()))
    matrix = system.matrix.copy()
    port_facets = []
    conditions = []
    for sign, z in zip((-1, 1), ends):
        facets = mesh.boundary_facets()
        tol = 1e-10 * np.ptp(mesh.p[1])
        facets = facets[np.all(abs(mesh.p[1, mesh.facets[:, facets]] - z) < tol, axis=0)]
        port_facets.extend(facets)
        fb = FacetBasis(mesh, system.basis.elem, facets=facets,
                        dofs=system.basis.dofs, intorder=system.quadrature_order)
        x, zz = fb.global_coordinates() * system.length_scale
        eps = evaluate_diagonal_coefficient(system.parameters.eps_r, x, zz, name="eps_r")
        mu = evaluate_diagonal_coefficient(system.parameters.mu_r, x, zz, name="mu_r")
        admittance = np.sqrt(np.stack((eps[0] / mu[1], eps[1] / mu[0])))
        factor = 1j * system.dimensionless_k0
        matrix += asm(_impedance, fb, factor=factor, admittance=admittance)
        electric, traction = [], []
        for mode in modes:
            outgoing = mode if sign == 1 else mode.counterpropagating()
            e, h = outgoing.sample_E(x), outgoing.sample_H(x)
            electric.append(e[:2])
            traction.append(sign * ETA_0 * np.stack((h[1], -h[0])))
        electric = np.asarray(electric)
        gram = np.einsum("mcij,ncij,ij->mn", electric.conj(), electric, fb.dx)
        condition = float(np.linalg.cond(gram))
        if not np.isfinite(condition) or condition > condition_limit:
            raise ConfigurationError("Matched-port electric traces are linearly dependent or ill-conditioned.")
        conditions.append(condition)
        overlap = np.asarray([asm(_trace, fb, values=e).conj() for e in electric])
        correction = np.asarray([
            asm(_trace, fb, values=factor * (t - admittance * e))
            for t, e in zip(traction, electric)
        ]).T
        dofs = np.asarray(fb.get_dofs(facets=facets).all())
        block = correction[dofs] @ np.linalg.solve(gram, overlap[:, dofs])
        rows, cols = np.meshgrid(dofs, dofs, indexing="ij")
        matrix += coo_matrix((block.ravel(), (rows.ravel(), cols.ravel())), shape=matrix.shape).tocsr()
    return replace(system, matrix=matrix.tocsr(), port_facets=np.asarray(port_facets, dtype=np.int64)), {
        "port_boundary": "matched-modal-with-impedance-complement",
        "matched_port_mode_count": len(modes),
        "matched_port_condition_numbers": conditions,
    }
