"""Measured reduced-system diagnostics shared by the waveguide kernels."""
import numpy as np
from scipy.sparse.linalg import eigs


def solve_eigenpairs(operator, *, k, sigma, tol, wide_search=False):
    """Retry an exactly singular shift without perturbing the physical operator."""
    # A larger Krylov space helps retain repeated eigenvalues in automatic
    # multi-mode sweeps. This does not increase the returned eigenpair count.
    options = {'ncv': min(operator.shape[0], max(6*k+1, 64))} if wide_search else {}
    try:
        return eigs(operator, k=k, sigma=sigma, tol=tol, **options)
    except RuntimeError as exc:
        if 'exactly singular' not in str(exc).lower() or sigma is None:
            raise
        offset = 1e-7*max(1., abs(sigma))*(1.+.37j)
        return eigs(operator, k=k, sigma=sigma+offset, tol=tol, **options)


def eigenpair_residuals(operator, eigenvalues, vectors):
    """Infinity-norm backward errors, one per reduced eigenpair."""
    norm = float(np.max(np.asarray(abs(operator).sum(axis=1))))
    numerator = np.max(abs(operator @ vectors - vectors * eigenvalues), axis=0)
    denominator = (norm + abs(eigenvalues)) * np.max(abs(vectors), axis=0)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.inf),
                     where=denominator > 0)


def boundary_provenance(backend):
    """Record actual conductor cells; physical intent is supplied by the port."""
    closed = (backend._pec_cell_mask | backend._pmc_cell_mask |
              (backend._surface_impedance_owner >= 0))
    edges = {}
    for axis, label in enumerate(('x', 'y')[:closed.ndim]):
        for index, side in ((0, '-'), (-1, '+')):
            edges[label + side] = bool(np.all(np.take(closed, index, axis=axis)))
    return {'conductor_edges': edges, 'closed_conductor_boundary': all(edges.values()),
            'closure': 'native staggered derivative endpoint rows and explicit conductor constraints'}


def equation_residual(*terms):
    """Relative residual of a reconstructed discrete field equation."""
    numerator = np.max(abs(sum(terms)), axis=0)
    denominator = sum(np.max(abs(term), axis=0) for term in terms)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
