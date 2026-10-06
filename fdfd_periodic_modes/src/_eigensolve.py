"""Shift-invert Arnoldi for periodic pencils with arbitrary mass matrices."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs as _arpack_eigs, splu
from fdfd_common.errors import SolverError


def eigenpair_residuals(A, B, values, vectors):
    """Relative residuals in the original, constrained Maxwell pencil."""
    left = A @ vectors
    right = (B @ vectors) * np.asarray(values)[None, :]
    scale = np.linalg.norm(left, axis=0) + np.linalg.norm(right, axis=0)
    return np.linalg.norm(left-right, axis=0) / np.maximum(scale, np.finfo(float).tiny)


def generalized_eigs(A, M, *, k, sigma, tol, ncv, v0=None):
    """Solve A v = gamma M v without assuming M is Hermitian or positive.

    Periodic Yee averages do not satisfy SciPy's generalized-M contract.
    Arnoldi therefore operates on (A - sigma M)^-1 M with the Euclidean
    inner product; gamma = sigma + 1 / theta restores the physical values.
    This also permits singular M for an even number of periodic cells.
    """
    try:
        factor = splu((A-sigma*M).tocsc())
    except RuntimeError as exc:
        raise SolverError('The periodic shifted pencil is singular; choose a nearby neff_guess.') from exc
    operator = LinearOperator(A.shape, dtype=np.complex128,
        matvec=lambda vector: factor.solve(M @ vector))
    transformed, vectors = _arpack_eigs(operator, k=k, which='LM', tol=tol, ncv=ncv, v0=v0)
    values = sigma + 1.0/transformed
    residuals = eigenpair_residuals(A, M, values, vectors)
    limit = max(1e-8, 100.0*float(tol))
    if not np.isfinite(values).all() or not np.isfinite(residuals).all() or np.max(residuals) > limit:
        raise SolverError(f'Periodic eigenpairs failed the Maxwell residual test: {np.max(residuals):.3e}.')
    return values, vectors
