"""The single sparse eigensolver entry point used by every modal solver.

All methods use shift-and-invert: the eigenvalues nearest ``sigma`` of the
pencil ``A v = lambda B v`` (``B = I`` when omitted) are found from
``(A - sigma B)^-1 B``. ``B`` may be singular or non-Hermitian.

methods
    ``"arpack"``  SciPy/ARPACK implicitly restarted Arnoldi.
    ``"refined"`` Restarted Arnoldi with refined Ritz vectors (see
                  :mod:`fdfd.core._refined`); more robust for clustered or
                  defective spectra such as 3D periodic leaky modes.
devices
    ``"cpu"``     SciPy SuperLU factorization.
    ``"cuda"``    cuDSS factorization and custom resident refined Arnoldi,
                  including CUDA Ritz eigensolving/SVD and backward errors.
                  Both method selections use this resident CUDA algorithm.
    ``"auto"``    ``"cuda"`` when it is available, else ``"cpu"``.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg  # noqa: F401  (sparse.linalg.norm)
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs, splu

from .errors import BackendCapabilityError, ConfigurationError, SolverError

METHODS = ("arpack", "refined")
DEVICES = ("auto", "cpu", "cuda")


def _cpu_fallback(reason):
    warnings.warn(
        f"device='auto' is falling back to CPU: {reason}. "
        "Select device='cpu' to request CPU execution explicitly.",
        RuntimeWarning,
        stacklevel=3,
    )
    return "cpu"


def check_device(device):
    """Resolve a direct CUDA runtime, warning on auto fallback and rejecting unavailable cuda."""
    if device not in DEVICES:
        raise ConfigurationError(f"device must be one of {DEVICES}; received {device!r}.")
    if device == "cpu":
        return device
    from .cuda import available, require_available

    if device == "auto":
        return "cuda" if available() else _cpu_fallback("the CUDA/cuDSS runtime is unavailable")
    try:
        require_available()
    except Exception as exc:
        raise BackendCapabilityError(f"device='cuda' requested but CUDA/cuDSS is unavailable: {exc}") from exc
    return "cuda"


def check_eigen_device(device):
    """Require the custom resident CUDA core; never select CPU Ritz extraction."""
    selected = check_device(device)
    if selected == "cpu":
        return selected
    from .cuda import probe_eigen_available

    try:
        probe_eigen_available()
    except Exception as exc:
        reason = f"the resident CUDA eigensolver is unavailable: {exc}"
        if device == "auto":
            return _cpu_fallback(reason)
        raise BackendCapabilityError(f"device='cuda' requested but {reason}") from exc
    return "cuda"


def check_method(method):
    """Validate an eigensolver ``method`` argument."""
    if method not in METHODS:
        raise ConfigurationError(f"method must be one of {METHODS}; received {method!r}.")
    return method


@dataclass(frozen=True)
class Eigenpairs:
    """Eigenvalues, eigenvectors (columns) and their backward errors."""

    values: np.ndarray
    vectors: np.ndarray
    residuals: np.ndarray


def _norm_estimate(matrix):
    # ||M||_2 <= sqrt(||M||_1 ||M||_inf); exact for diagonal and close for Yee operators.
    if matrix is None:
        return 1.0
    return float(np.sqrt(sparse.linalg.norm(matrix, 1) * sparse.linalg.norm(matrix, np.inf)))


def backward_errors(A, B, values, vectors):
    """Normwise backward error of each eigenpair of ``A v = lambda B v``.

    ``||A v - lambda B v|| / ((||A|| + |lambda| ||B||) ||v||)``: the smallest
    relative perturbation of ``A`` and ``B`` for which the pair is exact. It
    stays meaningful for ``lambda = 0`` (e.g. static modes), and a backward
    stable solve gives values near machine precision.
    """
    values = np.asarray(values)
    left = A @ vectors
    right = vectors if B is None else B @ vectors
    error = np.linalg.norm(left - right * values[None, :], axis=0)
    scale = (_norm_estimate(A) + np.abs(values) * _norm_estimate(B)) * np.linalg.norm(vectors, axis=0)
    return error / np.maximum(scale, np.finfo(float).tiny)


def _perturbed(sigma):
    # A shift on an exact eigenvalue (e.g. the Gamma-point zero of a band
    # problem) makes A - sigma*B singular. Move the shift, not the operator.
    sigma = complex(sigma)
    return sigma + 1e-7 * max(1.0, abs(sigma)) * (1.0 + 0.37j)


def _factorize(A, B, sigma):
    shifted = (A - sigma * B).tocsc() if B is not None else (A - sigma * sparse.eye(A.shape[0])).tocsc()
    return splu(shifted)


def _arpack(A, B, *, k, sigma, tol, ncv, v0):
    try:
        factor = _factorize(A, B, sigma)
    except RuntimeError as exc:
        if "singular" not in str(exc).lower():
            raise
        sigma = _perturbed(sigma)
        try:
            factor = _factorize(A, B, sigma)
        except RuntimeError as again:
            raise SolverError("The shifted operator is singular; choose a different guess.") from again
    if B is None:
        matvec = factor.solve
    else:

        def matvec(vector):
            return factor.solve(B @ vector)

    operator = LinearOperator(A.shape, dtype=np.complex128, matvec=matvec)
    theta, vectors = eigs(operator, k=k, which="LM", tol=tol, ncv=ncv, v0=v0)
    return sigma + 1.0 / theta, vectors


def solve(
    A,
    B=None,
    *,
    k,
    sigma,
    method="arpack",
    device="cpu",
    tol=0.0,
    ncv=None,
    v0=None,
    max_restarts=12,
    random_seed=0,
):
    """Return the ``k`` eigenpairs of ``A v = lambda B v`` nearest ``sigma``.

    Parameters
    ----------
    A, B : sparse matrix
        Square operators of equal shape. ``B=None`` means the identity.
    k : int
        Number of eigenpairs.
    sigma : complex
        Shift; the eigenvalues closest to it are returned.
    method : {"arpack", "refined"}
        Krylov method (see module documentation).
    device : {"auto", "cpu", "cuda"}
        Numerical backend. CUDA uses resident refined Arnoldi for both methods;
        auto warns on CPU fallback and explicit cuda raises if unavailable.
    tol : float
        Convergence tolerance; 0 requests machine precision for ARPACK.
        The refined method uses ``max(tol, 1e-12)``.
    ncv : int, optional
        Krylov subspace size.
    v0 : ndarray, optional
        ARPACK start vector.
    max_restarts, random_seed : int
        Refined-method restart limit and start-vector seed.

    Returns
    -------
    Eigenpairs
        Unsorted eigenpairs with their backward errors.
    """
    check_method(method)
    device = check_eigen_device(device)
    A = sparse.csc_matrix(A, dtype=np.complex128)
    B = None if B is None else sparse.csc_matrix(B, dtype=np.complex128)
    if k >= A.shape[0]:
        raise ConfigurationError(f"Not enough unconstrained degrees of freedom ({A.shape[0]}) for {k} modes.")
    if device == "cuda":
        from .cuda import solve_eigenpairs

        return solve_eigenpairs(
            A,
            B,
            k=k,
            sigma=sigma,
            tol=tol,
            ncv=ncv,
            v0=v0,
            max_restarts=max_restarts,
            random_seed=random_seed,
        )
    if method == "refined":
        from ._refined import solve_generalized

        mass = B if B is not None else sparse.eye(A.shape[0], format="csc", dtype=np.complex128)
        result = solve_generalized(
            A,
            mass,
            sigma=sigma,
            num_modes=k,
            tol=max(float(tol), 1e-12),
            ncv=ncv,
            max_restarts=max_restarts,
            random_seed=random_seed,
            device=device,
        )
        values, vectors = np.asarray(result.eigenvalues), np.asarray(result.eigenvectors)
    else:
        values, vectors = _arpack(A, B, k=k, sigma=sigma, tol=tol, ncv=ncv, v0=v0)
    return Eigenpairs(values, vectors, backward_errors(A, B, values, vectors))
