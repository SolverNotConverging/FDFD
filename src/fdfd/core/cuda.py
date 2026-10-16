"""Optional cuDSS direct solves and custom device-resident refined Arnoldi.

Numerical Krylov vectors, Ritz eigensolving/SVD and backward-error checks stay
on CUDA. CPU work prepares sparse matrices and exports the final results.
"""

from types import SimpleNamespace

import numpy as np
from scipy.sparse import csc_matrix, eye

from .errors import SolverError


def _runtime():
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
    except ImportError as exc:
        raise ImportError("CUDA requires CuPy; install fdfd[cuda12] or fdfd[cuda13].") from exc
    from ._cudss import CudssFactor, configure_dll_search

    configure_dll_search()
    capability = int(cp.cuda.Device().compute_capability)
    version = cp.cuda.runtime.runtimeGetVersion()
    if capability < 60 or (version >= 13000 and capability < 75):
        raise RuntimeError("Unsupported CUDA/device combination; CUDA 13 needs Turing or newer.")
    return cp, csr_matrix, CudssFactor


class CudaFactor:
    """Own a reusable cuDSS factor with deterministic host-vector solves."""

    def __init__(self, matrix):
        self.factor = None
        cp, csr_matrix, factor_type = _runtime()
        self.cp = cp
        self.factor = factor_type(csr_matrix(matrix))

    def solve(self, rhs):
        return self.cp.asnumpy(self.factor.solve(self.cp.asarray(rhs)))

    __call__ = solve

    def close(self):
        if self.factor is not None:
            self.factor.close()
            self.factor = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def require_available():
    """Verify a real cuDSS solve on the active device; raise if unusable."""
    with CudaFactor(eye(1, format="csr", dtype=complex)) as factor:
        actual = factor.solve(np.ones(1, dtype=complex))
    if not np.allclose(actual, 1.0):
        raise RuntimeError("The CUDA/cuDSS readiness solve returned an incorrect result.")


def available():
    """Probe the active device without caching failures or a different GPU's readiness."""
    try:
        require_available()
    except Exception:
        return False
    return True


def probe_eigen_available():
    from ._gpu_resident_arnoldi import probe_resident_available

    probe_resident_available()


def _resident_pairs(A, B, *, k, sigma, tol, ncv, v0, max_restarts, random_seed):
    from ._gpu_resident_arnoldi import iterate_device
    from ._gpu_validation import positive_integer
    from .eigs import _norm_estimate, _perturbed

    cp, csr_matrix, factor_type = _runtime()
    A = csc_matrix(A, dtype=np.complex128)
    B = eye(A.shape[0], format="csc", dtype=complex) if B is None else csc_matrix(B, dtype=complex)
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of equal shape.")
    if not np.isfinite(A.data).all() or not np.isfinite(B.data).all():
        raise ValueError("A and B must contain finite values.")
    k = positive_integer(k, "k")
    if k >= A.shape[0]:
        raise ValueError("k must be smaller than the eigenproblem size.")
    ncv = min(A.shape[0], max(96, 6 * k + 16)) if ncv is None else positive_integer(ncv, "ncv", k + 1)
    restarts = positive_integer(max_restarts, "max_restarts", 0)
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tol must be finite and nonnegative.")
    target = complex(0.0 if sigma is None else sigma)
    if not np.isfinite(target):
        raise ValueError("sigma must be finite.")
    shift = _perturbed(target)
    norm_a, norm_b = _norm_estimate(A), _norm_estimate(B)
    a_gpu, b_gpu = csr_matrix(A), csr_matrix(B)
    with factor_type(a_gpu - shift * b_gpu) as factor:

        def residual(vectors, values):
            error = cp.linalg.norm(a_gpu @ vectors - (b_gpu @ vectors) * values[None], axis=0)
            scale = (norm_a + cp.abs(values) * norm_b) * cp.linalg.norm(vectors, axis=0)
            return error / cp.maximum(scale, np.finfo(float).tiny)

        operator = SimpleNamespace(
            shape=A.shape, matvec=lambda vector: factor.solve(b_gpu @ vector), residual=residual
        )
        return iterate_device(
            operator,
            target,
            shift,
            k,
            max(tolerance, 1e-12),
            ncv,
            (restarts + 1) * ncv,
            random_seed,
            initial_vector=v0,
        )


def solve_eigenpairs(A, B=None, *, k, sigma, tol=0.0, ncv=None, v0=None, max_restarts=12, random_seed=0):
    """Return final host eigenpairs after the complete resident numerical core."""
    import cupy as cp

    from .eigs import Eigenpairs

    values, vectors, residuals, _ = _resident_pairs(
        A, B, k=k, sigma=sigma, tol=tol, ncv=ncv, v0=v0, max_restarts=max_restarts, random_seed=random_seed
    )
    if not bool(
        cp.all(cp.isfinite(values) & cp.isfinite(residuals) & (residuals <= max(1e-8, 100 * float(tol))))
    ):
        raise SolverError("CUDA eigenpairs failed the original-pencil backward-error test.")
    return Eigenpairs(cp.asnumpy(values), cp.asnumpy(vectors), cp.asnumpy(residuals))


def eigenpairs(A, B=None, *, k, sigma, tol=0.0, ncv=None, v0=None):
    """Compatibility wrapper returning the final eigenvalues and eigenvectors."""
    result = solve_eigenpairs(A, B, k=k, sigma=sigma, tol=tol, ncv=ncv, v0=v0)
    return result.values, result.vectors


def refined_generalized(A, B, *, sigma, num_modes, tol, ncv, max_restarts, random_seed):
    """Preserve the private refined-result contract using CUDA kernels."""
    import cupy as cp

    from ._refined import ArnoldiResult

    values, vectors, residuals, work = _resident_pairs(
        A,
        B,
        k=num_modes,
        sigma=sigma,
        tol=tol,
        ncv=ncv,
        v0=None,
        max_restarts=max_restarts,
        random_seed=random_seed,
    )
    converged = bool(cp.all(cp.isfinite(residuals) & (residuals <= max(float(tol), 1e-12))))
    return ArnoldiResult(
        eigenvalues=cp.asnumpy(values),
        eigenvectors=cp.asnumpy(vectors),
        physical_residuals=cp.asnumpy(residuals),
        projected_residuals=cp.asnumpy(work["projected_residuals"]),
        restart_count=work["cycles"] - 1,
        step_count=work["steps"],
        converged=converged,
        kernels="cuda",
        device="cuda",
    )
