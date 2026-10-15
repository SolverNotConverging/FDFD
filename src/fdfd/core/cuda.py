"""Optional cuDSS sparse direct and shift-invert solves (``device="cuda"``).

ARPACK iteration stays on the CPU. Sparse factorization, matrix products and
shifted solves use CUDA, with host/device vector transfers each iteration.
"""

from functools import lru_cache

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import LinearOperator, eigs

from .errors import SolverError


@lru_cache(maxsize=1)
def available():
    """Whether ``device="cuda"`` can run here: CuPy, a CUDA GPU, and a loadable cuDSS.

    The check runs once per process; ``device="auto"`` uses it to fall back to
    the CPU.
    """
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        from ._cudss import configure_dll_search, find_cudss_library

        configure_dll_search()
        path = find_cudss_library()
        import ctypes
        import os

        (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)(path)
    except Exception:
        return False
    return True


class CudaFactor:
    """Own a cuDSS factor with deterministic close and host-vector solves."""

    def __init__(self, matrix):
        self.factor = None
        try:
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix
        except ImportError as exc:
            raise ImportError("CUDA requires CuPy; install fdfd[cuda12] or fdfd[cuda13].") from exc
        from ._cudss import CudssFactor

        self.cp = cp
        self.factor = CudssFactor(csr_matrix(matrix))

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


def eigenpairs(A, B=None, *, k, sigma, tol=0.0, ncv=None, v0=None):
    """CUDA shift-invert with original-pencil residual validation."""
    A = csc_matrix(A, dtype=np.complex128)
    B = (
        eye(A.shape[0], format="csc", dtype=np.complex128)
        if B is None
        else csc_matrix(B, dtype=np.complex128)
    )
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of equal shape.")
    if not np.isfinite(A.data).all() or not np.isfinite(B.data).all():
        raise ValueError("A and B must contain finite values.")
    shift = complex(0.0 if sigma is None else sigma)
    shift += 1e-7 * max(1.0, abs(shift)) * (1.0 + 0.37j)
    with CudaFactor(A - shift * B) as factor:
        from cupyx.scipy.sparse import csr_matrix

        cp = factor.cp
        b_gpu = csr_matrix(B)
        operator = LinearOperator(
            A.shape,
            dtype=np.complex128,
            matvec=lambda vector: cp.asnumpy(factor.factor.solve(b_gpu @ cp.asarray(vector))),
        )
        transformed, vectors = eigs(operator, k=k, which="LM", tol=tol, ncv=ncv, v0=v0)
    values = shift + 1.0 / transformed
    left, right = A @ vectors, (B @ vectors) * values
    # Zero roots need a backward-error scale based on the uncancelled rows.
    row_scale = abs(A) @ abs(vectors) + (abs(B) @ abs(vectors)) * abs(values)
    backward = np.linalg.norm(left - right, axis=0) / np.maximum(
        np.linalg.norm(row_scale, axis=0), np.finfo(float).tiny
    )
    if (
        not np.isfinite(values).all()
        or not np.isfinite(backward).all()
        or np.max(backward) > max(1e-8, 100 * float(tol))
    ):
        raise SolverError(
            f"CUDA eigenpairs failed the original-pencil residual test: {np.max(backward):.3e}."
        )
    return values, vectors
