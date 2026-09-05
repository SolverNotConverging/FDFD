"""Paper-faithful refined shift-and-invert Arnoldi for sparse pencils."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType
from typing import Literal

import numpy as np
from scipy import linalg
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import splu

from . import _numpy_kernels

PublicBackend = Literal["auto", "cython", "python"]
KernelBackend = Literal["auto", "cython", "python", "numpy"]
ResolvedBackend = Literal["cython", "python"]

try:
    _cython_kernels = import_module(f"{__package__}._cython_kernels")
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}._cython_kernels":
        raise
    _cython_kernels = None


def _immutable_array(
    value: np.ndarray,
    *,
    dtype,
    order: Literal["C", "F"] = "C",
) -> np.ndarray:
    """Copy an array onto an immutable bytes-backed buffer."""
    copied = np.array(value, dtype=dtype, order=order, copy=True)
    frozen = np.frombuffer(copied.tobytes(order=order), dtype=copied.dtype)
    return frozen.reshape(copied.shape, order=order)


@dataclass(frozen=True, slots=True)
class ArnoldiResult:
    """Immutable result of a refined shift-and-invert Arnoldi solve."""

    eigenvalues: np.ndarray = field(repr=False)
    eigenvectors: np.ndarray = field(repr=False)
    physical_residuals: np.ndarray = field(repr=False)
    projected_residuals: np.ndarray = field(repr=False)
    restart_count: int
    step_count: int
    converged: bool
    resolved_backend: ResolvedBackend

    def __post_init__(self) -> None:
        eigenvalues = _immutable_array(
            self.eigenvalues, dtype=np.complex128
        ).reshape(-1)
        eigenvectors = _immutable_array(
            self.eigenvectors, dtype=np.complex128, order="F"
        )
        physical = _immutable_array(
            self.physical_residuals, dtype=np.float64
        ).reshape(-1)
        projected = _immutable_array(
            self.projected_residuals, dtype=np.float64
        ).reshape(-1)

        if eigenvectors.ndim != 2:
            raise ValueError("eigenvectors must be a two-dimensional array.")
        mode_count = eigenvalues.size
        if eigenvectors.shape[1] != mode_count:
            raise ValueError("The eigenvector and eigenvalue counts must match.")
        if physical.size != mode_count or projected.size != mode_count:
            raise ValueError("Residual counts must match the eigenvalue count.")
        if int(self.restart_count) < 0 or int(self.step_count) < 0:
            raise ValueError("Arnoldi work counts must be nonnegative.")
        if self.resolved_backend not in {"cython", "python"}:
            raise ValueError("resolved_backend must be 'cython' or 'python'.")

        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "physical_residuals", physical)
        object.__setattr__(self, "projected_residuals", projected)
        object.__setattr__(self, "restart_count", int(self.restart_count))
        object.__setattr__(self, "step_count", int(self.step_count))
        object.__setattr__(self, "converged", bool(self.converged))

    @property
    def backend(self) -> ResolvedBackend:
        """Alias for the resolved public backend name."""
        return self.resolved_backend

    @property
    def residuals(self) -> np.ndarray:
        """Compatibility alias for physical residuals."""
        return self.physical_residuals

    @property
    def restarts(self) -> int:
        return self.restart_count

    @property
    def steps(self) -> int:
        return self.step_count


def native_backend_available() -> bool:
    """Return whether the optional Cython kernel module can be imported."""
    return _cython_kernels is not None


def resolve_kernel_backend(backend: KernelBackend = "auto") -> str:
    """Resolve a kernel backend, retaining the legacy ``numpy`` spelling."""
    backend = str(backend).lower()
    if backend == "auto":
        return "cython" if native_backend_available() else "numpy"
    if backend == "cython":
        if not native_backend_available():
            raise ImportError(
                "The Cython periodic-eigensolver backend is not built. "
                "Install the periodic_eigensolver distribution or choose "
                "backend='python' (legacy kernel_backend='numpy')."
            )
        return backend
    if backend in {"python", "numpy"}:
        return "numpy"
    raise ValueError(
        "kernel_backend must be 'auto', 'cython', 'python', or the legacy "
        "'numpy' alias."
    )


def resolve_backend(backend: PublicBackend = "auto") -> ResolvedBackend:
    """Resolve the public backend name to ``python`` or ``cython``."""
    backend = str(backend).lower()
    if backend not in {"auto", "cython", "python"}:
        raise ValueError("backend must be 'auto', 'cython', or 'python'.")
    resolved = resolve_kernel_backend(backend)
    return "cython" if resolved == "cython" else "python"


def _kernel_module(backend: KernelBackend) -> ModuleType:
    resolved = resolve_kernel_backend(backend)
    if resolved == "cython":
        return _cython_kernels
    return _numpy_kernels


def _default_arnoldi_ncv(n: int, num_modes: int, ncv: int | None) -> int:
    if n <= num_modes:
        raise ValueError(f"Not enough DOFs to solve {num_modes} modes.")
    if ncv is None:
        ncv = max(20, 4 * int(num_modes) + 8)
    ncv = max(int(num_modes) + 2, int(ncv))
    return min(n, ncv)


def _normalised_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Cannot normalize a zero or nonfinite vector.")
    return vector / norm


def _writable_work_vector(vector: np.ndarray, size: int) -> np.ndarray:
    work = np.asarray(vector, dtype=np.complex128).reshape(-1)
    if work.size != size:
        raise ValueError(
            f"Shift-invert operator returned {work.size} entries; expected {size}."
        )
    if not (
        work.flags.c_contiguous
        and work.flags.writeable
        and work.flags.owndata
    ):
        work = np.array(work, dtype=np.complex128, order="C", copy=True)
    return work


def _orthogonal_completion(
    basis: np.ndarray,
    rng: np.random.Generator,
    breakdown_tolerance: float,
) -> np.ndarray | None:
    """Return a deterministic seeded vector orthogonal to ``basis``."""
    size = basis.shape[0]
    for _attempt in range(4):
        candidate = rng.standard_normal(size) + 1j * rng.standard_normal(size)
        for _pass in range(2):
            candidate -= basis @ (basis.conj().T @ candidate)
        norm = float(np.linalg.norm(candidate))
        if np.isfinite(norm) and norm > breakdown_tolerance:
            return np.asarray(candidate / norm, dtype=np.complex128)
    return None


def _arnoldi_factorization(
    apply_op,
    n: int,
    ncv: int,
    v0: np.ndarray,
    *,
    kernel_backend: KernelBackend = "auto",
    breakdown_tolerance: float = 1e-14,
    continue_after_breakdown: bool = False,
    augmentation_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``V_(k+1), Hbar_k`` for the shift-invert Arnoldi relation."""
    kernels = _kernel_module(kernel_backend)
    basis = np.zeros((n, ncv + 1), dtype=np.complex128, order="F")
    hessenberg = np.zeros((ncv + 1, ncv), dtype=np.complex128, order="F")
    basis[:, 0] = _normalised_vector(v0)

    completed_steps = 0
    for column in range(ncv):
        work = _writable_work_vector(apply_op(basis[:, column]), n)
        beta = kernels.arnoldi_step(
            basis,
            hessenberg,
            work,
            column,
            float(breakdown_tolerance),
        )
        completed_steps = column + 1
        if beta <= breakdown_tolerance:
            if not continue_after_breakdown or column + 1 >= ncv:
                break
            if augmentation_rng is None:
                augmentation_rng = np.random.default_rng(0)
            completion = _orthogonal_completion(
                basis[:, : column + 1],
                augmentation_rng,
                float(breakdown_tolerance),
            )
            if completion is None:
                break
            basis[:, column + 1] = completion

    active_basis = basis[:, : completed_steps + 1]
    active_hessenberg = np.array(
        hessenberg[: completed_steps + 1, :completed_steps],
        dtype=np.complex128,
        order="F",
        copy=True,
    )
    return active_basis, active_hessenberg


def _relative_residual_norm(
    ax: np.ndarray, bx: np.ndarray, eigenvalue: complex
) -> float:
    residual = ax - eigenvalue * bx
    scale = np.linalg.norm(ax) + abs(eigenvalue) * np.linalg.norm(bx)
    if scale == 0.0:
        scale = 1.0
    return float(np.linalg.norm(residual) / scale)


def _batched_relative_residuals(
    ax: np.ndarray,
    bx: np.ndarray,
    eigenvalues: np.ndarray,
    backend: KernelBackend,
) -> np.ndarray:
    kernels = _kernel_module(backend)
    ax = np.ascontiguousarray(ax, dtype=np.complex128)
    bx = np.ascontiguousarray(bx, dtype=np.complex128)
    eigenvalues = np.ascontiguousarray(eigenvalues, dtype=np.complex128)
    if kernels is _numpy_kernels:
        return kernels.relative_residuals(ax, bx, eigenvalues)

    output = np.empty(eigenvalues.size, dtype=np.float64)
    work = np.empty(ax.shape[0], dtype=np.complex128)
    kernels.relative_residuals(ax, bx, eigenvalues, output, work)
    return output


def _refined_candidate_data(
    A,
    B,
    V: np.ndarray,
    Hbar: np.ndarray,
    sigma: complex,
    num_modes: int,
    *,
    kernel_backend: KernelBackend = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract paper-defined refined Ritz vectors from ``Hbar``."""
    arnoldi_size = Hbar.shape[1]
    if arnoldi_size < num_modes:
        raise RuntimeError(
            "Arnoldi subspace is smaller than the requested number of modes."
        )
    if Hbar.shape != (arnoldi_size + 1, arnoldi_size):
        raise ValueError("Hbar must have shape (k + 1, k).")
    if V.shape[0] != A.shape[0] or V.shape[1] < arnoldi_size:
        raise ValueError("V is incompatible with the Arnoldi pencil.")

    projected = Hbar[:arnoldi_size, :]
    mu_values = linalg.eigvals(projected, check_finite=False)
    finite = np.isfinite(mu_values) & (np.abs(mu_values) > 1e-14)
    mu_values = mu_values[finite]
    if mu_values.size < num_modes:
        raise RuntimeError("Not enough finite Ritz values were generated.")

    mapped_values = complex(sigma) + 1.0 / mu_values
    order = np.argsort(np.abs(mapped_values - sigma))
    eigenvalues: list[complex] = []
    refined_coefficients: list[np.ndarray] = []
    projected_residuals: list[float] = []

    diagonal = np.arange(arnoldi_size)
    clusters: list[list[int]] = []
    repeated_root_rtol = 64.0 * np.finfo(np.float64).eps
    for raw_index in order:
        index = int(raw_index)
        eigenvalue = mapped_values[index]
        for cluster in clusters:
            reference = mapped_values[cluster[0]]
            scale = max(1.0, abs(eigenvalue), abs(reference))
            if abs(eigenvalue - reference) <= repeated_root_rtol * scale:
                cluster.append(index)
                break
        else:
            clusters.append([index])

    def residual_matrix(mu: complex) -> np.ndarray:
        small_residual = np.array(Hbar, dtype=np.complex128, order="F", copy=True)
        small_residual[diagonal, diagonal] -= mu
        return small_residual

    for cluster in clusters:
        if len(eigenvalues) == num_modes:
            break

        if len(cluster) == 1:
            index = cluster[0]
            try:
                _u, singular_values, vh = linalg.svd(
                    residual_matrix(mu_values[index]),
                    full_matrices=False,
                    check_finite=False,
                    lapack_driver="gesvd",
                )
            except linalg.LinAlgError:
                continue
            coefficient = np.asarray(vh[-1, :].conj(), dtype=np.complex128)
            projected_residual = float(singular_values[-1])
            if not (
                np.all(np.isfinite(coefficient))
                and np.isfinite(projected_residual)
            ):
                continue
            eigenvalues.append(complex(mapped_values[index]))
            refined_coefficients.append(coefficient)
            projected_residuals.append(projected_residual)
            continue

        # A repeated root makes the smallest singular vector non-unique. Build
        # the whole small right-singular subspace at the cluster centre, then
        # minimize each root's residual successively in its unused complement.
        representative_mu = complex(np.mean(mu_values[cluster]))
        try:
            _u, _singular_values, representative_vh = linalg.svd(
                residual_matrix(representative_mu),
                full_matrices=False,
                check_finite=False,
                lapack_driver="gesvd",
            )
        except linalg.LinAlgError:
            continue
        cluster_size = len(cluster)
        available = np.asarray(
            representative_vh[-cluster_size:, :].conj().T[:, ::-1],
            dtype=np.complex128,
            order="F",
        )

        for index in cluster:
            if len(eigenvalues) == num_modes or available.shape[1] == 0:
                break
            try:
                _u, restricted_singular_values, restricted_vh = linalg.svd(
                    residual_matrix(mu_values[index]) @ available,
                    full_matrices=False,
                    check_finite=False,
                    lapack_driver="gesvd",
                )
            except linalg.LinAlgError:
                continue
            restricted_coefficient = np.asarray(
                restricted_vh[-1, :].conj(), dtype=np.complex128
            )
            coefficient = np.asarray(
                available @ restricted_coefficient, dtype=np.complex128
            )
            coefficient /= np.linalg.norm(coefficient)
            projected_residual = float(restricted_singular_values[-1])

            if available.shape[1] > 1:
                complement = linalg.null_space(
                    restricted_coefficient.conj()[np.newaxis, :]
                )
                available = np.asfortranarray(available @ complement)
            else:
                available = np.empty(
                    (arnoldi_size, 0), dtype=np.complex128, order="F"
                )

            if not (
                np.all(np.isfinite(coefficient))
                and np.isfinite(projected_residual)
            ):
                continue
            eigenvalues.append(complex(mapped_values[index]))
            refined_coefficients.append(coefficient)
            projected_residuals.append(projected_residual)

    if len(eigenvalues) < num_modes:
        raise RuntimeError("Refined extraction produced too few candidate modes.")

    coefficient_matrix = np.asfortranarray(
        np.column_stack(refined_coefficients), dtype=np.complex128
    )
    eigenvectors = np.asfortranarray(
        V[:, :arnoldi_size] @ coefficient_matrix,
        dtype=np.complex128,
    )
    norms = np.linalg.norm(eigenvectors, axis=0)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise RuntimeError("Refined extraction produced an invalid eigenvector.")
    eigenvectors /= norms[np.newaxis, :]

    eigenvalue_array = np.asarray(eigenvalues, dtype=np.complex128)
    ax = A @ eigenvectors
    bx = B @ eigenvectors
    residuals = _batched_relative_residuals(
        ax, bx, eigenvalue_array, kernel_backend
    )
    return (
        eigenvalue_array,
        eigenvectors,
        residuals,
        np.asarray(projected_residuals, dtype=np.float64),
    )


def _refined_candidates(
    A,
    B,
    V: np.ndarray,
    Hbar: np.ndarray,
    sigma: complex,
    num_modes: int,
    *,
    kernel_backend: KernelBackend = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy three-array wrapper around refined candidate extraction."""
    eigenvalues, eigenvectors, residuals, _projected = (
        _refined_candidate_data(
            A,
            B,
            V,
            Hbar,
            sigma,
            num_modes,
            kernel_backend=kernel_backend,
        )
    )
    return eigenvalues, eigenvectors, residuals


def _restart_vector(
    eigenvectors: np.ndarray, rng: np.random.Generator, n: int
) -> np.ndarray:
    weights = np.ones(eigenvectors.shape[1], dtype=np.complex128)
    start = eigenvectors @ weights
    start += 1e-3 * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    return _normalised_vector(start)


def _candidate_score(candidate) -> tuple[float, float]:
    physical_residuals = candidate[2]
    return (
        float(np.max(physical_residuals)),
        float(np.linalg.norm(physical_residuals)),
    )


def _result_from_candidate(
    candidate,
    *,
    restart_count: int,
    step_count: int,
    converged: bool,
    backend: ResolvedBackend,
) -> ArnoldiResult:
    eigenvalues, eigenvectors, physical, projected = candidate
    return ArnoldiResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        physical_residuals=physical,
        projected_residuals=projected,
        restart_count=restart_count,
        step_count=step_count,
        converged=converged,
        resolved_backend=backend,
    )


def solve_generalized(
    A,
    B,
    *,
    sigma,
    num_modes,
    tol=1e-10,
    ncv=None,
    max_restarts=12,
    random_seed=0,
    backend: PublicBackend = "auto",
) -> ArnoldiResult:
    """Solve ``A x = lambda B x`` with refined shift-invert Arnoldi.

    Refined vectors follow the paper definition: each coefficient vector is
    the smallest right singular vector of
    ``Hbar_k - mu_i * [I_k; 0]`` for the shift-invert Ritz value ``mu_i``.
    Convergence is always checked against the original sparse pencil.
    """
    if A.shape != B.shape or len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square sparse matrices of equal shape.")
    num_modes = int(num_modes)
    if num_modes <= 0:
        raise ValueError("num_modes must be positive.")
    max_restarts = int(max_restarts)
    if max_restarts < 0:
        raise ValueError("max_restarts must be nonnegative.")
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tol must be a finite nonnegative value.")
    try:
        shift = complex(sigma)
    except (TypeError, ValueError) as exc:
        raise ValueError("sigma must be a finite scalar.") from exc
    if not (np.isfinite(shift.real) and np.isfinite(shift.imag)):
        raise ValueError("sigma must be a finite scalar.")
    resolved_backend = resolve_backend(backend)

    n = int(A.shape[0])
    ncv = _default_arnoldi_ncv(n, num_modes, ncv)

    if issparse(A):
        A = A.astype(np.complex128, copy=False)
    else:
        A = csc_matrix(np.asarray(A, dtype=np.complex128))
    if issparse(B):
        B = B.astype(np.complex128, copy=False)
    else:
        B = csc_matrix(np.asarray(B, dtype=np.complex128))
    if not (np.all(np.isfinite(A.data)) and np.all(np.isfinite(B.data))):
        raise ValueError("A and B must contain only finite values.")

    shifted = (A - shift * B).tocsc()
    try:
        lu = splu(shifted)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"The shifted pencil A - sigma*B could not be factorized at "
            f"sigma={shift!r}; choose a shift away from an eigenvalue."
        ) from exc

    def apply_shift_invert(vector):
        right_hand_side = np.asarray(B @ vector, dtype=np.complex128)
        return lu.solve(right_hand_side)

    rng = np.random.default_rng(random_seed)
    v0 = np.ones(n, dtype=np.complex128)
    v0 += 1e-3 * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )

    best = None
    total_steps = 0
    for restart_count in range(max_restarts + 1):
        basis, hessenberg = _arnoldi_factorization(
            apply_shift_invert,
            n,
            ncv,
            v0,
            kernel_backend=resolved_backend,
            continue_after_breakdown=num_modes > 1,
            augmentation_rng=rng,
        )
        completed_steps = int(hessenberg.shape[1])
        total_steps += completed_steps
        if completed_steps < num_modes:
            raise RuntimeError(
                "Arnoldi breakdown occurred after "
                f"{completed_steps} step(s), before {num_modes} requested "
                "modes could be extracted."
            )

        candidate = _refined_candidate_data(
            A,
            B,
            basis,
            hessenberg,
            shift,
            num_modes,
            kernel_backend=resolved_backend,
        )
        physical_residuals = candidate[2]
        if not np.all(np.isfinite(physical_residuals)):
            raise RuntimeError("Refined extraction produced nonfinite residuals.")

        if best is None or _candidate_score(candidate) < _candidate_score(best):
            best = candidate
        if np.all(physical_residuals <= tolerance):
            return _result_from_candidate(
                candidate,
                restart_count=restart_count,
                step_count=total_steps,
                converged=True,
                backend=resolved_backend,
            )

        if restart_count < max_restarts:
            best_physical = best[2]
            unconverged = best_physical > tolerance
            restart_vectors = best[1][:, unconverged]
            if restart_vectors.shape[1] == 0:
                restart_vectors = best[1]
            v0 = _restart_vector(restart_vectors, rng, n)

    return _result_from_candidate(
        best,
        restart_count=max_restarts,
        step_count=total_steps,
        converged=False,
        backend=resolved_backend,
    )




__all__ = [
    "ArnoldiResult",
    "native_backend_available",
    "resolve_backend",
    "resolve_kernel_backend",
    "solve_generalized",
]
