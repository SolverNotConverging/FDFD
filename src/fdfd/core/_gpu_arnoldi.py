"""Custom CUDA refined Arnoldi with CPU Ritz extraction.

Hybrid CPU Ritz extraction is selected by auto; resident CUDA Ritz extraction
requires an explicit gpu selection.
"""

from pathlib import Path

import numpy as np

from ._gpu_validation import positive_integer, positive_real
from ._refined_ritz import refined_coefficients
from .errors import ConfigurationError, SolverError


def iterate_hybrid_device(
    operator, target, shift, count, tolerance, ncv, max_iterations, random_seed, initial_vector=None
):
    """Apply refined Arnoldi to a device operator with original-pencil residuals.

    ``max_iterations`` limits operator applications on GPU. Without an explicit
    limit, use the historical thirteen Arnoldi cycles. Return a converged subset
    when the budget expires, as the CPU ARPACK path does after nonconvergence.
    """
    import cupy as cp

    size = operator.shape[0]
    if size < 2:
        raise ConfigurationError("Sparse eigensolving requires at least two unknowns.")
    count = positive_integer(count, "count")
    if count >= size:
        raise ConfigurationError("count must be smaller than the eigenproblem size.")
    tolerance = positive_real(tolerance, "tolerance")
    if ncv is None:
        # The former GPU benchmarks used ncv=80. Smaller spaces can repeatedly
        # converge the nearby roots while losing the outer candidate cluster.
        ncv = min(size, max(80, 4 * count + 8))
    else:
        ncv = positive_integer(ncv, "ncv", count + 1)
        if ncv > size:
            raise ConfigurationError("ncv must not exceed the eigenproblem size.")
    budget = 13 * ncv if max_iterations is None else positive_integer(max_iterations, "max_iterations")
    rng = np.random.default_rng(positive_integer(random_seed, "random_seed", 0))
    module = cp.RawModule(code=Path(__file__).with_name("_arnoldi.cu").read_text(), options=("--std=c++11",))
    project = module.get_function("project")
    subtract = module.get_function("subtract_projection")
    start = cp.asarray(np.ones(size) + 1e-3 * (rng.standard_normal(size) + 1j * rng.standard_normal(size)))
    if initial_vector is not None:
        initial = np.asarray(initial_vector, dtype=complex)
        if initial.shape != (size,) or not np.isfinite(initial).all() or np.linalg.norm(initial) == 0:
            raise ConfigurationError("v0 must be a finite nonzero vector matching the eigenproblem size.")
        start = cp.asarray(initial)
    best, best_score, steps = None, None, 0
    cycles = scalar_bytes = numeric_bytes = 0
    while steps < budget:
        width = min(ncv, budget - steps)
        if width < count:
            break
        basis = cp.zeros((size, width + 1), dtype=cp.complex128, order="F")
        hbar = cp.zeros((width + 1, width), dtype=cp.complex128, order="F")
        coefficients = cp.empty(width, dtype=cp.complex128)
        basis[:, 0] = start / cp.linalg.norm(start)
        completed = 0
        for column in range(width):
            work = operator.matvec(basis[:, column])
            active = column + 1
            for _ in range(2):
                project((active,), (256,), (basis, work, coefficients, np.int32(size), np.int32(active)))
                subtract(
                    ((size + 255) // 256,),
                    (256,),
                    (basis, work, coefficients, np.int32(size), np.int32(active)),
                )
                hbar[:active, column] += coefficients[:active]
            norm = float(cp.linalg.norm(work).get())
            scalar_bytes += 8
            if not np.isfinite(norm):
                raise SolverError("GPU Arnoldi produced a nonfinite work vector.")
            hbar[active, column] = norm
            completed = active
            if norm <= 1e-14:
                if count == 1 or active == width:
                    break
                # Augment an invariant subspace to retain repeated eigenvalues.
                work = cp.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size))
                for _ in range(2):
                    project((active,), (256,), (basis, work, coefficients, np.int32(size), np.int32(active)))
                    subtract(
                        ((size + 255) // 256,),
                        (256,),
                        (basis, work, coefficients, np.int32(size), np.int32(active)),
                    )
                norm = float(cp.linalg.norm(work).get())
                scalar_bytes += 8
                if norm <= 1e-14:
                    break
            basis[:, active] = work / norm
        steps += completed
        if completed < count:
            break
        cycles += 1
        projected_matrix = cp.asnumpy(hbar[: completed + 1, :completed])
        numeric_bytes += projected_matrix.nbytes
        try:
            values, small_vectors, projected_residuals = refined_coefficients(projected_matrix, shift, count)
        except RuntimeError as exc:
            raise SolverError("GPU refined extraction failed.") from exc
        vectors = basis[:, :completed] @ cp.asarray(small_vectors)
        vectors /= cp.linalg.norm(vectors, axis=0)[None, :]
        residuals = cp.asnumpy(operator.residual(vectors, cp.asarray(values)))
        numeric_bytes += residuals.nbytes
        if not np.isfinite(residuals).all():
            raise SolverError("GPU refined extraction produced nonfinite residuals.")
        score = (
            int(np.count_nonzero(residuals > tolerance)),
            float(np.max(residuals)),
            float(np.linalg.norm(residuals)),
        )
        if best is None or score < best_score:
            best, best_score = (values, vectors, residuals, projected_residuals), score
        if np.all(residuals <= tolerance):
            break
        # Near-cutoff roots can reach the accuracy floor of the original
        # pencil even after the transformed Ritz problem has converged. Further
        # restarts cannot repair that conditioning; return only the candidates
        # that pass the independent original-pencil check below.
        if np.all(projected_residuals <= tolerance * max(1.0, np.linalg.norm(projected_matrix))):
            break
        unconverged = best[2] > tolerance
        selected = best[1][:, unconverged] if unconverged.any() else best[1]
        start = selected.sum(axis=1) + cp.asarray(
            1e-3 * (rng.standard_normal(size) + 1j * rng.standard_normal(size))
        )
    if best is None or not np.any(best[2] <= tolerance):
        raise SolverError(f"GPU refined Arnoldi did not converge in {steps} operator applications.")
    return (
        cp.asarray(best[0]),
        best[1],
        cp.asarray(best[2]),
        dict(
            steps=steps,
            cycles=cycles,
            ritz_backend="scipy-eigvals-svd",
            application_control_download_bytes=scalar_bytes,
            application_numeric_download_bytes=numeric_bytes,
            projected_residuals=best[3].tolist(),
        ),
    )


def iterate_hybrid(
    operator, target, shift, count, tolerance, ncv, max_iterations, random_seed, *, vector_rows=None
):
    """GPU Arnoldi with CPU Ritz extraction and final physical-vector export."""
    import cupy as cp

    values, vectors, residuals, execution = iterate_hybrid_device(
        operator, target, shift, count, tolerance, ncv, max_iterations, random_seed
    )
    indices = cp.flatnonzero(cp.isfinite(values) & cp.isfinite(residuals) & (residuals <= tolerance))
    if not indices.size:
        raise SolverError(
            f"GPU hybrid Arnoldi did not converge in {execution['steps']} operator applications."
        )
    indices = indices[cp.argsort(cp.abs(values[indices] - target))]
    return cp.asnumpy(values[indices]), cp.asnumpy(vectors[:vector_rows, indices])


def probe_hybrid_available():
    """Check custom CUDA kernels and CPU Ritz extraction without CUDA geev/SVD."""
    import cupy as cp

    module = cp.RawModule(code=Path(__file__).with_name("_arnoldi.cu").read_text(), options=("--std=c++11",))
    basis = cp.eye(2, dtype=cp.complex128, order="F")
    work = cp.asarray([1.0 + 2j, 3.0 - 1j], dtype=cp.complex128)
    coefficients = cp.empty(2, dtype=cp.complex128)
    parameters = (basis, work, coefficients, np.int32(2), np.int32(2))
    module.get_function("project")((2,), (256,), parameters)
    module.get_function("subtract_projection")((1,), (256,), parameters)
    values, _, residuals = refined_coefficients(
        np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], complex), 0.0, 2
    )
    if not (
        np.allclose(cp.asnumpy(work), 0.0, atol=1e-12)
        and np.allclose(values, [0.5, 1.0], atol=1e-12)
        and np.all(np.isfinite(residuals) & (residuals < 1e-12))
    ):
        raise RuntimeError("The hybrid CUDA kernel/CPU Ritz readiness check returned an incorrect result.")
    cp.cuda.get_current_stream().synchronize()
