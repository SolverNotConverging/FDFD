"""CUDA Arnoldi with CUDA Ritz eigensolving/SVD and device normalization.

Only a two-byte convergence/status packet is downloaded per restart cycle.
Numerical arrays are returned on device; the caller owns final result export.
cuDSS analysis and cuSOLVER's internal pinned host workspace remain library work.
"""

from pathlib import Path

import numpy as np

from ._gpu_validation import positive_integer, positive_real
from .errors import ConfigurationError, SolverError


def check_resident_available():
    import cupy as cp
    from cupyx.cusolver import check_availability

    available = False
    if hasattr(cp.linalg, "eigvals"):
        try:
            available = check_availability("geev")
        except (KeyError, ValueError):
            pass
    if not available:
        raise ConfigurationError(
            "Resident CUDA Ritz extraction requires CuPy/cuSOLVER with CUDA geev support."
        )


def is_available():
    """Probe optional CUDA Ritz support; numerical calls never use this fallback."""
    try:
        check_resident_available()
    except (ImportError, ConfigurationError):
        return False
    return True


def refined_gpu(hbar, shift, count):
    import cupy as cp

    width = hbar.shape[1]
    mu = cp.linalg.eigvals(hbar[:width])
    finite = cp.isfinite(mu) & (cp.abs(mu) > 1e-14)
    values = cp.where(finite, shift + 1 / cp.where(finite, mu, 1), cp.inf)
    selected = cp.argsort(cp.abs(values - shift))[:count]
    values, mu = values[selected], mu[selected]
    identity = cp.eye(width + 1, width, dtype=cp.complex128)
    residual = hbar[None, :, :] - mu[:, None, None] * identity[None, :, :]
    _, _, vh = cp.linalg.svd(residual, full_matrices=False)
    close = cp.abs(values[:, None] - values[None, :]) <= (
        64
        * np.finfo(float).eps
        * cp.maximum(1.0, cp.maximum(cp.abs(values[:, None]), cp.abs(values[None, :])))
    )
    ordinal = cp.sum(cp.tril(close, -1), axis=1)
    coefficients = vh[cp.arange(count), -1 - ordinal, :].conj().T.copy()
    # Independent vectors for repeated eigenvalues, without host clustering.
    for i in range(count):
        for _ in range(2):
            previous = coefficients[:, :i]
            weights = (previous.conj().T @ coefficients[:, i]) * close[i, :i]
            coefficients[:, i] -= previous @ weights
        coefficients[:, i] /= cp.linalg.norm(coefficients[:, i])
    projected = cp.linalg.norm(cp.einsum("kij,jk->ik", residual, coefficients), axis=0)
    return values, coefficients, projected


def _load_kernels():
    import cupy as cp

    source = Path(__file__).with_name("_arnoldi.cu").read_text()
    project_source, subtract_source = source.split('extern "C" __global__ void subtract_projection(')
    conditional = project_source.replace("void project(", "void project_breakdown(").replace(
        "int n, int columns) {", "int n, int columns, const double* norm) {\n    if (norm[0] > 1e-14) return;"
    )
    conditional += 'extern "C" __global__ void subtract_breakdown(' + subtract_source.replace(
        "int n, int columns) {", "int n, int columns, const double* norm) {\n    if (norm[0] > 1e-14) return;"
    )
    conditional += """
extern "C" __global__ void augment(cuDoubleComplex* work, const double* norm,
                                   int n, unsigned int seed) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= n || norm[0] > 1e-14) return;
    unsigned long long x = ((unsigned long long)seed << 32) + row + 1;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    work[row] = make_cuDoubleComplex(((x & 0xffffffffULL) / 2147483648.)-1.,
                                    ((x >> 32) / 2147483648.)-1.);
}
"""
    module = cp.RawModule(code=source + conditional, options=("--std=c++11",))
    project, subtract = module.get_function("project"), module.get_function("subtract_projection")
    project_b, subtract_b = (
        module.get_function("project_breakdown"),
        module.get_function("subtract_breakdown"),
    )
    augment = module.get_function("augment")
    return project, subtract, project_b, subtract_b, augment


def probe_resident_available():
    """Exercise compilation, CUDA geev and CUDA SVD before selecting GPU."""
    import cupy as cp

    check_resident_available()
    project, subtract, _, _, _ = _load_kernels()
    basis = cp.eye(2, dtype=cp.complex128, order="F")
    work = cp.asarray([1.0 + 2j, 3.0 - 1j], dtype=cp.complex128)
    coefficients = cp.empty(2, dtype=cp.complex128)
    parameters = (basis, work, coefficients, np.int32(2), np.int32(2))
    project((2,), (256,), parameters)
    subtract((1,), (256,), parameters)
    hbar = cp.asarray([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=cp.complex128)
    values, _, residuals = refined_gpu(hbar, 0.0, 2)
    valid = (
        cp.allclose(work, 0.0, atol=1e-12)
        & cp.allclose(values, cp.asarray([0.5, 1.0]), atol=1e-12)
        & cp.all(cp.isfinite(residuals) & (residuals < 1e-12))
    )
    if not bool(valid):
        raise RuntimeError("The resident CUDA kernel/Ritz readiness check returned an incorrect result.")
    cp.cuda.get_current_stream().synchronize()


def iterate_device(
    operator, target, shift, count, tolerance, ncv, max_iterations, random_seed, initial_vector=None
):
    import cupy as cp

    check_resident_available()
    size = operator.shape[0]
    if size < 2:
        raise ConfigurationError("Sparse eigensolving requires at least two unknowns.")
    count = positive_integer(count, "count")
    if count >= size:
        raise ConfigurationError("count must be smaller than the eigenproblem size.")
    tolerance = positive_real(tolerance, "tolerance")
    ncv = min(size, max(96, 6 * count + 16)) if ncv is None else positive_integer(ncv, "ncv", count + 1)
    if ncv > size:
        raise ConfigurationError("ncv must not exceed the eigenproblem size.")
    budget = 13 * ncv if max_iterations is None else positive_integer(max_iterations, "max_iterations")
    seed = positive_integer(random_seed, "random_seed", 0)
    rng = cp.random.RandomState(seed)
    project, subtract, project_b, subtract_b, augment = _load_kernels()
    start = cp.ones(size, dtype=complex) + 1e-3 * (rng.standard_normal(size) + 1j * rng.standard_normal(size))
    if initial_vector is not None:
        initial = np.asarray(initial_vector, dtype=complex)
        if initial.shape != (size,) or not np.isfinite(initial).all() or np.linalg.norm(initial) == 0:
            raise ConfigurationError("v0 must be a finite nonzero vector matching the eigenproblem size.")
        start = cp.asarray(initial)
    best = None
    steps = cycles = 0
    while steps < budget:
        width = min(ncv, budget - steps)
        if width < count:
            break
        basis = cp.zeros((size, width + 1), dtype=complex, order="F")
        hbar = cp.zeros((width + 1, width), dtype=complex, order="F")
        coefficients = cp.empty(width, dtype=complex)
        basis[:, 0] = start / cp.linalg.norm(start)
        for column in range(width):
            active = column + 1
            work = operator.matvec(basis[:, column])
            parameters = (basis, work, coefficients, np.int32(size), np.int32(active))
            for _ in range(2):
                project((active,), (256,), parameters)
                subtract(((size + 255) // 256,), (256,), parameters)
                hbar[:active, column] += coefficients[:active]
            norm = cp.linalg.norm(work)
            hbar[active, column] = norm
            augment(
                ((size + 255) // 256,),
                (256,),
                (work, norm, np.int32(size), np.uint32((seed + steps + column) % 2**32)),
            )
            for _ in range(2):
                project_b((active,), (256,), (*parameters, norm))
                subtract_b(((size + 255) // 256,), (256,), (*parameters, norm))
            next_norm = cp.linalg.norm(work)
            basis[:, active] = work / cp.where(next_norm > 1e-14, next_norm, 1.0)
        steps += width
        cycles += 1
        values, small, projected = refined_gpu(hbar, shift, count)
        vectors = basis[:, :width] @ small
        vectors /= cp.linalg.norm(vectors, axis=0)[None, :]
        residuals = operator.residual(vectors, values)
        bad = cp.count_nonzero(~cp.isfinite(residuals) | (residuals > tolerance))
        score = cp.max(cp.where(cp.isfinite(residuals), residuals, cp.inf))
        if best is None:
            best = (values, vectors, residuals, bad, score, projected)
        else:
            improved = (bad < best[3]) | ((bad == best[3]) & (score < best[4]))
            best = tuple(
                cp.where(improved, current, previous)
                for current, previous in zip(
                    (values, vectors, residuals, bad, score, projected), best, strict=True
                )
            )
        status = cp.stack((bad == 0, cp.all(projected <= tolerance * cp.maximum(1.0, cp.linalg.norm(hbar)))))
        # Control only: no Hessenberg/basis/vector/residual arrays cross PCIe.
        if cp.asnumpy(status).any():
            break
        mask = best[2] > tolerance
        start = cp.sum(best[1] * mask[None, :], axis=1) + 1e-3 * (
            rng.standard_normal(size) + 1j * rng.standard_normal(size)
        )
    if best is None:
        raise SolverError("GPU resident Arnoldi exhausted its operator budget.")
    return (
        best[0],
        best[1],
        best[2],
        dict(
            steps=steps,
            cycles=cycles,
            application_control_download_bytes=2 * cycles,
            ritz_backend="cusolver-geev-svd",
            projected_residuals=best[5],
        ),
    )
