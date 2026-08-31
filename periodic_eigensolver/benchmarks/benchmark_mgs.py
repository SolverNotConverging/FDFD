"""Controlled single-thread benchmark for the Arnoldi MGS kernel.

This is deliberately not an ordinary pytest.  Run it on an otherwise idle
machine; ``--enforce`` applies the release performance gates from the project
plan at ``n=100000, ncv=32``.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import median
from time import perf_counter

for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np

from periodic_eigensolver import native_backend_available
from periodic_eigensolver import _numpy_kernels


def _time_kernel(kernel, basis: np.ndarray, template: np.ndarray, column: int, repeats: int) -> float:
    timings: list[float] = []
    hessenberg = np.zeros((column + 2, column + 1), dtype=np.complex128, order="F")
    for _ in range(repeats + 1):
        work = template.copy()
        basis[:, column + 1] = 0.0
        hessenberg.fill(0.0)
        start = perf_counter()
        kernel.arnoldi_step(basis, hessenberg, work, column, 1e-14)
        elapsed = perf_counter() - start
        if timings or elapsed >= 0.0:  # skip the first warm-up below
            timings.append(elapsed)
    return median(timings[1:])


def benchmark(n: int, ncv: int, repeats: int) -> dict[str, float | int]:
    if not native_backend_available():
        raise RuntimeError("The Cython extension must be built for this benchmark.")
    from periodic_eigensolver import _cython_kernels

    rng = np.random.default_rng(20260831)
    values = rng.standard_normal((n, ncv + 1)) + 1j * rng.standard_normal((n, ncv + 1))
    values /= np.linalg.norm(values, axis=0, keepdims=True)
    basis_f = np.array(values, dtype=np.complex128, order="F")
    basis_c = np.array(values, dtype=np.complex128, order="C")
    work = np.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=np.complex128)
    column = ncv - 1

    native = _time_kernel(_cython_kernels, basis_f, work, column, repeats)
    python_fortran = _time_kernel(_numpy_kernels, basis_f, work, column, repeats)
    python_c = _time_kernel(_numpy_kernels, basis_c, work, column, repeats)
    return {
        "n": n,
        "ncv": ncv,
        "repeats": repeats,
        "cython_seconds": native,
        "python_fortran_seconds": python_fortran,
        "python_c_seconds": python_c,
        "cython_speedup_vs_fortran": python_fortran / native,
        "cython_speedup_vs_c": python_c / native,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--ncv", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--enforce", action="store_true")
    arguments = parser.parse_args()
    result = benchmark(arguments.n, arguments.ncv, arguments.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))
    if arguments.enforce:
        if arguments.n != 100_000 or arguments.ncv != 32:
            raise SystemExit("--enforce requires the release dimensions n=100000, ncv=32")
        if result["cython_speedup_vs_fortran"] < 1.20:
            raise SystemExit("native MGS is less than 20% faster than the Fortran-order fallback")
        if result["cython_speedup_vs_c"] < 2.0:
            raise SystemExit("native MGS is less than 2x faster than the C-order fallback")


if __name__ == "__main__":
    main()
