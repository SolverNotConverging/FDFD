"""Controlled end-to-end shift-invert Arnoldi performance check.

This benchmark includes CSC construction, SuperLU factorization, sparse
products, Arnoldi, and refined extraction.  It complements ``benchmark_mgs``:
LU dominance is reported explicitly, while ``--enforce`` still rejects a
median paired Cython/Python runtime ratio above 1.05.
"""

from __future__ import annotations

import argparse
import gc
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
from scipy.linalg import subspace_angles
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

from periodic_eigensolver import native_backend_available, solve_generalized


RELEASE_SIDE = 128
RELEASE_NCV = 16
RELEASE_NUM_MODES = 2
MIN_ENFORCED_REPEATS = 5
MAX_RUNTIME_RATIO = 1.05
LU_DOMINATED_FRACTION = 0.50


def _build_problem(side: int) -> tuple[csc_matrix, csc_matrix, complex]:
    """Return a deterministic complex five-point finite-difference pencil."""
    side = int(side)
    if side < 4:
        raise ValueError("side must be at least 4.")
    one_dimensional = diags(
        (-1.0, 2.0, -1.0),
        (-1, 0, 1),
        shape=(side, side),
        format="csc",
        dtype=np.complex128,
    )
    identity = eye(side, dtype=np.complex128, format="csc")
    laplacian = kron(identity, one_dimensional, format="csc") + kron(
        one_dimensional, identity, format="csc"
    )

    size = side * side
    indices = np.arange(size, dtype=np.float64)
    # Break grid symmetries and keep the target factorization away from a pole.
    diagonal_perturbation = (
        1.0e-3 * np.sin(indices * 0.6180339887498948) + 2.0e-2j
    )
    matrix_a = csc_matrix(
        laplacian + diags(diagonal_perturbation, format="csc"),
        dtype=np.complex128,
    )
    matrix_b = eye(size, dtype=np.complex128, format="csc")
    sigma = 0.15 + 0.01j
    return matrix_a, matrix_b, sigma


def _solve_once(
    matrix_a: csc_matrix,
    matrix_b: csc_matrix,
    sigma: complex,
    *,
    num_modes: int,
    ncv: int,
    backend: str,
):
    return solve_generalized(
        matrix_a,
        matrix_b,
        sigma=sigma,
        num_modes=num_modes,
        tol=0.0,
        ncv=ncv,
        max_restarts=0,
        random_seed=20260831,
        backend=backend,
    )


def _time_factorization(
    shifted: csc_matrix, repeats: int
) -> tuple[float, list[float]]:
    timings: list[float] = []
    for attempt in range(repeats + 1):
        gc.collect()
        started = perf_counter()
        splu(shifted)
        elapsed = perf_counter() - started
        if attempt:
            timings.append(elapsed)
    return median(timings), timings


def benchmark(
    side: int,
    ncv: int,
    num_modes: int,
    repeats: int,
) -> dict[str, float | int | bool | str]:
    """Measure paired Python/Cython solves and return JSON-ready metrics."""
    if not native_backend_available():
        raise RuntimeError("The Cython extension must be built for this benchmark.")
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be positive.")

    matrix_a, matrix_b, sigma = _build_problem(side)
    shifted = (matrix_a - sigma * matrix_b).tocsc()
    lu_seconds, _lu_samples = _time_factorization(shifted, repeats)

    # Warm both paths before paired timings. The alternating order balances
    # cache and slow thermal drift effects on a controlled runner.
    last_results = {
        backend: _solve_once(
            matrix_a,
            matrix_b,
            sigma,
            num_modes=num_modes,
            ncv=ncv,
            backend=backend,
        )
        for backend in ("python", "cython")
    }
    timings: dict[str, list[float]] = {"python": [], "cython": []}
    paired_ratios: list[float] = []
    for repeat in range(repeats):
        order = ("python", "cython") if repeat % 2 == 0 else ("cython", "python")
        pair: dict[str, float] = {}
        for backend in order:
            gc.collect()
            started = perf_counter()
            last_results[backend] = _solve_once(
                matrix_a,
                matrix_b,
                sigma,
                num_modes=num_modes,
                ncv=ncv,
                backend=backend,
            )
            elapsed = perf_counter() - started
            timings[backend].append(elapsed)
            pair[backend] = elapsed
        paired_ratios.append(pair["cython"] / pair["python"])

    python_seconds = median(timings["python"])
    cython_seconds = median(timings["cython"])
    runtime_ratio = median(paired_ratios)
    python_result = last_results["python"]
    cython_result = last_results["cython"]
    eigenvalue_error = max(
        max(
            np.min(np.abs(cython_result.eigenvalues - value))
            for value in python_result.eigenvalues
        ),
        max(
            np.min(np.abs(python_result.eigenvalues - value))
            for value in cython_result.eigenvalues
        ),
    )
    largest_subspace_angle = float(
        np.max(
            subspace_angles(
                python_result.eigenvectors,
                cython_result.eigenvectors,
            )
        )
    )
    lu_fraction = lu_seconds / python_seconds
    lu_dominated = bool(lu_fraction >= LU_DOMINATED_FRACTION)

    return {
        "n": int(matrix_a.shape[0]),
        "side": int(side),
        "ncv": int(ncv),
        "num_modes": int(num_modes),
        "repeats": repeats,
        "python_seconds": python_seconds,
        "cython_seconds": cython_seconds,
        "cython_to_python_ratio": runtime_ratio,
        "cython_regression_percent": 100.0 * (runtime_ratio - 1.0),
        "lu_factor_seconds": lu_seconds,
        "lu_fraction_of_python_solve": lu_fraction,
        "lu_dominated": lu_dominated,
        "classification": "lu-dominated" if lu_dominated else "backend-sensitive",
        "eigenvalue_max_matching_error": float(eigenvalue_error),
        "max_subspace_angle_radians": largest_subspace_angle,
    }


def enforce_release_gate(
    result: dict[str, float | int | bool | str],
    *,
    side: int,
    ncv: int,
    num_modes: int,
    repeats: int,
) -> None:
    """Raise ``RuntimeError`` when controlled release requirements fail."""
    if (side, ncv, num_modes) != (
        RELEASE_SIDE,
        RELEASE_NCV,
        RELEASE_NUM_MODES,
    ):
        raise RuntimeError(
            "--enforce requires side=128, ncv=16, and num_modes=2."
        )
    if repeats < MIN_ENFORCED_REPEATS:
        raise RuntimeError("--enforce requires at least 5 paired repeats.")
    if float(result["eigenvalue_max_matching_error"]) > 1.0e-9:
        raise RuntimeError("Python/Cython eigenvalues differ by more than 1e-9.")
    if float(result["max_subspace_angle_radians"]) > 1.0e-7:
        raise RuntimeError(
            "Python/Cython invariant subspaces differ by more than 1e-7."
        )
    if float(result["cython_to_python_ratio"]) > MAX_RUNTIME_RATIO:
        classification = result["classification"]
        raise RuntimeError(
            "end-to-end Cython solve regressed by more than 5% "
            f"(classification={classification}, "
            f"ratio={float(result['cython_to_python_ratio']):.6f})."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=RELEASE_SIDE)
    parser.add_argument("--ncv", type=int, default=RELEASE_NCV)
    parser.add_argument("--num-modes", type=int, default=RELEASE_NUM_MODES)
    parser.add_argument("--repeats", type=int, default=MIN_ENFORCED_REPEATS)
    parser.add_argument("--enforce", action="store_true")
    arguments = parser.parse_args()
    try:
        result = benchmark(
            arguments.side,
            arguments.ncv,
            arguments.num_modes,
            arguments.repeats,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if arguments.enforce:
            enforce_release_gate(
                result,
                side=arguments.side,
                ncv=arguments.ncv,
                num_modes=arguments.num_modes,
                repeats=arguments.repeats,
            )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
