from __future__ import annotations

from dataclasses import FrozenInstanceError
from inspect import Parameter, signature
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import subspace_angles
from scipy.sparse import csc_matrix, diags, eye

from periodic_eigensolver.refined import (
    ArnoldiResult,
    native_backend_available,
    resolve_backend,
    resolve_kernel_backend,
    solve_generalized,
)
from periodic_eigensolver.refined import (
    _arnoldi_factorization,
    _refined_candidate_data,
)


def available_backends() -> tuple[str, ...]:
    if native_backend_available():
        return ("numpy", "cython")
    return ("numpy",)


def test_solve_generalized_has_stable_keyword_only_contract() -> None:
    parameters = signature(solve_generalized).parameters
    assert list(parameters) == [
        "A",
        "B",
        "sigma",
        "num_modes",
        "tol",
        "ncv",
        "max_restarts",
        "random_seed",
        "backend",
    ]
    assert parameters["A"].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["B"].kind is Parameter.POSITIONAL_OR_KEYWORD
    for name in list(parameters)[2:]:
        assert parameters[name].kind is Parameter.KEYWORD_ONLY
    assert parameters["tol"].default == 1e-10
    assert parameters["ncv"].default is None
    assert parameters["max_restarts"].default == 12
    assert parameters["random_seed"].default == 0
    assert parameters["backend"].default == "auto"


@pytest.mark.parametrize("backend", available_backends())
def test_arnoldi_retains_rectangular_paper_relation(backend: str) -> None:
    rng = np.random.default_rng(20260831)
    size = 18
    steps = 7
    operator = rng.standard_normal((size, size)) + 1j * rng.standard_normal(
        (size, size)
    )
    operator /= np.linalg.norm(operator)
    initial = rng.standard_normal(size) + 1j * rng.standard_normal(size)

    basis, hbar = _arnoldi_factorization(
        lambda vector: operator @ vector,
        size,
        steps,
        initial,
        kernel_backend=backend,
    )

    assert basis.shape == (size, steps + 1)
    assert hbar.shape == (steps + 1, steps)
    assert basis.flags.f_contiguous
    assert hbar.flags.f_contiguous
    np.testing.assert_allclose(
        operator @ basis[:, :steps],
        basis @ hbar,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        basis.conj().T @ basis,
        np.eye(steps + 1),
        rtol=2e-12,
        atol=2e-12,
    )


class TrackingMatrix:
    def __init__(self, values: np.ndarray):
        self.values = values
        self.shape = values.shape
        self.multiplied_shapes: list[tuple[int, ...]] = []

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        self.multiplied_shapes.append(other.shape)
        return self.values @ other


def test_refinement_uses_small_hessenberg_svd_and_batched_pencil_residual() -> None:
    rng = np.random.default_rng(17)
    size = 14
    steps = 8
    mode_count = 2
    sigma = 0.35 + 0.1j
    diagonal = np.linspace(1.0, 4.0, size) - 0.03j * np.arange(size)
    a_values = np.diag(diagonal.astype(np.complex128))
    b_values = np.eye(size, dtype=np.complex128)
    shift_invert = np.diag(1.0 / (diagonal - sigma))
    initial = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    basis, hbar = _arnoldi_factorization(
        lambda vector: shift_invert @ vector,
        size,
        steps,
        initial,
        kernel_backend="numpy",
    )
    tracked_a = TrackingMatrix(a_values)
    tracked_b = TrackingMatrix(b_values)

    from periodic_eigensolver import refined as refined_module

    with patch.object(
        refined_module.linalg,
        "svd",
        wraps=refined_module.linalg.svd,
    ) as svd_mock:
        eigenvalues, eigenvectors, residuals, projected_residuals = (
            _refined_candidate_data(
                tracked_a,
                tracked_b,
                basis,
                hbar,
                sigma,
                mode_count,
                kernel_backend="numpy",
            )
        )

    small_residuals = [call.args[0] for call in svd_mock.call_args_list]
    assert [matrix.shape for matrix in small_residuals] == [
        (steps + 1, steps),
        (steps + 1, steps),
    ]
    expected_projected = [
        np.linalg.svd(matrix, compute_uv=False)[-1]
        for matrix in small_residuals
    ]
    np.testing.assert_allclose(projected_residuals, expected_projected)
    assert tracked_a.multiplied_shapes == [(size, mode_count)]
    assert tracked_b.multiplied_shapes == [(size, mode_count)]
    assert eigenvectors.shape == (size, mode_count)
    for index, eigenvalue in enumerate(eigenvalues):
        ax = a_values @ eigenvectors[:, index]
        bx = b_values @ eigenvectors[:, index]
        expected = np.linalg.norm(ax - eigenvalue * bx) / (
            np.linalg.norm(ax) + abs(eigenvalue) * np.linalg.norm(bx)
        )
        assert residuals[index] == pytest.approx(expected, rel=1e-13, abs=1e-15)


def _diagonal_projected_problem(eigenvalues: np.ndarray):
    sigma = 0.5
    arnoldi_size = eigenvalues.size
    physical_size = arnoldi_size + 1
    hbar = np.zeros(
        (arnoldi_size + 1, arnoldi_size), dtype=np.complex128
    )
    hbar[:arnoldi_size, :] = np.diag(1.0 / (eigenvalues - sigma))
    basis = np.eye(physical_size, dtype=np.complex128)
    matrix_a = diags(np.append(eigenvalues, 5.0), format="csr")
    matrix_b = eye(physical_size, format="csr")
    return matrix_a, matrix_b, basis, hbar, sigma


@pytest.mark.parametrize("backend", available_backends())
def test_refinement_retains_exact_eigenvalue_multiplicity(backend: str) -> None:
    problem = _diagonal_projected_problem(np.array([1.0, 1.0, 2.0, 3.0]))
    values, vectors, physical, projected = _refined_candidate_data(
        *problem,
        num_modes=2,
        kernel_backend=backend,
    )

    np.testing.assert_allclose(values, [1.0, 1.0], atol=1e-14)
    assert np.linalg.matrix_rank(vectors, tol=1e-12) == 2
    assert abs(np.vdot(vectors[:, 0], vectors[:, 1])) <= 1e-12
    assert np.max(physical) <= 1e-14
    assert np.max(projected) <= 1e-14


@pytest.mark.parametrize("backend", available_backends())
def test_clustered_refinement_recovers_invariant_subspace(backend: str) -> None:
    eigenvalues = np.array([1.0, 1.0 + 5e-9, 2.0, 3.0])
    problem = _diagonal_projected_problem(eigenvalues)
    values, vectors, physical, _projected = _refined_candidate_data(
        *problem,
        num_modes=2,
        kernel_backend=backend,
    )
    target = np.eye(vectors.shape[0], dtype=np.complex128)[:, :2]
    angles = subspace_angles(target, vectors)

    np.testing.assert_allclose(
        np.sort(values.real), eigenvalues[:2], rtol=0.0, atol=1e-12
    )
    assert float(np.max(angles)) <= 1e-7
    assert np.max(physical) <= 1e-12


@pytest.mark.parametrize("backend", available_backends())
@pytest.mark.parametrize(
    ("separation", "off_diagonal"),
    [(1.0e-5, 1.0), (5.0e-9, 0.1)],
)
def test_nonnormal_nearby_roots_keep_legitimate_parallel_eigenvectors(
    backend: str, separation: float, off_diagonal: float
) -> None:
    size = 30
    diagonal = np.r_[1.0, 1.0 + separation, np.arange(2.0, 30.0)]
    matrix = np.diag(diagonal).astype(np.complex128)
    matrix[0, 1] = off_diagonal

    result = solve_generalized(
        csc_matrix(matrix),
        eye(size, format="csc"),
        sigma=0.8,
        num_modes=2,
        tol=1e-9,
        ncv=20,
        max_restarts=2,
        backend="python" if backend == "numpy" else backend,
    )

    assert result.converged
    assert np.max(np.abs(result.eigenvalues - 1.0)) < 1e-4
    assert np.max(result.physical_residuals) <= 1e-9
    assert abs(np.vdot(result.eigenvectors[:, 0], result.eigenvectors[:, 1])) > 0.99


@pytest.mark.parametrize("backend", available_backends())
def test_diagonal_pencil_returns_nearest_modes_and_physical_residuals(
    backend: str,
) -> None:
    matrix_a = diags(np.arange(1.0, 9.0), format="csr")
    matrix_b = eye(8, format="csr")

    result = solve_generalized(
        matrix_a,
        matrix_b,
        sigma=1.2,
        num_modes=2,
        tol=1e-8,
        ncv=7,
        max_restarts=4,
        random_seed=1,
        backend="python" if backend == "numpy" else backend,
    )
    values, vectors, residuals, restarts = result.eigenvalues, result.eigenvectors, result.physical_residuals, result.restart_count

    np.testing.assert_allclose(values, [1.0, 2.0], rtol=0.0, atol=1e-8)
    assert np.max(residuals) <= 1e-8
    assert 0 <= restarts <= 4
    np.testing.assert_allclose(
        vectors.conj().T @ vectors,
        np.eye(2),
        rtol=1e-9,
        atol=1e-9,
    )


def test_public_result_is_immutable_and_reports_total_work() -> None:
    result = solve_generalized(
        diags(np.arange(1.0, 9.0), format="csr"),
        eye(8, format="csr"),
        sigma=1.2,
        num_modes=2,
        tol=1e-8,
        ncv=7,
        max_restarts=3,
        random_seed=1,
        backend="python",
    )

    assert isinstance(result, ArnoldiResult)
    assert result.converged
    assert 0 <= result.restart_count <= 3
    assert result.step_count == (result.restart_count + 1) * 7
    assert result.backend == result.resolved_backend == "python"
    assert result.residuals is result.physical_residuals
    assert result.restarts == result.restart_count
    assert result.steps == result.step_count
    assert result.eigenvalues.shape == (2,)
    assert result.eigenvectors.shape == (8, 2)
    assert result.physical_residuals.shape == (2,)
    assert result.projected_residuals.shape == (2,)
    assert np.all(result.projected_residuals >= 0.0)

    for array in (
        result.eigenvalues,
        result.eigenvectors,
        result.physical_residuals,
        result.projected_residuals,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        result.eigenvalues[0] = 99.0
    with pytest.raises(FrozenInstanceError):
        result.converged = False


def test_singular_b_supports_finite_generalized_modes() -> None:
    matrix_a = diags([1.0, 2.0, 3.0, 4.0], format="csr")
    matrix_b = diags([1.0, 1.0, 0.0, 0.0], format="csr")

    result = solve_generalized(
        matrix_a,
        matrix_b,
        sigma=0.5,
        num_modes=2,
        tol=1e-10,
        ncv=3,
        max_restarts=4,
        random_seed=3,
        backend="python",
    )

    assert result.converged
    np.testing.assert_allclose(result.eigenvalues, [1.0, 2.0], atol=1e-10)
    assert np.max(result.physical_residuals) <= 1e-10


def test_bad_shift_has_actionable_factorization_error() -> None:
    with pytest.raises(ValueError, match="choose a shift away"):
        solve_generalized(
            diags([1.0, 2.0, 3.0, 4.0], format="csr"),
            eye(4, format="csr"),
            sigma=2.0,
            num_modes=1,
            backend="python",
        )


def test_happy_breakdown_continues_to_recover_multiplicity() -> None:
    matrix_a = diags([2.0] * 6, format="csr")
    matrix_b = eye(6, format="csr")
    result = solve_generalized(
        matrix_a,
        matrix_b,
        sigma=0.0,
        num_modes=1,
        tol=1e-12,
        ncv=4,
        backend="python",
    )

    assert result.converged
    assert result.step_count == 1
    np.testing.assert_allclose(result.eigenvalues, [2.0], atol=1e-12)

    repeated = solve_generalized(
        matrix_a,
        matrix_b,
        sigma=0.0,
        num_modes=2,
        tol=1e-12,
        ncv=4,
        backend="python",
    )
    assert repeated.converged
    assert repeated.step_count == 4
    np.testing.assert_allclose(repeated.eigenvalues, [2.0, 2.0], atol=1e-12)
    assert np.linalg.matrix_rank(repeated.eigenvectors, tol=1e-12) == 2

    with pytest.raises(RuntimeError, match="finite Ritz"):
        solve_generalized(
            eye(6, format="csr"),
            diags([0.0] * 6, format="csr"),
            sigma=0.0,
            num_modes=2,
            tol=1e-12,
            ncv=4,
            backend="python",
        )


@pytest.mark.parametrize("backend", available_backends())
def test_public_solver_recovers_exact_repeated_invariant_subspace(
    backend: str,
) -> None:
    matrix_a = diags(
        [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], format="csr"
    )
    result = solve_generalized(
        matrix_a,
        eye(8, format="csr"),
        sigma=0.8,
        num_modes=2,
        tol=1e-10,
        max_restarts=2,
        random_seed=5,
        backend="python" if backend == "numpy" else backend,
    )
    target = np.eye(8, dtype=np.complex128)[:, :2]

    assert result.converged
    np.testing.assert_allclose(result.eigenvalues, [1.0, 1.0], atol=1e-10)
    assert float(np.max(subspace_angles(target, result.eigenvectors))) <= 1e-7


def test_exhaustion_returns_deterministic_best_candidates_and_total_steps() -> None:
    arguments = dict(
        sigma=1.2,
        num_modes=2,
        tol=0.0,
        ncv=4,
        random_seed=3,
        backend="python",
    )
    matrix_a = diags(np.arange(1.0, 9.0), format="csr")
    matrix_b = eye(8, format="csr")
    first_cycle = solve_generalized(
        matrix_a, matrix_b, max_restarts=0, **arguments
    )
    exhausted = solve_generalized(
        matrix_a, matrix_b, max_restarts=2, **arguments
    )
    repeated = solve_generalized(
        matrix_a, matrix_b, max_restarts=2, **arguments
    )

    assert not exhausted.converged
    assert exhausted.restart_count == 2
    assert exhausted.step_count == 3 * 4
    assert np.max(exhausted.physical_residuals) <= np.max(
        first_cycle.physical_residuals
    )
    np.testing.assert_array_equal(exhausted.eigenvalues, repeated.eigenvalues)
    np.testing.assert_array_equal(
        exhausted.eigenvectors, repeated.eigenvectors
    )
    np.testing.assert_array_equal(
        exhausted.physical_residuals, repeated.physical_residuals
    )
    np.testing.assert_array_equal(
        exhausted.projected_residuals, repeated.projected_residuals
    )


def test_restart_uses_best_unconverged_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from periodic_eigensolver import refined as refined_module

    size = 8
    vectors = np.zeros((size, 2), dtype=np.complex128)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    candidate_sets = [
        (
            np.array([1.0, 2.0], dtype=np.complex128),
            vectors.copy(),
            np.array([0.4, 0.01]),
            np.array([0.3, 0.005]),
        ),
        (
            np.array([1.1, 2.1], dtype=np.complex128),
            np.roll(vectors, 2, axis=0),
            np.array([0.8, 0.02]),
            np.array([0.6, 0.01]),
        ),
        (
            np.array([1.2, 2.2], dtype=np.complex128),
            np.roll(vectors, 4, axis=0),
            np.array([0.7, 0.03]),
            np.array([0.5, 0.02]),
        ),
    ]
    monkeypatch.setattr(
        refined_module,
        "_refined_candidate_data",
        lambda *_args, **_kwargs: candidate_sets.pop(0),
    )
    original_restart = refined_module._restart_vector
    restart_inputs: list[np.ndarray] = []

    def capture_restart(eigenvectors, rng, n):
        restart_inputs.append(eigenvectors.copy())
        return original_restart(eigenvectors, rng, n)

    monkeypatch.setattr(refined_module, "_restart_vector", capture_restart)
    result = solve_generalized(
        diags(np.arange(1.0, size + 1.0), format="csr"),
        eye(size, format="csr"),
        sigma=0.5,
        num_modes=2,
        tol=0.1,
        ncv=4,
        max_restarts=2,
        random_seed=7,
        backend="python",
    )

    assert not result.converged
    assert len(restart_inputs) == 2
    np.testing.assert_array_equal(restart_inputs[0], vectors[:, :1])
    np.testing.assert_array_equal(restart_inputs[1], vectors[:, :1])
    np.testing.assert_array_equal(result.eigenvalues, [1.0, 2.0])


def test_backend_resolution_is_explicit() -> None:
    assert resolve_backend("python") == "python"
    assert resolve_backend("auto") in {"python", "cython"}
    assert resolve_kernel_backend("numpy") == "numpy"
    assert resolve_kernel_backend("python") == "numpy"
    assert resolve_kernel_backend("auto") in {"numpy", "cython"}
    with pytest.raises(ValueError, match="backend"):
        resolve_backend("numpy")
    with pytest.raises(ValueError, match="kernel_backend"):
        resolve_kernel_backend("unknown")


def test_auto_backend_falls_back_only_when_native_module_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from periodic_eigensolver import refined as refined_module

    monkeypatch.setattr(refined_module, "_cython_kernels", None)
    assert refined_module.resolve_kernel_backend("auto") == "numpy"
    assert refined_module.resolve_backend("auto") == "python"
    with pytest.raises(ImportError, match="not built"):
        refined_module.resolve_kernel_backend("cython")
    with pytest.raises(ImportError, match="not built"):
        solve_generalized(
            diags(np.arange(1.0, 5.0), format="csr"),
            eye(4, format="csr"),
            sigma=0.5,
            num_modes=1,
            backend="cython",
        )


@pytest.mark.parametrize("backend", available_backends())
def test_happy_breakdown_keeps_rectangular_hessenberg(backend: str) -> None:
    size = 9
    initial = np.arange(1, size + 1, dtype=np.complex128)
    basis, hbar = _arnoldi_factorization(
        lambda vector: vector.copy(),
        size,
        5,
        initial,
        kernel_backend=backend,
    )

    assert basis.shape == (size, 2)
    assert hbar.shape == (2, 1)
    assert abs(hbar[1, 0]) <= 1e-14
    np.testing.assert_allclose(
        basis[:, :1],
        basis @ hbar,
        rtol=2e-14,
        atol=2e-14,
    )


@pytest.mark.skipif(
    not native_backend_available(), reason="optional Cython extension is not built"
)
def test_cython_and_numpy_backends_agree() -> None:
    matrix_a = diags(np.arange(1.0, 9.0), format="csr")
    matrix_b = eye(8, format="csr")
    arguments = dict(
        sigma=1.2,
        num_modes=2,
        tol=1e-8,
        ncv=7,
        max_restarts=2,
        random_seed=4,
    )

    numpy_result = solve_generalized(
        matrix_a, matrix_b, backend="python", **arguments
    )
    cython_result = solve_generalized(
        matrix_a, matrix_b, backend="cython", **arguments
    )

    np.testing.assert_allclose(cython_result.eigenvalues, numpy_result.eigenvalues, atol=1e-11)
    np.testing.assert_allclose(cython_result.physical_residuals, numpy_result.physical_residuals, atol=1e-11)
    assert cython_result.restart_count == numpy_result.restart_count
