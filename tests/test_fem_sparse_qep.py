"""Compare reduced shift-invert actions with an independent full-pencil solve."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg
from scipy.sparse import bmat, csr_matrix, eye

import FEM_Mode_Solver.assembly as standalone
import wavefem.modes as wave_modes


@pytest.mark.parametrize("backend", [standalone, wave_modes])
def test_sparse_qep_elimination_handles_complex_and_singular_mass(backend, monkeypatch):
    rng = np.random.default_rng(41)
    n = 6
    a0 = csr_matrix(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    a1 = csr_matrix(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    a2 = eye(n, format="lil", dtype=complex)
    a2[-1, -1] = 0.0  # Maxwell longitudinal mass blocks can be singular.
    a2 = a2.tocsr()
    system = SimpleNamespace(
        ndofs=n, A0=a0, A1=a1, A2=a2,
        polynomial=lambda value: a0 + value * a1 + value**2 * a2,
    )
    identity = eye(n, dtype=complex)
    zero = csr_matrix((n, n), dtype=complex)
    left = bmat(((zero, identity), (-a0, -a1))).toarray()
    right = bmat(((identity, zero), (zero, a2))).toarray()
    target = 0.7 - 0.2j
    factor_shapes = []
    original_splu = backend.splu
    original_eigs = backend.eigs

    def factor(matrix):
        factor_shapes.append(matrix.shape)
        return original_splu(matrix)

    def iterate(operator, **kwargs):
        # Infer the actual perturbed shift from the companion first row:
        # y_second = x_first + shift * y_first.
        vector = rng.normal(size=2 * n) + 1j * rng.normal(size=2 * n)
        actual = operator @ vector
        shift = (actual[n:] - vector[:n])[0] / actual[0]
        expected = linalg.solve(left - shift * right, right @ vector)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
        return original_eigs(operator, **kwargs)

    monkeypatch.setattr(backend, "splu", factor)
    monkeypatch.setattr(backend, "eigs", iterate)
    if backend is standalone:
        values, vectors, method = standalone.solve_qep_candidates(
            system, target=target, candidate_count=3,
            tolerance=1e-11, dense_linearization_limit=4,
        )
    else:
        solver = SimpleNamespace(dense_linearization_limit=4)
        values, vectors, method = wave_modes.ModeSolver._linearized_candidates(
            solver, system, target, 6, 1e-11,
        )
    assert factor_shapes == [(n, n)]
    assert method == "sparse-shift-invert"
    reference = linalg.eigvals(left, right)
    reference = reference[np.isfinite(reference)]
    nearest = reference[np.argsort(abs(reference - target))[:3]]
    for value in nearest:
        assert np.min(abs(values - value)) < 1e-8
    for value, vector in zip(values, vectors.T):
        terms = [a0 @ vector, value * (a1 @ vector), value**2 * (a2 @ vector)]
        assert np.linalg.norm(sum(terms)) / sum(np.linalg.norm(t) for t in terms) < 1e-8
