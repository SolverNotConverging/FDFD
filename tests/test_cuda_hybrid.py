"""Hybrid preference and numerical checks independent of CUDA dense Ritz support."""

import os

import numpy as np
import pytest
from scipy import sparse

from fdfd.core import cuda, eigs
from fdfd.core.errors import BackendCapabilityError


def test_auto_prefers_hybrid_and_never_attempts_resident(monkeypatch):
    monkeypatch.setattr(cuda, "available", lambda: True)
    monkeypatch.setattr(cuda, "require_available", lambda: None)

    def forbidden():
        pytest.fail("auto must never probe resident GPU")

    monkeypatch.setattr(cuda, "probe_eigen_available", forbidden)
    monkeypatch.setattr(cuda, "probe_hybrid_available", lambda: None)
    assert eigs.check_eigen_device("auto") == "hybrid"

    def missing():
        raise RuntimeError("hybrid compiler missing")

    monkeypatch.setattr(cuda, "probe_hybrid_available", missing)
    with pytest.warns(RuntimeWarning, match="hybrid compiler missing"):
        assert eigs.check_eigen_device("auto") == "cpu"
    with pytest.raises(BackendCapabilityError, match="hybrid compiler missing"):
        eigs.check_eigen_device("hybrid")
    monkeypatch.setattr(cuda, "probe_eigen_available", lambda: None)
    assert eigs.check_eigen_device("gpu") == eigs.check_eigen_device("cuda") == "cuda"


@pytest.mark.skipif(os.environ.get("FDFD_TEST_CUDA") != "1", reason="Set FDFD_TEST_CUDA=1")
@pytest.mark.parametrize("device", ["hybrid", "auto"])
@pytest.mark.parametrize("method", ["arpack", "refined"])
def test_hybrid_repeated_roots_and_singular_mass_without_cuda_ritz(monkeypatch, device, method):
    import cupy as cp

    from fdfd.core import _gpu_arnoldi as hybrid

    def forbidden(*args, **kwargs):
        pytest.fail("hybrid must use CPU dense Ritz extraction")

    monkeypatch.setattr(cuda, "probe_eigen_available", forbidden)
    monkeypatch.setattr(cp.linalg, "eigvals", forbidden)
    monkeypatch.setattr(cp.linalg, "svd", forbidden)
    original = hybrid.refined_coefficients
    calls = []

    def extract(hbar, *args):
        assert isinstance(hbar, np.ndarray)
        calls.append(hbar.shape)
        return original(hbar, *args)

    monkeypatch.setattr(hybrid, "refined_coefficients", extract)
    roots = np.r_[0.0, 1.0, 1.0, np.arange(2.0, 30.0)]
    mass = np.ones(roots.size)
    mass[-2:] = 0.0
    result = eigs.solve(
        sparse.diags(roots),
        sparse.diags(mass),
        k=3,
        sigma=0.3,
        ncv=roots.size,
        tol=1e-10,
        method=method,
        device=device,
    )
    order = np.argsort(result.values.real)
    np.testing.assert_allclose(result.values[order], [0.0, 1.0, 1.0], atol=1e-9)
    assert max(result.residuals) < 1e-10
    assert np.linalg.matrix_rank(result.vectors[:, order[1:]], tol=1e-5) == 2
    assert any(shape[1] > 2 for shape in calls)


@pytest.mark.skipif(os.environ.get("FDFD_TEST_CUDA") != "1", reason="Set FDFD_TEST_CUDA=1")
def test_private_hybrid_refined_contract_and_invalid_initial_vector():
    from fdfd.core._refined import solve_generalized
    from fdfd.core.errors import ConfigurationError

    A = sparse.diags(np.arange(1.0, 9.0), dtype=complex)
    result = solve_generalized(A, sparse.eye(8), sigma=2.2, num_modes=2, tol=1e-10, ncv=8, device="hybrid")
    assert result.device == "hybrid" and result.kernels == "cuda"
    assert result.converged and result.step_count == 8
    assert not result.eigenvectors.flags.writeable
    with pytest.raises(ConfigurationError, match="v0"):
        cuda.solve_eigenpairs(A, k=2, sigma=2.2, v0=np.zeros(8), device="hybrid")
