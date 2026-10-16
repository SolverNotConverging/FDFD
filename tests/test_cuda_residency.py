"""Hardware tests for the resident numerical path, independent of field assembly."""

import os

import numpy as np
import pytest
from scipy import sparse

from fdfd.core import cuda, eigs

pytestmark = pytest.mark.skipif(
    os.environ.get("FDFD_TEST_CUDA") != "1", reason="set FDFD_TEST_CUDA=1 on a cuDSS host"
)


@pytest.mark.parametrize("device", ["cuda", "gpu"])
@pytest.mark.parametrize("method", ["arpack", "refined"])
def test_no_cpu_iteration_or_numerical_array_downloads(monkeypatch, device, method):
    import cupy as cp

    from fdfd.core import _gpu_resident_arnoldi as resident

    original_iterate, original_download = resident.iterate_device, cp.asnumpy
    original_eigvals, original_svd = cp.linalg.eigvals, cp.linalg.svd
    downloads, ritz, svds = [], [], []

    def forbid_cpu(*args, **kwargs):
        pytest.fail("CPU eigensolving was called during resident CUDA iteration")

    def download(array, *args, **kwargs):
        downloads.append((array.shape, array.dtype))
        return original_download(array, *args, **kwargs)

    def gpu_eigvals(array, *args, **kwargs):
        ritz.append(array.shape)
        return original_eigvals(array, *args, **kwargs)

    def gpu_svd(array, *args, **kwargs):
        svds.append(array.shape)
        return original_svd(array, *args, **kwargs)

    def iterate(*args, **kwargs):
        with monkeypatch.context() as scoped:
            scoped.setattr(cp, "asnumpy", download)
            scoped.setattr(cp.linalg, "eigvals", gpu_eigvals)
            scoped.setattr(cp.linalg, "svd", gpu_svd)
            result = original_iterate(*args, **kwargs)
        assert all(isinstance(array, cp.ndarray) for array in result[:3])
        return result

    monkeypatch.setattr(resident, "iterate_device", iterate)
    monkeypatch.setattr(eigs, "_arpack", forbid_cpu)
    monkeypatch.setattr(eigs, "backward_errors", forbid_cpu)
    diagonal = np.r_[0.0, 1.0, 1.0, np.arange(2.0, 31.0)]
    A = sparse.diags(diagonal, dtype=complex)
    result = eigs.solve(
        A,
        k=3,
        sigma=0.3,
        ncv=len(diagonal),
        tol=1e-10,
        method=method,
        device=device,
        v0=np.ones(len(diagonal)),
    )
    np.testing.assert_allclose(np.sort(result.values.real), [0.0, 1.0, 1.0], atol=1e-9)
    assert max(result.residuals) < 1e-10
    assert np.linalg.matrix_rank(result.vectors[:, np.argsort(result.values.real)[1:]]) == 2
    assert downloads and all(shape == (2,) and dtype == np.bool_ for shape, dtype in downloads)
    assert ritz and ritz[0] == (len(diagonal), len(diagonal))
    assert svds and svds[0][-2:] == (len(diagonal) + 1, len(diagonal))


def test_private_refined_contract_and_initial_vector_validation():
    from fdfd.core._refined import solve_generalized
    from fdfd.core.errors import ConfigurationError

    A = sparse.diags(np.arange(1.0, 9.0), dtype=complex)
    result = solve_generalized(A, sparse.eye(8), sigma=2.2, num_modes=2, tol=1e-10, ncv=8, device="cuda")
    assert result.kernels == result.device == "cuda"
    assert result.converged and result.step_count == 8
    assert not result.eigenvectors.flags.writeable
    with pytest.raises(ConfigurationError, match="v0"):
        cuda.solve_eigenpairs(A, k=2, sigma=2.2, v0=np.zeros(8))
