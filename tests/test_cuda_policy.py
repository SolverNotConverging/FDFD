"""Backend selection must never hide an unavailable CUDA numerical core."""

import pytest

from fdfd.core import cuda, eigs
from fdfd.core.errors import BackendCapabilityError


def unavailable():
    raise RuntimeError("missing dependency")


def test_cpu_does_not_probe_cuda(monkeypatch):
    monkeypatch.setattr(cuda, "available", unavailable)
    monkeypatch.setattr(cuda, "require_available", unavailable)
    monkeypatch.setattr(cuda, "probe_eigen_available", unavailable)
    assert eigs.check_eigen_device("cpu") == "cpu"


@pytest.mark.parametrize("resolve", [eigs.check_device, eigs.check_eigen_device])
def test_missing_runtime_warns_auto_and_rejects_cuda(monkeypatch, resolve):
    monkeypatch.setattr(cuda, "available", lambda: False)
    monkeypatch.setattr(cuda, "require_available", unavailable)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        assert resolve("auto") == "cpu"
    with pytest.raises(BackendCapabilityError, match="missing dependency"):
        resolve("cuda")


def test_missing_resident_core_never_selects_hybrid_cuda(monkeypatch):
    monkeypatch.setattr(cuda, "available", lambda: True)
    monkeypatch.setattr(cuda, "require_available", lambda: None)
    monkeypatch.setattr(cuda, "probe_eigen_available", unavailable)
    monkeypatch.setattr(cuda, "probe_hybrid_available", lambda: None)
    assert eigs.check_eigen_device("auto") == "hybrid"
    with pytest.raises(BackendCapabilityError, match="resident CUDA eigensolver"):
        eigs.check_eigen_device("cuda")
    assert eigs.check_device("cuda") == "cuda"  # Direct solves do not need Ritz extraction.
