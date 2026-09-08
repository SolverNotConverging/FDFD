from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from benchmarks.periodic_eigensolver.benchmark_end_to_end import (
    RELEASE_NCV,
    RELEASE_NUM_MODES,
    RELEASE_SIDE,
    _build_problem,
    enforce_release_gate,
)
from libraries.periodic_eigensolver.scripts.verify_native_wheel import verify_native_wheel


def _write_wheel(path: Path, native_members: tuple[str, ...]) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("periodic_eigensolver/__init__.py", "")
        for member in native_members:
            archive.writestr(member, b"native-placeholder")


def test_native_wheel_contract_requires_exactly_one_extension(tmp_path: Path) -> None:
    extension = "periodic_eigensolver/_cython_kernels.cp312-win_amd64.pyd"
    valid = tmp_path / "valid.whl"
    missing = tmp_path / "missing.whl"
    duplicated = tmp_path / "duplicated.whl"
    _write_wheel(valid, (extension,))
    _write_wheel(missing, ())
    _write_wheel(
        duplicated,
        (extension, "periodic_eigensolver/_cython_kernels.abi3.so"),
    )

    assert verify_native_wheel(valid) == extension
    with pytest.raises(RuntimeError, match="exactly one"):
        verify_native_wheel(missing)
    with pytest.raises(RuntimeError, match="exactly one"):
        verify_native_wheel(duplicated)


def test_end_to_end_gate_enforces_five_percent_limit() -> None:
    result = {
        "cython_to_python_ratio": 1.05,
        "classification": "lu-dominated",
        "eigenvalue_max_matching_error": 1.0e-10,
        "max_subspace_angle_radians": 1.0e-8,
    }
    enforce_release_gate(
        result,
        side=RELEASE_SIDE,
        ncv=RELEASE_NCV,
        num_modes=RELEASE_NUM_MODES,
        repeats=5,
    )
    result["cython_to_python_ratio"] = 1.050001
    with pytest.raises(RuntimeError, match="more than 5%"):
        enforce_release_gate(
            result,
            side=RELEASE_SIDE,
            ncv=RELEASE_NCV,
            num_modes=RELEASE_NUM_MODES,
            repeats=5,
        )


def test_end_to_end_fixture_is_complex_sparse_pencil() -> None:
    matrix_a, matrix_b, sigma = _build_problem(4)
    assert matrix_a.shape == matrix_b.shape == (16, 16)
    assert matrix_a.format == matrix_b.format == "csc"
    assert matrix_a.dtype.name == matrix_b.dtype.name == "complex128"
    assert isinstance(sigma, complex)
