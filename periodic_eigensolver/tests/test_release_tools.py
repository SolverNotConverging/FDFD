from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from periodic_eigensolver.benchmarks.benchmark_end_to_end import (
    RELEASE_NCV,
    RELEASE_NUM_MODES,
    RELEASE_SIDE,
    _build_problem,
    enforce_release_gate,
)
from periodic_eigensolver.scripts import build_release_wheel as release_module
from periodic_eigensolver.scripts.verify_native_wheel import verify_native_wheel


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


def test_release_builder_verifies_before_moving_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension = "periodic_eigensolver/_cython_kernels.abi3.so"
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        assert check is True
        assert Path(cwd).name == "periodic_eigensolver"
        commands.append(list(command))
        wheel_directory = Path(command[command.index("--wheel-dir") + 1])
        _write_wheel(
            wheel_directory / "periodic_eigensolver-0.2.0-cp312-abi3-any.whl",
            (extension,),
        )

    monkeypatch.setattr(release_module.subprocess, "run", fake_run)
    destination = release_module.build_release_wheel(
        tmp_path / "dist", no_build_isolation=True
    )

    assert destination.is_file()
    assert verify_native_wheel(destination) == extension
    assert "--no-build-isolation" in commands[0]


def test_release_builder_does_not_publish_fallback_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "dist"

    def fake_run(command, *, cwd, check):
        wheel_directory = Path(command[command.index("--wheel-dir") + 1])
        _write_wheel(
            wheel_directory / "periodic_eigensolver-0.2.0-py3-none-any.whl",
            (),
        )

    monkeypatch.setattr(release_module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exactly one"):
        release_module.build_release_wheel(output)
    assert not list(output.glob("*.whl"))


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
