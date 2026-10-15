"""Result files: atomic writes, one shared header, and one loader."""

import h5py
import numpy as np
import pytest

import fdfd
from fdfd.core.persistence import FORMAT, SCHEMA, atomic_h5, read_value, write_value


def test_failed_write_preserves_previous_file_and_removes_temporary(tmp_path):
    path = tmp_path / "result.h5"
    with h5py.File(path, "w") as archive:
        archive.create_dataset("important", data=[1, 2, 3])
    original = path.read_bytes()
    with pytest.raises(fdfd.PersistenceError), atomic_h5(path) as archive:
        write_value(archive, "unsupported", lambda: None)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_mappings_keep_insertion_order_beyond_ten_entries(tmp_path):
    data = {f"key{i}": i for i in range(12)}
    with atomic_h5(tmp_path / "map.h5") as handle:
        write_value(handle, "value", data)
    with h5py.File(tmp_path / "map.h5", "r") as handle:
        assert list(read_value(handle["value"], {})) == list(data)


def test_every_result_file_has_the_same_header(tmp_path):
    solver = fdfd.ModeSolver1D(frequency=10e9, x_range=(0.0, 0.03))
    solver.mesh(resolution=30)
    path = solver.solve(num_modes=1).save(tmp_path / "modes.h5")
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
    assert attrs["format"] == FORMAT and attrs["schema"] == SCHEMA
    assert attrs["solver_family"] == "waveguide"
    assert attrs["time_convention"] == fdfd.TIME_CONVENTION
    assert attrs["units"] == "SI"
    assert attrs["fdfd_version"] == fdfd.__version__


def test_load_rejects_foreign_or_legacy_files(tmp_path):
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs.update(format="cem-fdfd-results", schema="1.0", solver_family="fdfd_waveguide_modes")
    with pytest.raises(fdfd.PersistenceError, match=r"1.x"):
        fdfd.load(path)
    with pytest.raises(fdfd.PersistenceError):
        fdfd.load(tmp_path / "missing.h5")


def test_round_trip_preserves_fields_and_info(tmp_path):
    solver = fdfd.ModeSolver2D(frequency=30e9, x_range=(0.0, 0.02), y_range=(0.0, 0.01))
    solver.add_rectangle(
        x_range=(0.005, 0.015), y_range=(0.002, 0.008), material=fdfd.Material(name="d", epsilon=4)
    )
    solver.mesh(resolution=(16, 8))
    result = solver.solve(num_modes=2)
    restored = fdfd.load(result.save(tmp_path / "modes.h5"))
    assert isinstance(restored, fdfd.ModeResult)
    np.testing.assert_array_equal(restored.neff, result.neff)
    for name, values in result.fields.items():
        np.testing.assert_array_equal(restored.fields[name], values)
    assert (restored.info.device, restored.info.method, restored.info.guess) == (
        result.info.device,
        result.info.method,
        result.info.guess,
    )
    np.testing.assert_array_equal(restored.info.residuals, result.info.residuals)
    assert restored.grid == result.grid
    np.testing.assert_allclose(restored.beta, result.beta)
