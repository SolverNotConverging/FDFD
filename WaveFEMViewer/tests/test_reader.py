from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from wavefem_viewer import load_h5


def write_result_file(path, *, include_scene: bool = True, frequency: float = 2.0e9) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "wavefem"
        handle.attrs["schema_version"] = 1
        handle.attrs["kind"] = "single"
        handle.attrs["result_count"] = 1
        handle.create_dataset("frequencies_hz", data=np.asarray((frequency,)))
        result = handle.create_group("results/000000")
        result.attrs["index"] = 0
        if np.isfinite(frequency):
            result.attrs["frequency_hz"] = frequency
        result.attrs["ky"] = 0.0
        result.attrs["metadata_json"] = json.dumps({"ndofs": 12})
        coordinates = np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 2.0)))
        result.create_dataset("coordinates", data=coordinates)
        fields = result.create_group("fields")
        incident = np.arange(9, dtype=float).reshape(3, 3).astype(np.complex128)
        scattered = np.full((3, 3), 0.25j, dtype=np.complex128)
        for name, value in (
            ("E_incident", incident),
            ("E_scattered", scattered),
            ("E_total", incident + scattered),
            ("H_incident", incident / 377.0),
            ("H_scattered", scattered / 377.0),
            ("H_total", (incident + scattered) / 377.0),
        ):
            fields.create_dataset(name, data=value)
        s_group = result.create_group("s_parameters")
        s_group.create_dataset("side", data=np.asarray((b"left", b"right"), dtype="S5"))
        s_group.create_dataset("out_mode", data=np.asarray((0, 0), dtype=np.int64))
        s_group.create_dataset("in_mode", data=np.asarray((0, 0), dtype=np.int64))
        s_group.create_dataset("value", data=np.asarray((0.1j, 0.9), dtype=np.complex128))
        powers = result.create_group("powers")
        for name, value in (
            ("reflected_power", 0.01),
            ("transmitted_power", 0.90),
            ("radiated_power", 0.04),
            ("absorbed_power", 0.05),
            ("incident_power", 1.0),
        ):
            powers.attrs[name] = value
        modes = result.create_group("modes")
        modes.attrs["count"] = 0

        if include_scene:
            scene = result.create_group("scene")
            scene.attrs["format"] = "wavefem-scene"
            scene.attrs["version"] = 1
            scene.attrs["coordinate_order"] = "x,z"
            scene.create_dataset(
                "points",
                data=np.asarray(((-1.0, 1.0, 1.0, -1.0), (0.0, 0.0, 2.0, 2.0))),
            )
            scene.create_dataset(
                "triangles", data=np.asarray(((0, 0), (1, 2), (2, 3)), dtype=np.int64)
            )
            scene.create_dataset("eps_r", data=np.asarray((1.0, 3.4 + 0.01j)))
            scene.create_dataset("x_span", data=np.asarray((-1.0, 1.0)))
            scene.create_dataset("z_span", data=np.asarray((0.0, 2.0)))
            lines = scene.create_group("lines")
            lines.attrs["count"] = 2
            lines.create_dataset("kind", data=np.asarray((b"pec", b"pml"), dtype="S9"))
            lines.create_dataset(
                "endpoints",
                data=np.asarray(
                    (((-1.0, 0.0), (-1.0, 2.0)), ((-1.0, 0.5), (1.0, 0.5)))
                ),
            )
            lines.create_dataset("label", data=np.asarray((b"outer PEC", b"left PML")))


def test_load_h5_reads_scene_without_importing_solver(tmp_path) -> None:
    path = tmp_path / "scene.h5"
    write_result_file(path)
    loaded = load_h5(path)
    assert loaded.kind == "single"
    assert loaded.frequencies_hz.tolist() == [2.0e9]
    assert not loaded.frequencies_hz.flags.writeable
    result = loaded.results[0]
    assert result.scene is not None
    np.testing.assert_array_equal(result.scene.points.shape, (2, 4))
    np.testing.assert_array_equal(result.scene.triangles.shape, (3, 2))
    np.testing.assert_allclose(result.scene.eps_r, (1.0, 3.4 + 0.01j))
    assert [line.kind for line in result.scene.lines] == ["pec", "pml"]
    assert result.scene.lines[0].label == "outer PEC"
    assert not result.scene.points.flags.writeable
    assert not result.scene.lines[0].endpoints.flags.writeable
    assert result.s_parameters[("right", 0, 0)] == pytest.approx(0.9)


def test_load_h5_keeps_legacy_schema_v1_without_scene(tmp_path) -> None:
    path = tmp_path / "legacy.h5"
    write_result_file(path, include_scene=False)
    assert load_h5(path).results[0].scene is None


def test_load_h5_allows_unknown_frequency_sentinel_for_single_result(tmp_path) -> None:
    path = tmp_path / "unknown.h5"
    write_result_file(path, frequency=np.nan)
    loaded = load_h5(path)
    assert np.isnan(loaded.frequencies_hz[0])
    assert loaded.results[0].frequency_hz is None


def test_load_h5_rejects_bad_scene_line_kind(tmp_path) -> None:
    path = tmp_path / "bad-kind.h5"
    write_result_file(path)
    with h5py.File(path, "r+") as handle:
        del handle["results/000000/scene/lines/kind"]
        handle["results/000000/scene/lines"].create_dataset(
            "kind", data=np.asarray((b"magic", b"pml"), dtype="S9")
        )
    with pytest.raises(ValueError, match="line kind"):
        load_h5(path)
