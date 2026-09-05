from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fem_waveguide_scattering.constants import C0
from fem_waveguide_scattering.exceptions import ConfigurationError
from fem_waveguide_scattering.hdf5 import (
    H5FileData,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    load_h5,
    save_result_h5,
    save_sweep_h5,
)
from fem_waveguide_scattering.modes import Mode
from fem_waveguide_scattering.results import ScatteringResult
from fem_waveguide_scattering.scene import Scene2D, SceneLine
from fem_waveguide_scattering.sweep import FrequencySweepResult


FREQUENCY_HZ = 2.5e9
KY = 3.25


def make_scene() -> Scene2D:
    return Scene2D(
        points=np.asarray(((0.0, 1.0, 1.0, 0.0), (2.0, 2.0, 3.0, 3.0))),
        triangles=np.asarray(((0, 0), (1, 2), (2, 3))),
        eps_r=np.asarray((1.0 + 0.0j, 4.0 + 0.05j)),
        x_span=(0.0, 1.0),
        z_span=(2.0, 3.0),
        lines=(
            SceneLine("pec", ((0.0, 2.0), (1.0, 2.0)), "PEC boundary"),
            SceneLine("pmc", ((0.0, 3.0), (1.0, 3.0)), "PMC boundary"),
            SceneLine("wave_port", ((0.0, 2.4), (1.0, 2.4)), "left wave port"),
            SceneLine("pml", ((0.2, 2.0), (0.2, 3.0)), "x-PML interface"),
        ),
    )


def make_result(*, frequency_hz: float | None = FREQUENCY_HZ) -> ScatteringResult:
    coordinates = np.vstack((np.linspace(-1.0, 1.0, 5), np.linspace(2.0, 3.0, 5)))
    incident = np.arange(15, dtype=float).reshape(3, 5) + 1.0j
    scattered = (0.1 + 0.2j) * np.ones((3, 5), dtype=np.complex128)
    solve_info: dict[str, object] = {
        "projected_incoming_amplitude": 1.0 + 0.25j,
        "nested": {"array": np.asarray([1.0, 2.0])},
    }
    if frequency_hz is not None:
        solve_info["length_scale"] = C0 / (2.0 * np.pi * frequency_hz)
    return ScatteringResult(
        coordinates=coordinates,
        E_incident=incident,
        E_scattered=scattered,
        H_incident=incident / 377.0,
        H_scattered=scattered / 377.0,
        s_parameters={
            ("left", 0, 0): 0.1 + 0.2j,
            ("right", 0, 0): 0.9 - 0.05j,
        },
        reflected_power=0.05,
        transmitted_power=0.85,
        radiated_power=0.07,
        absorbed_power=0.03,
        incident_power=1.0,
        ndofs=42,
        solve_info=solve_info,
        mesh_info={"elements": 19, "span": (1.0, 2.0)},
        projection_condition_numbers={"left": 2.0, "right": 3.0},
        reference_planes={"left": -0.5, "right": 0.5},
        port_betas={
            ("left", 0): 4.0 + 0.0j,
            ("right", 0): 4.0 + 0.0j,
        },
    )


def make_mode(*, frequency_hz: float = FREQUENCY_HZ, ky: float = KY) -> Mode:
    x_nodes = np.linspace(-0.5, 0.5, 5)
    cells = x_nodes.size - 1
    nodal = x_nodes.size
    e_x = np.linspace(1.0, 2.0, cells).astype(np.complex128)
    e_y = (0.2j * np.linspace(1.0, 2.0, nodal)).astype(np.complex128)
    e_z = np.linspace(0.1, 0.5, nodal).astype(np.complex128)
    h_x = np.linspace(0.01, 0.04, cells).astype(np.complex128)
    h_y = np.linspace(0.05, 0.08, cells).astype(np.complex128)
    h_z = np.linspace(0.09, 0.12, cells).astype(np.complex128)
    omega = 2.0 * np.pi * frequency_hz
    return Mode(
        beta=5.0 + 0.0j,
        neff=1.5 + 0.0j,
        E_x=e_x,
        E_y=e_y,
        E_z=e_z,
        H_x=h_x,
        H_y=h_y,
        H_z=h_z,
        x_nodes=x_nodes,
        power=1.0,
        complex_power=1.0 + 0.01j,
        ky=ky,
        omega=omega,
        direction="forward",
        classification="propagating",
        normalization="unit-power",
        residual=1e-10,
        divergence_residual=2e-10,
        H_x_left=h_x - 0.001,
        H_x_right=h_x + 0.001,
    )


def test_single_result_round_trip_preserves_complex_fields_modes_and_metadata(
    tmp_path,
) -> None:
    source = make_result()
    mode = make_mode()
    path = tmp_path / "single.h5"

    written = save_result_h5(source, path, modes=(mode,))
    loaded = load_h5(path)

    assert written == path.resolve()
    assert isinstance(loaded, H5FileData)
    assert loaded.path == path.resolve()
    assert loaded.kind == "single"
    np.testing.assert_allclose(loaded.frequencies_hz, [FREQUENCY_HZ])
    assert not loaded.frequencies_hz.flags.writeable
    assert len(loaded.results) == 1

    result = loaded.results[0]
    assert result.frequency_hz == pytest.approx(FREQUENCY_HZ)
    assert result.ky == pytest.approx(KY)
    np.testing.assert_array_equal(result.coordinates, source.coordinates)
    np.testing.assert_array_equal(result.E_incident, source.E_incident)
    np.testing.assert_array_equal(result.E_scattered, source.E_scattered)
    np.testing.assert_array_equal(result.E_total, source.E_total)
    np.testing.assert_array_equal(result.H_total, source.H_total)
    assert result.s_parameters == source.s_parameters
    assert result.powers["reflected_power"] == source.reflected_power
    assert result.powers["incident_power"] == source.incident_power
    assert result.metadata["ndofs"] == 42
    assert result.metadata["solve_info"]["projected_incoming_amplitude"] == 1.0 + 0.25j
    np.testing.assert_array_equal(
        result.metadata["solve_info"]["nested"]["array"], [1.0, 2.0]
    )

    assert len(result.modes) == 1
    stored_mode = result.modes[0]
    np.testing.assert_array_equal(stored_mode.x, mode.x)
    np.testing.assert_array_equal(stored_mode.E, mode.E)
    np.testing.assert_array_equal(stored_mode.H, mode.H)
    assert stored_mode.metadata["beta"] == mode.beta
    assert stored_mode.metadata["normalization"] == "unit-power"
    np.testing.assert_array_equal(stored_mode.raw_components["E_y"], mode.E_y)
    np.testing.assert_array_equal(
        stored_mode.raw_components["H_x_right"], mode.H_x_right
    )

    with h5py.File(path, "r") as handle:
        assert handle.attrs["format"] == SCHEMA_NAME
        assert handle.attrs["schema_version"] == SCHEMA_VERSION
        assert handle.attrs["kind"] == "single"
        field = handle["results/000000/fields/E_incident"]
        assert field.dtype.kind == "c"
        assert field.compression == "gzip"
        mode_field = handle["results/000000/modes/000000/E"]
        assert mode_field.dtype.kind == "c"
        assert mode_field.compression == "gzip"
        s_values = handle["results/000000/s_parameters/value"]
        assert s_values.dtype.kind == "c"
        assert s_values.compression == "gzip"


def test_scene_round_trip_preserves_material_mesh_spans_and_overlays(tmp_path) -> None:
    source = replace(make_result(), scene=make_scene())
    path = tmp_path / "scene.h5"

    save_result_h5(source, path)
    loaded = load_h5(path).results[0]

    assert loaded.scene is not None
    np.testing.assert_array_equal(loaded.scene.points, source.scene.points)
    np.testing.assert_array_equal(loaded.scene.triangles, source.scene.triangles)
    np.testing.assert_array_equal(loaded.scene.eps_r, source.scene.eps_r)
    assert loaded.scene.x_span == source.scene.x_span
    assert loaded.scene.z_span == source.scene.z_span
    assert [line.kind for line in loaded.scene.lines] == [
        "pec",
        "pmc",
        "wave_port",
        "pml",
    ]
    assert [line.label for line in loaded.scene.lines] == [
        line.label for line in source.scene.lines
    ]
    assert not loaded.scene.points.flags.writeable
    assert not loaded.scene.triangles.flags.writeable
    assert not loaded.scene.eps_r.flags.writeable
    assert all(not line.endpoints.flags.writeable for line in loaded.scene.lines)

    with h5py.File(path, "r") as handle:
        scene_group = handle["results/000000/scene"]
        assert scene_group.attrs["format"] == "fem_waveguide_scattering-scene"
        assert scene_group.attrs["version"] == 1
        assert scene_group.attrs["coordinate_order"] == "x,z"
        assert scene_group["points"].compression == "gzip"
        assert scene_group["eps_r"].dtype.kind == "c"
        assert scene_group["lines/endpoints"].shape == (4, 2, 2)


def test_result_without_scene_remains_backward_compatible(tmp_path) -> None:
    path = tmp_path / "legacy-no-scene.h5"
    save_result_h5(make_result(), path)

    with h5py.File(path, "r") as handle:
        assert "scene" not in handle["results/000000"]
    assert load_h5(path).results[0].scene is None


def test_load_rejects_invalid_scene_connectivity(tmp_path) -> None:
    path = tmp_path / "bad-scene.h5"
    save_result_h5(replace(make_result(), scene=make_scene()), path)
    with h5py.File(path, "r+") as handle:
        handle["results/000000/scene/triangles"][0, 0] = 99

    with pytest.raises(ValueError, match="scene.*vertex index"):
        load_h5(path)


def test_single_result_allows_explicitly_unknown_frequency_and_ky(tmp_path) -> None:
    path = tmp_path / "unknown.h5"
    save_result_h5(make_result(frequency_hz=None), path)

    loaded = load_h5(path)

    assert loaded.kind == "single"
    assert np.isnan(loaded.frequencies_hz[0])
    assert loaded.results[0].frequency_hz is None
    assert loaded.results[0].ky is None
    assert loaded.results[0].modes == ()


def test_duck_typed_future_result_uses_frequency_ky_and_embedded_modes(tmp_path) -> None:
    current = make_result(frequency_hz=None)
    mode = make_mode()
    future = SimpleNamespace(
        **{
            name: getattr(current, name)
            for name in (
                "coordinates",
                "E_incident",
                "E_scattered",
                "H_incident",
                "H_scattered",
                "s_parameters",
                "reflected_power",
                "transmitted_power",
                "radiated_power",
                "absorbed_power",
                "incident_power",
                "ndofs",
                "solve_info",
                "mesh_info",
            )
        },
        frequency_hz=FREQUENCY_HZ,
        ky=KY,
        modes=(mode,),
    )
    path = tmp_path / "future.h5"

    save_result_h5(future, path)
    loaded = load_h5(path)

    assert loaded.results[0].frequency_hz == pytest.approx(FREQUENCY_HZ)
    assert loaded.results[0].ky == pytest.approx(KY)
    assert len(loaded.results[0].modes) == 1


def test_sweep_round_trip_preserves_order_and_per_result_modes(tmp_path) -> None:
    frequencies = np.asarray([1.0e9, 1.5e9, 2.0e9])
    results = tuple(make_result(frequency_hz=None) for _ in frequencies)
    modes = tuple((make_mode(frequency_hz=float(frequency)),) for frequency in frequencies)
    path = tmp_path / "sweep.h5"

    written = save_sweep_h5(
        frequencies,
        results,
        path,
        modes_per_result=modes,
    )
    loaded = load_h5(path)

    assert written == path.resolve()
    assert loaded.kind == "sweep"
    np.testing.assert_array_equal(loaded.frequencies_hz, frequencies)
    assert [result.frequency_hz for result in loaded.results] == pytest.approx(
        frequencies
    )
    assert [result.ky for result in loaded.results] == pytest.approx([KY] * 3)
    assert [len(result.modes) for result in loaded.results] == [1, 1, 1]


@pytest.mark.parametrize(
    ("frequencies", "results", "modes", "message"),
    [
        ([], [], None, "nonempty"),
        ([1.0], [], None, "same number"),
        ([1.0, np.nan], [make_result(), make_result()], None, "non-finite"),
        ([2.0, 1.0], [make_result(), make_result()], None, "strictly increasing"),
        ([1.0, 1.0], [make_result(), make_result()], None, "strictly increasing"),
        ([1.0], [make_result()], [(), ()], "one entry per result"),
    ],
)
def test_sweep_rejects_inconsistent_inputs(
    tmp_path, frequencies, results, modes, message
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        save_sweep_h5(
            frequencies,
            results,
            tmp_path / "bad-sweep.h5",
            modes_per_result=modes,
        )


def test_sweep_rejects_frequency_that_disagrees_with_result_metadata(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="Inconsistent frequency"):
        save_sweep_h5(
            [FREQUENCY_HZ * 2.0],
            [make_result()],
            tmp_path / "mismatch.h5",
        )


def test_load_rejects_wrong_schema_version(tmp_path) -> None:
    path = tmp_path / "future-schema.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = SCHEMA_NAME
        handle.attrs["schema_version"] = SCHEMA_VERSION + 1
        handle.attrs["kind"] = "single"
        handle.attrs["result_count"] = 1

    with pytest.raises(ValueError, match="Incompatible.*schema"):
        load_h5(path)


def test_failed_write_leaves_existing_destination_untouched(tmp_path, monkeypatch) -> None:
    from fem_waveguide_scattering import hdf5 as persistence

    path = tmp_path / "existing.h5"
    original = b"existing-file-content"
    path.write_bytes(original)

    def fail_write(*args, **kwargs) -> None:
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(persistence, "_write_file", fail_write)
    with pytest.raises(RuntimeError, match="injected write failure"):
        save_result_h5(make_result(), path)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_write_retries_transient_windows_sharing_violation(
    tmp_path, monkeypatch
) -> None:
    from fem_waveguide_scattering import hdf5 as persistence

    path = tmp_path / "retry.h5"
    real_replace = persistence.os.replace
    attempts = 0

    def intermittently_locked(source, destination) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            error = OSError("injected sharing violation")
            error.winerror = 32
            raise error
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", intermittently_locked)
    monkeypatch.setattr(persistence.time, "sleep", lambda _delay: None)

    save_result_h5(make_result(), path)

    assert attempts == 3
    assert path.is_file()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_write_does_not_retry_unrelated_replace_error(
    tmp_path, monkeypatch
) -> None:
    from fem_waveguide_scattering import hdf5 as persistence

    path = tmp_path / "unrelated-error.h5"
    attempts = 0

    def fail_replace(_source, _destination) -> None:
        nonlocal attempts
        attempts += 1
        error = OSError("injected disk failure")
        error.winerror = 112
        raise error

    monkeypatch.setattr(persistence.os, "replace", fail_replace)

    with pytest.raises(ConfigurationError, match="injected disk failure"):
        save_result_h5(make_result(), path)

    assert attempts == 1
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_public_result_and_sweep_save_wrappers_return_written_paths(tmp_path) -> None:
    result_mode = make_mode()
    result = replace(
        make_result(frequency_hz=None),
        frequency_hz=FREQUENCY_HZ,
        ky=KY,
        modes=(result_mode,),
    )
    result_path = tmp_path / "result-wrapper.h5"

    written_result = result.save(result_path)

    assert written_result == result_path.resolve()
    assert written_result.is_file()
    assert len(load_h5(written_result).results[0].modes) == 1

    sweep_frequencies = np.asarray([1.0e9, 2.0e9])
    sweep_results = tuple(
        replace(
            make_result(frequency_hz=None),
            frequency_hz=float(frequency),
            ky=KY,
            modes=(make_mode(frequency_hz=float(frequency)),),
        )
        for frequency in sweep_frequencies
    )
    sweep = FrequencySweepResult(sweep_frequencies, sweep_results)
    sweep_path = tmp_path / "sweep-wrapper.h5"

    written_sweep = sweep.save(sweep_path)

    assert written_sweep == sweep_path.resolve()
    assert written_sweep.is_file()
    assert [len(item.modes) for item in load_h5(written_sweep).results] == [1, 1]
