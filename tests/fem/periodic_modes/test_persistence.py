from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from fem_periodic_modes import PeriodicMode, PeriodicModeSet, PeriodicSampledFields
from cem_common.errors import PersistenceError
from fem_periodic_modes.persistence import load_periodic_h5, open_periodic_h5, save_periodic_h5, save_periodic_sweep_h5, validate_periodic_h5
from fem_periodic_modes import persistence
from fem_periodic_modes import visualization
from fem_periodic_modes import inspect_h5


def _mode_set(
    frequency: float = 10.0e9,
    *,
    gauss_residual: float | None = None,
    **metadata_overrides,
) -> PeriodicModeSet:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]])
    cells = np.asarray([[0, 1, 2], [0, 2, 3]])
    samples = np.asarray([[2.0 / 3.0, 2.0 / 3.0], [1.0 / 3.0, 4.0 / 3.0]])
    common_metadata = {
        "cell_epsilon_r": np.full((2, 3), 2.25 - 0.01j),
        "cell_mu_r": np.ones((2, 3), dtype=np.complex128),
        "cell_pml_fraction": np.asarray([0.0, 0.25]),
        "mesh_element_tags": np.asarray([1, 2]),
        "periodic_node_pairs": np.asarray([[2, 1], [3, 0]]),
        **metadata_overrides,
    }
    modes = []
    for index, neff in enumerate((1.1 - 0.01j, -0.9 - 0.02j), 1):
        scale = float(index)
        fields = PeriodicSampledFields(
            samples,
            {
                "Ex": scale * np.asarray([1.0 + 0.2j, 0.4]),
                "Ey": np.zeros(2),
                "Ez": scale * np.asarray([0.1, 0.3j]),
                "Hx": np.asarray([0.2, 0.3]),
                "Hy": scale * np.asarray([0.5j, 0.6]),
                "Hz": np.zeros(2),
            },
            dimension=2,
            mesh_points=points,
            mesh_cells=cells,
            sample_element_indices=[0, 1],
            material=np.full((2, 3), 2.25 - 0.01j),
            metadata=common_metadata,
        )
        modes.append(
            PeriodicMode(
                neff=neff,
                k0=2.0 * np.pi * frequency / 299_792_458.0,
                period=2.0,
                fields=fields,
                coefficients=scale * np.asarray([1.0, 0.2j, 0.3, -0.1j]),
                index=index,
                polarization="TE" if index == 1 else "TM",
                power=complex((-1) ** (index + 1)),
                direction="forward" if index == 1 else "backward",
                normalization="unit-longitudinal-power",
                residual=index * 1.0e-11,
                gauss_residual=gauss_residual,
                pml_fraction=0.02 * index,
            )
        )
    return PeriodicModeSet(
        modes,
        frequency=frequency,
        period=2.0,
        dimension=2,
        metadata={
            "backend": "dense-qz",
            **common_metadata,
            "boundary_facets": {
                "outer_pec": np.asarray([[0, 1], [1, 2]]),
                "periodic_master": np.asarray([[3, 0]]),
                "periodic_slave": np.asarray([[1, 2]]),
            },
        },
    )


def _mode_set_3d(**metadata_overrides) -> PeriodicModeSet:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    cells = np.asarray([[0, 1, 2, 3]])
    metadata = {
        "cell_epsilon_r": np.ones((1, 3), dtype=np.complex128),
        "cell_mu_r": np.ones((1, 3), dtype=np.complex128),
        "cell_pml_fraction": np.zeros(1),
        "edge_nodes": np.asarray(
            [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]], dtype=np.int64
        ),
        "cell_edges": np.asarray([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        "cell_edge_signs": np.ones((1, 6), dtype=np.int8),
        "periodic_node_pairs": np.asarray([[2, 1], [3, 0]], dtype=np.int64),
        "periodic_edge_pairs": np.asarray([[5, 0, -1]], dtype=np.int64),
        "physical_names": {1: "tetrahedral-domain"},
        **metadata_overrides,
    }
    fields = PeriodicSampledFields(
        [[0.25, 0.25, 0.25]],
        {name: np.asarray([1.0 + 0.0j]) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")},
        dimension=3,
        mesh_points=points,
        mesh_cells=cells,
        sample_element_indices=[0],
        metadata=metadata,
    )
    frequency = 10.0e9
    mode = PeriodicMode(
        neff=1.0,
        k0=2.0 * np.pi * frequency / 299_792_458.0,
        period=1.0,
        fields=fields,
        coefficients=np.ones(6, dtype=np.complex128),
        residual=1.0e-12,
    )
    return PeriodicModeSet(
        [mode], frequency=frequency, period=1.0, dimension=3, metadata=metadata
    )


def test_exact_round_trip_and_one_mode_hyperslab(tmp_path) -> None:
    original = _mode_set()
    path = save_periodic_h5(original, tmp_path / "modes.h5")
    report = validate_periodic_h5(path, deep=True)
    assert report.case_count == 1
    assert report.mode_count == 2
    with h5py.File(path, "r", libver=("v110", "v110")) as compatible:
        assert compatible.attrs["format"] == "cem-fem-results"

    with open_periodic_h5(path) as archive:
        assert archive.mode_count == 2
        selected = archive.load_case(modes=1)
        assert len(selected) == 1
        assert selected[0].index == 1
        np.testing.assert_array_equal(selected[0].coefficients, original[1].coefficients)

    loaded = load_periodic_h5(path)
    assert isinstance(loaded, PeriodicModeSet)
    np.testing.assert_array_equal(loaded.neff, original.neff)
    for actual, expected in zip(loaded, original, strict=True):
        np.testing.assert_array_equal(actual.coefficients, expected.coefficients)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            np.testing.assert_array_equal(actual.component(component), expected.component(component))

    with h5py.File(path, "r") as archive:
        for name in ("E", "H"):
            dataset = archive[f"cases/000000/visualization/{name}"]
            assert dataset.chunks[0] == 1
            assert dataset.compression == "gzip" and dataset.compression_opts == 4
            assert dataset.shuffle and dataset.fletcher32
        assert archive["meshes/000000/periodic/node_pairs"].shape == (2, 2)
        assert archive["meshes/000000/boundary/facets"].shape[1] == 2


def test_sweep_deduplicates_mesh_and_material_state(tmp_path) -> None:
    path = save_periodic_sweep_h5(
        (_mode_set(10.0e9), _mode_set(11.0e9)), tmp_path / "sweep.h5"
    )
    with h5py.File(path, "r") as archive:
        assert archive.attrs["kind"] == "sweep"
        assert len(archive["meshes"]) == 1
        assert len(archive["material_states"]) == 1
        np.testing.assert_array_equal(archive["index/mesh_index"], [0, 0])
    loaded = load_periodic_h5(path, modes=[0])
    assert isinstance(loaded, tuple) and len(loaded) == 2
    assert all(len(case) == 1 for case in loaded)


def test_mixed_gauss_availability_round_trip(tmp_path) -> None:
    path = save_periodic_sweep_h5(
        (_mode_set(10.0e9), _mode_set(11.0e9, gauss_residual=3.0e-8)),
        tmp_path / "mixed-gauss.h5",
    )
    with h5py.File(path, "r") as archive:
        np.testing.assert_array_equal(archive["index/gauss_available"], [0, 0, 1, 1])
    first, second = load_periodic_h5(path)
    assert all(mode.gauss_residual is None for mode in first)
    assert all(mode.gauss_residual == pytest.approx(3.0e-8) for mode in second)


def test_atomic_failure_preserves_existing_archive(tmp_path, monkeypatch) -> None:
    path = save_periodic_h5(_mode_set(), tmp_path / "atomic.h5")
    before = hashlib.sha256(path.read_bytes()).digest()

    def fail(*args, **kwargs):
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(persistence, "_write_archive", fail)
    with pytest.raises(RuntimeError, match="injected"):
        save_periodic_h5(_mode_set(), path)
    assert hashlib.sha256(path.read_bytes()).digest() == before
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"cell_edges": np.asarray([[0, 1, 2, 3, 5, 4]])}, "local Nedelec edge order"),
        ({"cell_edge_signs": np.asarray([[1, 1, 1, 1, 1, -1]])}, "canonical edge orientation"),
        ({"periodic_edge_pairs": np.asarray([[5, 0, 1]])}, "periodic node map"),
    ],
)
def test_3d_writer_rejects_inconsistent_canonical_edge_metadata(
    tmp_path, override, message
) -> None:
    with pytest.raises(PersistenceError, match=message):
        save_periodic_h5(_mode_set_3d(**override), tmp_path / "invalid-edges.h5")


def test_2d_writer_rejects_non_bijective_periodic_node_map(tmp_path) -> None:
    with pytest.raises(PersistenceError, match="one-to-one"):
        save_periodic_h5(
            _mode_set(periodic_node_pairs=np.asarray([[2, 1], [2, 0]])),
            tmp_path / "duplicate-periodic-node.h5",
        )


def test_3d_deep_validation_and_physical_names_round_trip(tmp_path) -> None:
    path = save_periodic_h5(_mode_set_3d(), tmp_path / "tetrahedron.h5")
    assert validate_periodic_h5(path, deep=True).mode_count == 1
    loaded = load_periodic_h5(path)
    assert loaded.metadata["physical_names"] == {"1": "tetrahedral-domain"}

    with h5py.File(path, "r+") as archive:
        del archive["meshes/000000/cell_edge_sign"]
    with pytest.raises(PersistenceError, match="cell_edge_signs"):
        validate_periodic_h5(path, deep=True)


def test_reader_enforces_expanded_coefficient_contract_and_scalar_primary(tmp_path) -> None:
    path = save_periodic_h5(_mode_set(), tmp_path / "coefficients.h5")
    with h5py.File(path, "r+") as archive:
        coefficients = archive["cases/000000/coefficients"]
        primary = coefficients["primary_unknown"].asstr()[0]
        del coefficients["primary_unknown"]
        coefficients.attrs["primary_unknown"] = primary
    loaded = load_periodic_h5(path)
    assert all(mode.metadata["primary_unknown"] == primary for mode in loaded)

    with h5py.File(path, "r+") as archive:
        archive["cases/000000/coefficients"].attrs["full_expanded"] = 0
    with pytest.raises(PersistenceError, match="full_expanded"):
        validate_periodic_h5(path, deep=True)

    numeric_flag = save_periodic_h5(_mode_set(), tmp_path / "numeric-flag.h5")
    with h5py.File(numeric_flag, "r+") as archive:
        archive["cases/000000/coefficients"].attrs["full_expanded"] = 1.0
    with pytest.raises(PersistenceError, match="full_expanded"):
        validate_periodic_h5(numeric_flag, deep=True)

    numeric_primary = save_periodic_h5(_mode_set(), tmp_path / "numeric-primary.h5")
    with h5py.File(numeric_primary, "r+") as archive:
        coefficients = archive["cases/000000/coefficients"]
        del coefficients["primary_unknown"]
        coefficients.attrs["primary_unknown"] = 1
    with pytest.raises(PersistenceError, match="HDF5 string"):
        validate_periodic_h5(numeric_primary, deep=True)

    numeric_space = save_periodic_h5(_mode_set(), tmp_path / "numeric-space.h5")
    with h5py.File(numeric_space, "r+") as archive:
        archive["cases/000000/coefficients"].attrs["space"] = 1
    with pytest.raises(PersistenceError, match="HDF5 string"):
        validate_periodic_h5(numeric_space, deep=True)


def test_reader_rejects_complex64_payload(tmp_path) -> None:
    path = save_periodic_h5(_mode_set(), tmp_path / "complex64.h5")
    with h5py.File(path, "r+") as archive:
        visualization = archive["cases/000000/visualization"]
        values = visualization["E"][...].astype(np.complex64)
        del visualization["E"]
        visualization.create_dataset("E", data=values)
    with pytest.raises(PersistenceError, match="complex128|float64"):
        load_periodic_h5(path)


def test_reader_validates_optional_mode_power_contract(tmp_path) -> None:
    missing = save_periodic_h5(_mode_set(), tmp_path / "missing-mode-power.h5")
    with h5py.File(missing, "r+") as archive:
        del archive["cases/000000/mode_metadata/power"]
    with pytest.raises(PersistenceError, match="provide has_power and power together"):
        validate_periodic_h5(missing, deep=True)

    bad_mask = save_periodic_h5(_mode_set(), tmp_path / "bad-power-mask.h5")
    with h5py.File(bad_mask, "r+") as archive:
        metadata = archive["cases/000000/mode_metadata"]
        del metadata["has_power"]
        metadata.create_dataset("has_power", data=np.asarray([1.0, 2.0]))
    with pytest.raises(PersistenceError, match="integer 0/1 vector"):
        validate_periodic_h5(bad_mask, deep=True)


def test_mesh_deduplication_includes_physical_name_table(tmp_path) -> None:
    first = _mode_set_3d(physical_names={1: "first"})
    second = _mode_set_3d(physical_names={1: "second"})
    path = save_periodic_sweep_h5((first, second), tmp_path / "named-sweep.h5")
    with h5py.File(path, "r") as archive:
        assert len(archive["meshes"]) == 2
        np.testing.assert_array_equal(archive["index/mesh_index"], [0, 1])


def test_gui_visualization_saves_all_modes_and_launches(
    tmp_path, monkeypatch
) -> None:
    modes = _mode_set()
    marker = object()
    launched: list[Path] = []

    def fake_launch(path, *, _remove_on_exit=False):
        assert _remove_on_exit is True
        launched.append(Path(path))
        return marker

    monkeypatch.setattr(visualization.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(visualization, "launch_viewer", fake_launch)
    actual = visualization.visualize_with_gui(modes)
    assert actual is marker
    assert len(launched) == 1
    restored = load_periodic_h5(launched[0])
    assert len(restored) == len(modes)
    np.testing.assert_allclose(restored.neff, modes.neff)
    launched[0].unlink()


def test_static_visualization_shows_matplotlib_figure(monkeypatch) -> None:
    from matplotlib.figure import Figure

    modes = _mode_set()
    shown: list[bool] = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: shown.append(True))
    supplied_axes = Figure().subplots()

    figure, axes = visualization.visualize(
        modes[0], component="Ey", quantity="real", ax=supplied_axes, colorbar=False
    )

    assert axes is supplied_axes
    assert axes.figure is figure
    assert shown == [True]


def test_gui_visualization_marks_implicit_archive_for_native_cleanup(
    tmp_path, monkeypatch
) -> None:
    modes = _mode_set()
    marker = object()
    launched: list[tuple[Path, bool]] = []

    def fake_launch(path, *, _remove_on_exit=False):
        launched.append((Path(path), _remove_on_exit))
        return marker

    monkeypatch.setattr(visualization.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(visualization, "launch_viewer", fake_launch)
    assert visualization.visualize_with_gui(modes) is marker
    assert len(launched) == 1
    assert launched[0][0].is_file()
    assert launched[0][1] is True
    launched[0][0].unlink()


def test_launch_viewer_accepts_directory_and_discovers_macos_bundle(
    tmp_path, monkeypatch
) -> None:
    executable = tmp_path / "fem-periodic-mode-viewer.exe"
    executable.write_bytes(b"test executable placeholder")
    calls: list[list[str]] = []
    marker = object()

    monkeypatch.setenv("FEM_PERIODIC_MODE_VIEWER_EXECUTABLE", str(executable))
    monkeypatch.setattr(
        persistence.subprocess,
        "Popen",
        lambda arguments: calls.append(arguments) or marker,
    )
    assert persistence.launch_viewer(tmp_path) is marker
    assert calls == [[str(executable.resolve()), str(tmp_path.resolve())]]
    assert any(
        "fem-periodic-mode-viewer.app/Contents/MacOS/fem-periodic-mode-viewer"
        in candidate.as_posix()
        for candidate in persistence._viewer_candidates("fem-periodic-mode-viewer")
    )


def test_launch_viewer_passes_build_runtime_environment(
    tmp_path, monkeypatch
) -> None:
    executable = tmp_path / "fem-periodic-mode-viewer.exe"
    executable.write_bytes(b"test executable placeholder")
    environment = {"PATH": "native-runtime"}
    calls: list[tuple[list[str], object]] = []
    marker = object()

    monkeypatch.setenv("FEM_PERIODIC_MODE_VIEWER_EXECUTABLE", str(executable))
    monkeypatch.setattr(
        persistence, "_build_runtime_environment", lambda _executable: environment
    )
    monkeypatch.setattr(
        persistence.subprocess,
        "Popen",
        lambda arguments, *, env: calls.append((arguments, env)) or marker,
    )

    assert persistence.launch_viewer(tmp_path) is marker
    assert calls == [
        ([str(executable.resolve()), str(tmp_path.resolve())], environment)
    ]


@pytest.mark.skipif(
    persistence.os.name != "nt", reason="MinGW runtime is Windows-only"
)
def test_build_runtime_environment_uses_cmake_toolchain(
    tmp_path, monkeypatch
) -> None:
    runtime = tmp_path / "toolchain" / "bin"
    runtime.mkdir(parents=True)
    (runtime / "Qt6Core.dll").write_bytes(b"runtime")
    plugins = runtime.parent / "share" / "qt6" / "plugins"
    plugins.mkdir(parents=True)
    platform_plugins = plugins / "platforms"
    platform_plugins.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    executable = build / "fem-periodic-mode-viewer.exe"
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_CXX_COMPILER:FILEPATH={runtime / 'c++.exe'}\n",
        encoding="utf-8",
    )
    conda_runtime = tmp_path / "conda" / "Library" / "bin"
    conda_runtime.mkdir(parents=True)
    monkeypatch.setenv(
        "PATH",
        persistence.os.pathsep.join(
            (str(conda_runtime), str(runtime), r"C:\Windows")
        ),
    )
    monkeypatch.setenv("QT_PLUGIN_PATH", str(tmp_path / "wrong-plugins"))
    monkeypatch.setenv(
        "QT_QPA_PLATFORM_PLUGIN_PATH", str(tmp_path / "wrong-platforms")
    )

    environment = persistence._build_runtime_environment(executable)

    assert environment is not None
    path_entries = environment["PATH"].split(persistence.os.pathsep)
    assert path_entries[0] == str(runtime.resolve())
    assert (
        sum(
            entry.casefold() == str(runtime.resolve()).casefold()
            for entry in path_entries
        )
        == 1
    )
    assert environment["QT_PLUGIN_PATH"] == str(plugins)
    assert environment["QT_QPA_PLATFORM_PLUGIN_PATH"] == str(platform_plugins)


def test_launch_viewer_reports_early_native_exit(tmp_path) -> None:
    executable = tmp_path / "fem-periodic-mode-viewer.exe"

    class FailedProcess:
        def wait(self, *, timeout: float) -> int:
            assert timeout == pytest.approx(0.35)
            return 127

    with pytest.raises(PersistenceError, match="exit code 127"):
        persistence._confirm_viewer_started(FailedProcess(), executable)


def test_viewer_discovery_prefers_checkout_multiconfig_build(
    tmp_path, monkeypatch
) -> None:
    package_file = tmp_path / "package" / "persistence.py"
    package_file.parent.mkdir()
    checkout = (
        tmp_path
        / "apps" / "fem_periodic_mode_viewer"
        / "build-msvc"
        / "Debug"
        / "fem-periodic-mode-viewer.exe"
    )
    checkout.parent.mkdir(parents=True)
    checkout.write_bytes(b"checkout")
    installed = tmp_path / "installed" / "fem-periodic-mode-viewer.exe"
    installed.parent.mkdir()
    installed.write_bytes(b"installed")
    monkeypatch.setattr(persistence, "__file__", str(package_file))
    monkeypatch.delenv("FEM_PERIODIC_MODE_VIEWER_EXECUTABLE", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(persistence.shutil, "which", lambda _name: str(installed))

    candidates = persistence._viewer_candidates("fem-periodic-mode-viewer.exe")
    assert candidates.index(checkout) < candidates.index(installed)
    assert any("RelWithDebInfo" in candidate.parts for candidate in candidates)
    assert any("MinSizeRel" in candidate.parts for candidate in candidates)


def test_inspect_h5_gui_defaults_to_current_directory(tmp_path, monkeypatch) -> None:
    launched: list[object] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(inspect_h5, "launch_viewer", launched.append)
    assert inspect_h5.main(["--gui"]) == 0
    assert launched == [tmp_path.resolve()]


def test_inspect_h5_directory_fails_when_every_archive_is_invalid(tmp_path) -> None:
    (tmp_path / "not-periodic.h5").write_bytes(b"not hdf5")
    assert inspect_h5.main([str(tmp_path)]) == 1


def test_inspect_h5_rejects_deep_gui_combination(tmp_path) -> None:
    with pytest.raises(SystemExit, match="2"):
        inspect_h5.main(["--gui", "--deep", str(tmp_path)])
