from __future__ import annotations
from fem_waveguide_scattering.hdf5 import load_h5 as _internal_load_h5
from fem_waveguide_scattering.viewer import find_viewer_executable as _internal_find_viewer_executable
from fem_waveguide_scattering.viewer import launch_viewer as _internal_launch_viewer

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import fem_waveguide_scattering as wf
from fem_waveguide_scattering import inspect_h5, viewer
from fem_waveguide_scattering.results import ScatteringResult
from fem_waveguide_scattering.sweep import FrequencySweepResult


def _result(*, frequency_hz: float = 1.0e9) -> ScatteringResult:
    coordinates = np.asarray(
        ((0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)), dtype=float
    )
    incident = np.ones((3, 4), dtype=np.complex128)
    scattered = np.full((3, 4), 0.1j, dtype=np.complex128)
    return ScatteringResult(
        coordinates=coordinates,
        E_incident=incident,
        E_scattered=scattered,
        H_incident=incident / 377.0,
        H_scattered=scattered / 377.0,
        s_parameters={("left", 0, 0): 0.1j, ("right", 0, 0): 0.9},
        reflected_power=0.01,
        transmitted_power=0.81,
        radiated_power=0.09,
        absorbed_power=0.09,
        incident_power=1.0,
        ndofs=4,
        frequency_hz=frequency_hz,
    )


def _disable_external_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FEM_WAVEGUIDE_SCATTERING_VIEWER_EXECUTABLE", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(viewer.shutil, "which", lambda _name: None)


def test_finder_prefers_explicit_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / viewer._executable_names()[0]
    executable.write_bytes(b"viewer")
    monkeypatch.setenv("FEM_WAVEGUIDE_SCATTERING_VIEWER_EXECUTABLE", str(executable))
    monkeypatch.setattr(viewer.shutil, "which", lambda _name: None)
    monkeypatch.setattr(viewer, "_repository_root", lambda: None)

    assert _internal_find_viewer_executable() == executable.resolve()


def test_finder_discovers_repository_root_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_external_discovery(monkeypatch)
    path_copy = tmp_path / "installed" / viewer._executable_names()[0]
    path_copy.parent.mkdir()
    path_copy.write_bytes(b"older viewer")
    executable = (
        tmp_path / "build-native" / "apps" / "fem_waveguide_scattering_viewer" / viewer._executable_names()[0]
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"viewer")
    monkeypatch.setattr(viewer, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(viewer.shutil, "which", lambda _name: str(path_copy))

    assert _internal_find_viewer_executable() == executable.resolve()


def test_launch_viewer_defaults_to_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / viewer._executable_names()[0]
    executable.write_bytes(b"viewer")
    captured: list[list[str]] = []
    sentinel = object()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(viewer, "find_viewer_executable", lambda: executable)
    monkeypatch.setattr(
        viewer.subprocess,
        "Popen",
        lambda command: captured.append(command) or sentinel,
    )

    assert _internal_launch_viewer() is sentinel
    assert captured == [[str(executable), str(tmp_path.resolve())]]


def test_launch_viewer_passes_build_runtime_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / viewer._executable_names()[0]
    executable.write_bytes(b"viewer")
    environment = {"PATH": "native-runtime"}
    captured: list[tuple[list[str], object]] = []
    sentinel = object()
    monkeypatch.setattr(viewer, "find_viewer_executable", lambda: executable)
    monkeypatch.setattr(
        viewer, "_build_runtime_environment", lambda _executable: environment
    )
    monkeypatch.setattr(
        viewer.subprocess,
        "Popen",
        lambda command, *, env: captured.append((command, env)) or sentinel,
    )

    assert _internal_launch_viewer(tmp_path) is sentinel
    assert captured == [([str(executable), str(tmp_path.resolve())], environment)]


@pytest.mark.skipif(viewer.os.name != "nt", reason="MinGW runtime is Windows-only")
def test_build_runtime_environment_uses_cmake_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    executable = build / viewer._executable_names()[0]
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_CXX_COMPILER:FILEPATH={runtime / 'c++.exe'}\n",
        encoding="utf-8",
    )
    conda_runtime = tmp_path / "conda" / "Library" / "bin"
    conda_runtime.mkdir(parents=True)
    monkeypatch.setenv(
        "PATH",
        viewer.os.pathsep.join((str(conda_runtime), str(runtime), r"C:\Windows")),
    )
    monkeypatch.setenv("QT_PLUGIN_PATH", str(tmp_path / "wrong-plugins"))
    monkeypatch.setenv(
        "QT_QPA_PLATFORM_PLUGIN_PATH", str(tmp_path / "wrong-platforms")
    )

    environment = viewer._build_runtime_environment(executable)

    assert environment is not None
    path_entries = environment["PATH"].split(viewer.os.pathsep)
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


def test_launch_viewer_reports_early_native_exit(tmp_path: Path) -> None:
    executable = tmp_path / viewer._executable_names()[0]

    class FailedProcess:
        def wait(self, *, timeout: float) -> int:
            assert timeout == pytest.approx(0.35)
            return 127

    with pytest.raises(wf.ViewerError, match="exit code 127"):
        viewer._confirm_viewer_started(FailedProcess(), executable)


def test_plot_is_headless(monkeypatch):
    def unexpected_show(*args, **kwargs):
        raise AssertionError("plot() must never open a window")
    monkeypatch.setattr("matplotlib.pyplot.show", unexpected_show)
    figure = _result().plot(component="Ey", quantity="magnitude")
    assert figure.axes[0].get_title() == "total Ey: abs"
    sweep = FrequencySweepResult(np.asarray((1e9, 2e9)), (_result(), _result(frequency_hz=2e9)))
    figure = sweep.plot(quantity="db")
    assert [line.get_label() for line in figure.axes[0].lines] == ["S11", "S21"]
    assert figure.axes[0].get_xlabel() == "Frequency (Hz)"


@pytest.mark.parametrize("sweep", [False, True])
def test_show_uses_temporary_archive_and_cleans_up(tmp_path, monkeypatch, sweep):
    stale = tmp_path / "wavefem_result.h5"
    stale.write_bytes(b"preserved research output")
    monkeypatch.chdir(tmp_path)
    targets = []
    class Process:
        def wait(self):
            assert targets[0].is_file()
            assert _internal_load_h5(targets[0]).kind == ("sweep" if sweep else "single")
            return 0
    process = Process()
    monkeypatch.setattr(viewer, "launch_viewer", lambda path: targets.append(Path(path)) or process)
    result = _result()
    if sweep:
        result = FrequencySweepResult(np.asarray((1e9, 2e9)), (result, _result(frequency_hz=2e9)))
    assert result.show() is process
    assert len(targets) == 1 and not targets[0].exists()
    assert stale.read_bytes() == b"preserved research output"
    assert list(tmp_path.iterdir()) == [stale]


def test_show_nonblocking_cleanup_and_failure(tmp_path, monkeypatch):
    import threading
    finished, cleaned = threading.Event(), threading.Event()
    targets = []
    class Process:
        def wait(self):
            assert finished.wait(5)
            return 0
    process = Process()
    monkeypatch.setattr(viewer, "launch_viewer", lambda path: targets.append(Path(path)) or process)
    original_unlink = Path.unlink
    def unlink(path, *args, **kwargs):
        original_unlink(path, *args, **kwargs)
        if targets and path == targets[-1]: cleaned.set()
    monkeypatch.setattr(Path, "unlink", unlink)
    assert _result().show(block=False) is process
    assert targets[0].is_file()
    finished.set()
    assert cleaned.wait(5) and not targets[0].exists()
    def fail(path):
        targets.append(Path(path))
        raise wf.ViewerError("Install the native viewer")
    monkeypatch.setattr(viewer, "launch_viewer", fail)
    with pytest.raises(wf.ViewerError, match="Install"):
        _result().show()
    assert not targets[-1].exists()


def test_inspect_cli_gui_opens_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: list[Path] = []

    class Process:
        def wait(self) -> int:
            raise AssertionError("non-blocking launch must not wait")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        inspect_h5,
        "launch_viewer",
        lambda path: captured.append(Path(path)) or Process(),
    )

    assert inspect_h5.main(["--gui"]) == 0
    assert captured == [tmp_path]
    assert "native viewer opened" in capsys.readouterr().out


def test_inspect_cli_headless_behavior_is_preserved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _result().save(tmp_path / "result.h5")

    assert inspect_h5.main([str(archive)]) == 0
    output = capsys.readouterr().out
    assert "kind = single" in output
    assert "E/H samples = (3, 4) (3, 4)" in output


def test_missing_viewer_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_external_discovery(monkeypatch)
    monkeypatch.setattr(viewer, "_repository_root", lambda: None)

    with pytest.raises(wf.ViewerError, match="FEM_WAVEGUIDE_SCATTERING_VIEWER_EXECUTABLE"):
        _internal_find_viewer_executable()
