"""The complete wheel discovers its viewers without external installation state."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from cem_common import _native


@pytest.mark.skipif(_native.os.name != "nt", reason="Windows runtime paths")
@pytest.mark.parametrize("configuration", ["Release", "Debug"])
def test_msvc_runtime_uses_recorded_dependencies(tmp_path, monkeypatch, configuration):
    executable = tmp_path / "installed" / "viewer.exe"
    executable.parent.mkdir()
    prefix = tmp_path / "vcpkg" / "x64-windows"
    if configuration == "Debug":
        prefix /= "debug"
    runtime = prefix / "bin"
    runtime.mkdir(parents=True)
    platform = prefix / "Qt6" / "plugins" / "platforms"
    platform.mkdir(parents=True)
    executable.with_suffix(".runtime.txt").write_text(
        f"compiler=MSVC\n[dlls]\n{runtime / 'Qt6Core.dll'}\n"
        f"[directories]\n{runtime}\n[platform-plugin]\n{platform / 'qwindows.dll'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", "C:\\msys64\\mingw64\\bin")
    monkeypatch.setenv("QT_PLUGIN_PATH", "wrong-qt")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    environment = _native.source_build_environment(executable, ())
    assert environment["PATH"].split(_native.os.pathsep)[:2] == [
        str(executable.parent), str(runtime)
    ]
    assert environment["QT_PLUGIN_PATH"] == str(platform.parent)
    assert environment["QT_QPA_PLATFORM_PLUGIN_PATH"] == str(platform)
    assert environment["QT_QPA_PLATFORM"] == "offscreen"


@pytest.mark.skipif(_native.os.name != "nt", reason="Windows-only release bundle")
def test_bundled_runtime_location_and_environment(tmp_path, monkeypatch):
    package = tmp_path / "installed fdfd"
    binary = package / "native/bin/fem-periodic-mode-viewer.exe"
    binary.parent.mkdir(parents=True)
    binary.touch()
    (package / "native/build-manifest.json").write_text("{}")
    monkeypatch.setattr(_native, "find_spec", lambda name: SimpleNamespace(origin=str(package / "__init__.py")))
    monkeypatch.setenv("QT_PLUGIN_PATH", "another-python/qt/plugins")
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "another-python/qt/platforms")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert _native.bundled_executable("fem-periodic-mode-viewer") == binary
    assert _native.bundled_executable("missing") is None
    environment = _native.bundled_environment(binary)
    assert "QT_PLUGIN_PATH" not in environment
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in environment
    assert environment["QT_QPA_PLATFORM"] == "offscreen"
    assert environment["PATH"].split(_native.os.pathsep)[0] == str(binary.parent)


@pytest.mark.skipif(_native.os.name != "nt", reason="Windows-only native applications")
def test_editable_install_finds_cmake_install_tree(tmp_path, monkeypatch):
    source = tmp_path / "checkout/fdfd/__init__.py"
    source.parent.mkdir(parents=True)
    source.touch()
    site_packages = tmp_path / "environment/Lib/site-packages"
    binary = site_packages / "fdfd/native/bin/transmission-line-calculator.exe"
    binary.parent.mkdir(parents=True)
    binary.touch()
    installed = SimpleNamespace(locate_file=lambda path: site_packages / path)
    monkeypatch.setattr(_native, "find_spec", lambda name: SimpleNamespace(origin=str(source)))
    monkeypatch.setattr(_native, "distribution", lambda name: installed)

    assert _native.bundled_executable("transmission-line-calculator") == binary


@pytest.mark.parametrize("family", ("periodic", "scattering"))
def test_explicit_override_precedes_bundle_and_bundle_precedes_checkout(tmp_path, monkeypatch, family):
    bundled = tmp_path / "bundled.exe"
    configured = tmp_path / "override.exe"
    bundled.touch()
    configured.touch()
    if family == "periodic":
        from fem_periodic_modes import persistence as module
        variable = "FEM_PERIODIC_MODE_VIEWER_EXECUTABLE"
        find = lambda: module._viewer_candidates("fem-periodic-mode-viewer.exe")[0]
    else:
        from fem_waveguide_scattering import viewer as module
        variable = "FEM_WAVEGUIDE_SCATTERING_VIEWER_EXECUTABLE"
        find = module.find_viewer_executable
    monkeypatch.setattr(module, "bundled_executable", lambda name: bundled)
    monkeypatch.delenv(variable, raising=False)
    assert find() == bundled
    monkeypatch.setenv(variable, str(configured))
    assert find() == configured
