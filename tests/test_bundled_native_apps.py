"""The complete wheel discovers its viewers without external installation state."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from cem_common import _native


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
