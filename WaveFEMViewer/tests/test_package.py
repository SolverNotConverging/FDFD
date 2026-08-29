from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_standalone_sources_do_not_import_wavefem_solver() -> None:
    sources = tuple((ROOT / "src" / "wavefem_viewer").glob("*.py"))
    assert sources
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "from wavefem " not in text
        assert "from wavefem." not in text
        assert "import wavefem" not in text


def test_module_help_is_headless() -> None:
    code = """
import runpy
import sys
sys.argv = ['wavefem-viewer', '--help']
runpy.run_module('wavefem_viewer', run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert "Open and visualize WaveFEM HDF5 result files" in completed.stdout
    assert "tkinter" not in completed.stderr


def test_readme_documents_install_uninstall_and_use() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for heading in ("## install", "## use", "## uninstall"):
        assert heading in readme
    assert "rf_engineering_env" in readme
    assert "z is the horizontal axis" in readme
    assert "pec boundaries as solid yellow" in readme
