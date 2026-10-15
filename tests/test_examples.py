"""Examples use only the public API; FDFD_RUN_EXAMPLES=1 executes them.

``FDFD_RUN_EXAMPLES=1`` runs every example except the 3D periodic ones (about
three minutes); ``FDFD_RUN_EXAMPLES=all`` adds those, which take much longer.
Windows are suppressed and outputs go to each example's ``outputs/`` folder.
"""

import ast
import os
import runpy
import sys
from pathlib import Path

import pytest

import fdfd

ROOT = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = sorted(p for p in ROOT.rglob("*.py") if "outputs" not in p.parts)
SCRIPTS = [p for p in EXAMPLES if "postprocessing" not in p.parts]


@pytest.mark.parametrize("path", EXAMPLES, ids=[str(p.relative_to(ROOT)) for p in EXAMPLES])
def test_example_imports_only_public_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fdfd"):
            module = __import__(node.module, fromlist=[a.name for a in node.names])
            for alias in node.names:
                assert hasattr(module, alias.name), f"{path.name}: {node.module}.{alias.name}"
            assert not any(part.startswith("_") for part in node.module.split(".")), path.name
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "fdfd":
            assert hasattr(fdfd, node.attr), f"{path.name}: fdfd.{node.attr}"
    source = path.read_text(encoding="utf-8")
    assert "sys.path" not in source


@pytest.mark.skipif(
    os.environ.get("FDFD_RUN_EXAMPLES") not in ("1", "all"), reason="set FDFD_RUN_EXAMPLES=1 to run examples"
)
@pytest.mark.parametrize("path", SCRIPTS, ids=[str(p.relative_to(ROOT)) for p in SCRIPTS])
def test_example_runs(path, monkeypatch):
    if path.name.startswith("3d_") and os.environ.get("FDFD_RUN_EXAMPLES") != "all":
        pytest.skip("3D examples are slow; set FDFD_RUN_EXAMPLES=all")
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [str(path)])
    runpy.run_path(str(path), run_name="__main__")
    plt.close("all")
