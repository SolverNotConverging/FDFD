"""Curated FDFD API-reference checks for the 1.0 public surface."""

import importlib
import inspect
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = json.loads((ROOT / "doc/fdfd_public_api.json").read_text(encoding="utf-8"))
EXPORTS = {
    "fdfd_waveguide_modes": {"ModeSolver1D", "ModeSolver2D", "ModeSet", "load_result"},
    "fdfd_periodic_modes": {
        "PeriodicModeSolver2D",
        "PeriodicModeSolver3D",
        "PeriodicModeSet",
        "load_result",
    },
    "fdfd_band_structure": {"BandStructureSolver2D", "BandStructureResult", "load_result"},
    "fdfd_scattering": {"ScatteringSolver2D", "ScatteringResult", "load_result"},
}


@pytest.mark.parametrize("package", INVENTORY)
def test_fdfd_reference_matches_curated_public_api(package: str) -> None:
    module = importlib.import_module(package)
    assert set(module.__all__) == EXPORTS[package]
    family = package.removeprefix("fdfd_")
    reference = (ROOT / "doc" / "solvers" / "fdfd" / family / "API_REFERENCE.rst").read_text(
        encoding="utf-8"
    )
    for name in module.__all__:
        assert hasattr(module, name)
        assert f"``{name}``" in reference
    for solver_name, methods in INVENTORY[package].items():
        solver_type = getattr(module, solver_name)
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(solver_type).parameters.values()
        )
        for method in methods:
            target = solver_type if method == "__init__" else getattr(solver_type, method)
            label = solver_name if method == "__init__" else f"{solver_name}.{method}"
            assert f"``{label}``" in reference
            for parameter in inspect.signature(target).parameters.values():
                if parameter.name not in ("self", "cls"):
                    assert f"``{parameter.name}``" in reference


@pytest.mark.parametrize("package", INVENTORY)
def test_fdfd_user_rst_is_valid(package: str) -> None:
    from docutils.core import publish_doctree

    family = package.removeprefix("fdfd_")
    for filename in ("guide.rst", "API_REFERENCE.rst"):
        path = ROOT / "doc" / "solvers" / "fdfd" / family / filename
        messages = io.StringIO()
        publish_doctree(
            path.read_text(encoding="utf-8"),
            source_path=str(path),
            settings_overrides={
                "warning_stream": messages,
                "halt_level": 6,
                "report_level": 2,
                "syntax_highlight": "none",
            },
        )
        assert not messages.getvalue(), messages.getvalue()
