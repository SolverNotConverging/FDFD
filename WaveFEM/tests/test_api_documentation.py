from __future__ import annotations

from pathlib import Path

import wavefem
from wavefem import (
    constants,
    exceptions,
    hdf5,
    incident,
    materials,
    modes,
    monitors,
    projection,
    results,
    scene,
    sources,
    sweep,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_REFERENCE = PROJECT_ROOT / "API_REFERENCE.md"


def test_api_reference_covers_top_level_exports() -> None:
    documentation = API_REFERENCE.read_text(encoding="utf-8")

    for name in wavefem.__all__:
        assert f"`{name}`" in documentation, (
            f"Top-level public API {name!r} is missing from API_REFERENCE.md."
        )


def test_api_reference_covers_module_exports() -> None:
    documentation = API_REFERENCE.read_text(encoding="utf-8")
    documented_modules = (
        constants,
        exceptions,
        hdf5,
        incident,
        materials,
        modes,
        monitors,
        projection,
        results,
        scene,
        sources,
        sweep,
    )

    for module in documented_modules:
        for name in module.__all__:
            assert f"`{name}`" in documentation, (
                f"Public API {module.__name__}.{name} is missing from "
                "API_REFERENCE.md."
            )
