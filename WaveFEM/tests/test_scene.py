"""Validation tests for portable WaveFEM scene metadata."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from wavefem.scene import Scene2D, SceneLine


def _valid_scene(**changes: object) -> Scene2D:
    arguments: dict[str, object] = {
        "points": np.asarray(((0.0, 1.0, 1.0, 0.0), (2.0, 2.0, 4.0, 4.0))),
        "triangles": np.asarray(((0, 0), (1, 2), (2, 3))),
        "eps_r": np.asarray((1.0 + 0.0j, 12.0 + 0.01j)),
        "x_span": (0.0, 1.0),
        "z_span": (2.0, 4.0),
        "lines": (
            SceneLine("PEC", ((0.0, 2.0), (1.0, 2.0)), "input termination"),
            SceneLine("pml", ((0.0, 3.5), (1.0, 3.5))),
        ),
    }
    arguments.update(changes)
    return Scene2D(**arguments)  # type: ignore[arg-type]


def test_scene_normalizes_lines_and_owns_read_only_arrays() -> None:
    source_points = np.asarray(((0.0, 1.0, 1.0, 0.0), (2.0, 2.0, 4.0, 4.0)))
    source_triangles = np.asarray(((0, 0), (1, 2), (2, 3)))
    source_eps = np.asarray((1.0 + 0.0j, 12.0 + 0.01j))
    line_source = np.asarray(((0.0, 2.0), (1.0, 2.0)))
    line = SceneLine("  WAVE_PORT ", line_source, "left")
    scene = _valid_scene(
        points=source_points,
        triangles=source_triangles,
        eps_r=source_eps,
        lines=[line],
    )

    source_points[0, 0] = 99.0
    source_triangles[0, 0] = 3
    source_eps[0] = 99.0
    line_source[0, 0] = 99.0

    assert line.kind == "wave_port"
    assert scene.lines == (line,)
    assert scene.points[0, 0] == 0.0
    assert scene.triangles[0, 0] == 0
    assert scene.eps_r[0] == 1.0
    assert line.endpoints[0, 0] == 0.0
    assert not scene.points.flags.writeable
    assert not scene.triangles.flags.writeable
    assert not scene.eps_r.flags.writeable
    assert not line.endpoints.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        scene.points[0, 0] = 2.0
    with pytest.raises(FrozenInstanceError):
        scene.lines = ()  # type: ignore[misc]


@pytest.mark.parametrize("kind", ("pec", "pmc", "wave_port", "pml"))
def test_all_overlay_kinds_are_supported(kind: str) -> None:
    assert SceneLine(kind, ((0.0, 0.0), (1.0, 0.0))).kind == kind


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"kind": "radiation", "endpoints": ((0.0, 0.0), (1.0, 0.0))}, "one of"),
        ({"kind": 1, "endpoints": ((0.0, 0.0), (1.0, 0.0))}, "must be text"),
        ({"kind": "pec", "endpoints": (0.0, 1.0)}, r"shape \(2, 2\)"),
        ({"kind": "pec", "endpoints": ((0.0, 0.0), (0.0, 0.0))}, "nonzero-length"),
        ({"kind": "pec", "endpoints": ((0.0, 0.0), (np.nan, 0.0))}, "non-finite"),
        ({"kind": "pec", "endpoints": ((0.0, 0.0), (1.0, 0.0)), "label": 3}, "label"),
    ],
)
def test_scene_line_rejects_invalid_values(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SceneLine(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"points": np.ones((3, 4))}, "points must have shape"),
        ({"points": ((0.0, 1.0, np.nan), (2.0, 2.0, 4.0))}, "non-finite"),
        ({"triangles": np.asarray(((0, 1, 2),))}, "triangles must have shape"),
        ({"triangles": ((0, 0), (1, 2), (2, 4))}, "outside points"),
        ({"triangles": ((0, 0), (0, 2), (2, 3))}, "distinct vertices"),
        ({"triangles": ((0.5, 0), (1, 2), (2, 3))}, "integer vertex"),
        ({"eps_r": (1.0,)}, "one value per triangle"),
        ({"eps_r": (1.0, np.inf)}, "non-finite"),
        ({"x_span": (1.0, 0.0)}, "strictly increasing"),
        ({"z_span": (2.0, 4.0, 5.0)}, "exactly two"),
        ({"x_span": (0.25, 1.0)}, "points must lie"),
        ({"lines": (SceneLine("pec", ((0.0, 2.0), (2.0, 2.0))),)}, "lies outside"),
        ({"lines": ("pec",)}, "SceneLine"),
    ],
)
def test_scene_rejects_invalid_mesh_or_overlays(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _valid_scene(**changes)


def test_scene_rejects_geometrically_degenerate_triangle() -> None:
    with pytest.raises(ValueError, match="geometrically degenerate"):
        Scene2D(
            points=((0.0, 0.5, 1.0), (0.0, 0.5, 1.0)),
            triangles=((0,), (1,), (2,)),
            eps_r=(2.0,),
            x_span=(0.0, 1.0),
            z_span=(0.0, 1.0),
        )
