"""Portable two-dimensional scene metadata for result visualization.

Scene coordinates follow WaveFEM's physical ``(x, z)`` convention even when
a viewer chooses to display ``z`` horizontally.  The material background is
stored on a conforming triangular mesh, while boundary-like objects are
represented by labelled line segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
IntArray: TypeAlias = NDArray[np.int64]
SceneKind: TypeAlias = Literal["pec", "pmc", "wave_port", "pml"]

_SCENE_KINDS = frozenset(("pec", "pmc", "wave_port", "pml"))


def _readonly_real_array(value: ArrayLike, name: str) -> FloatArray:
    """Return a finite, owned, read-only real array."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError(f"{name} must contain real values.")
    try:
        result = np.array(np.real(raw), dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value.")
    result.setflags(write=False)
    return result


def _readonly_complex_array(value: ArrayLike, name: str) -> ComplexArray:
    """Return a finite, owned, read-only complex array."""

    try:
        result = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain complex numeric values.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value.")
    result.setflags(write=False)
    return result


def _readonly_triangles(value: ArrayLike) -> IntArray:
    """Return exact, nonnegative, read-only triangle connectivity."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("triangles must contain integer vertex indices.") from exc
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise ValueError("triangles must contain integer vertex indices.")
    try:
        converted = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("triangles must contain integer vertex indices.") from exc
    try:
        exact = bool(np.all(raw == converted))
    except (TypeError, ValueError):
        exact = False
    if not exact:
        raise ValueError("triangles must contain integer vertex indices.")
    result = np.array(converted, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _span(value: object, name: str) -> tuple[float, float]:
    """Normalize one finite, strictly increasing coordinate span."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain exactly two real values.") from exc
    if raw.shape != (2,) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain exactly two real values.")
    try:
        lower, upper = (float(item) for item in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain exactly two real values.") from exc
    if not np.isfinite((lower, upper)).all() or not lower < upper:
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return lower, upper


def _domain_tolerance(*spans: tuple[float, float]) -> float:
    return 64.0 * np.finfo(float).eps * max(
        1.0, *(abs(value) for span in spans for value in span)
    )


@dataclass(frozen=True, slots=True)
class SceneLine:
    """One visualization overlay segment stored in physical ``(x, z)`` order.

    Parameters
    ----------
    kind:
        One of ``"pec"``, ``"pmc"``, ``"wave_port"``, or ``"pml"``.
        The value is normalized to lowercase.
    endpoints:
        Array with shape ``(2, 2)``.  Each row is one endpoint and each
        endpoint is ordered as ``(x, z)``.
    label:
        Optional human-readable label, such as ``"left wave port"``.
    """

    kind: SceneKind | str
    endpoints: FloatArray
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise ValueError("SceneLine.kind must be text.")
        kind = self.kind.strip().lower()
        if kind not in _SCENE_KINDS:
            choices = ", ".join(sorted(_SCENE_KINDS))
            raise ValueError(f"SceneLine.kind must be one of {choices}; received {self.kind!r}.")
        if not isinstance(self.label, str):
            raise ValueError("SceneLine.label must be text.")
        endpoints = _readonly_real_array(self.endpoints, "SceneLine.endpoints")
        if endpoints.shape != (2, 2):
            raise ValueError(
                "SceneLine.endpoints must have shape (2, 2), with rows in (x, z) order."
            )
        if np.array_equal(endpoints[0], endpoints[1]):
            raise ValueError("SceneLine endpoints must define a nonzero-length segment.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "endpoints", endpoints)


@dataclass(frozen=True, slots=True)
class Scene2D:
    """Conforming material mesh and line overlays for an x-z result scene.

    ``points`` has shape ``(2, npoints)`` in ``(x, z)`` component order,
    ``triangles`` has shape ``(3, nelements)``, and ``eps_r`` stores one
    physical (untransformed) relative permittivity value per triangle.
    Defensive copies of all arrays are marked read-only.
    """

    points: FloatArray
    triangles: IntArray
    eps_r: ComplexArray
    x_span: tuple[float, float]
    z_span: tuple[float, float]
    lines: tuple[SceneLine, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        points = _readonly_real_array(self.points, "points")
        if points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 3:
            raise ValueError("points must have shape (2, npoints), with npoints >= 3.")

        triangles = _readonly_triangles(self.triangles)
        if triangles.ndim != 2 or triangles.shape[0] != 3 or triangles.shape[1] == 0:
            raise ValueError("triangles must have shape (3, nelements), with nelements > 0.")
        if np.any(triangles < 0) or np.any(triangles >= points.shape[1]):
            raise ValueError("triangles contains a vertex index outside points.")
        if np.any(
            (triangles[0] == triangles[1])
            | (triangles[1] == triangles[2])
            | (triangles[2] == triangles[0])
        ):
            raise ValueError("Each triangle must reference three distinct vertices.")

        eps_r = _readonly_complex_array(self.eps_r, "eps_r")
        if eps_r.shape != (triangles.shape[1],):
            raise ValueError(
                "eps_r must have shape (nelements,), with one value per triangle."
            )

        x_span = _span(self.x_span, "x_span")
        z_span = _span(self.z_span, "z_span")
        tolerance = _domain_tolerance(x_span, z_span)
        if (
            np.any(points[0] < x_span[0] - tolerance)
            or np.any(points[0] > x_span[1] + tolerance)
            or np.any(points[1] < z_span[0] - tolerance)
            or np.any(points[1] > z_span[1] + tolerance)
        ):
            raise ValueError("points must lie inside x_span and z_span.")

        p0 = points[:, triangles[0]]
        p1 = points[:, triangles[1]]
        p2 = points[:, triangles[2]]
        twice_area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (
            p1[1] - p0[1]
        ) * (p2[0] - p0[0])
        area_scale = max(
            (x_span[1] - x_span[0]) * (z_span[1] - z_span[0]),
            np.finfo(float).tiny,
        )
        if np.any(np.abs(twice_area) <= 64.0 * np.finfo(float).eps * area_scale):
            raise ValueError("triangles contains a geometrically degenerate element.")

        if isinstance(self.lines, (str, bytes)):
            raise ValueError("lines must be an iterable of SceneLine objects.")
        try:
            lines = tuple(self.lines)
        except TypeError as exc:
            raise ValueError("lines must be an iterable of SceneLine objects.") from exc
        if any(not isinstance(line, SceneLine) for line in lines):
            raise ValueError("lines must contain only SceneLine objects.")
        for line in lines:
            endpoints = line.endpoints
            if (
                np.any(endpoints[:, 0] < x_span[0] - tolerance)
                or np.any(endpoints[:, 0] > x_span[1] + tolerance)
                or np.any(endpoints[:, 1] < z_span[0] - tolerance)
                or np.any(endpoints[:, 1] > z_span[1] + tolerance)
            ):
                raise ValueError(
                    f"Scene line {line.label or line.kind!r} lies outside x_span or z_span."
                )

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "eps_r", eps_r)
        object.__setattr__(self, "x_span", x_span)
        object.__setattr__(self, "z_span", z_span)
        object.__setattr__(self, "lines", lines)


__all__ = ["Scene2D", "SceneKind", "SceneLine"]
