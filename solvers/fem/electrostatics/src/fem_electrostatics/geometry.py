"""Continuous 1D/2D geometry recorded before FEM discretization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import GeometryError


FloatArray: TypeAlias = NDArray[np.float64]
Shape: TypeAlias = "Interval | Rectangle | Circle | Polygon"


def _span(value: Sequence[float], name: str) -> tuple[float, float]:
    try:
        lower, upper = (float(entry) for entry in value)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{name} must be a finite (minimum, maximum) pair.") from exc
    if not np.isfinite((lower, upper)).all() or upper <= lower:
        raise GeometryError(f"{name} must be finite and satisfy maximum > minimum.")
    return lower, upper


class Shape2D(Protocol):
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...

    def contains(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]: ...


@dataclass(frozen=True, slots=True)
class Interval:
    x: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _span(self.x, "interval"))

    @property
    def bounds(self) -> tuple[float, float]:
        return self.x

    def contains(self, x: ArrayLike, *_: ArrayLike) -> NDArray[np.bool_]:
        values = np.asarray(x, dtype=float)
        tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(self.x[0]), abs(self.x[1]))
        return (values >= self.x[0] - tolerance) & (values <= self.x[1] + tolerance)


@dataclass(frozen=True, slots=True)
class Rectangle:
    x: tuple[float, float]
    y: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _span(self.x, "rectangle x"))
        object.__setattr__(self, "y", _span(self.y, "rectangle y"))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x[0], self.x[1], self.y[0], self.y[1]

    def contains(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        scale = max(1.0, *(abs(value) for value in self.bounds))
        tolerance = 32.0 * np.finfo(float).eps * scale
        return (
            (xa >= self.x[0] - tolerance)
            & (xa <= self.x[1] + tolerance)
            & (ya >= self.y[0] - tolerance)
            & (ya <= self.y[1] + tolerance)
        )


@dataclass(frozen=True, slots=True)
class Circle:
    center: tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        radius = float(self.radius)
        if len(center) != 2 or not np.isfinite(center).all():
            raise GeometryError("circle center must contain two finite coordinates.")
        if not isfinite(radius) or radius <= 0.0:
            raise GeometryError("circle radius must be finite and positive.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        return cx - self.radius, cx + self.radius, cy - self.radius, cy + self.radius

    def contains(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, self.radius)
        return (xa - self.center[0]) ** 2 + (ya - self.center[1]) ** 2 <= (
            self.radius + tolerance
        ) ** 2


@dataclass(frozen=True, slots=True)
class Polygon:
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        points = tuple((float(x), float(y)) for x, y in self.points)
        if len(points) < 3 or not np.isfinite(points).all():
            raise GeometryError("polygon requires at least three finite points.")
        area = 0.5 * sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        if abs(area) <= np.finfo(float).eps:
            raise GeometryError("polygon points must enclose a nonzero area.")
        object.__setattr__(self, "points", points)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        values = np.asarray(self.points, dtype=float)
        return (
            float(values[:, 0].min()),
            float(values[:, 0].max()),
            float(values[:, 1].min()),
            float(values[:, 1].max()),
        )

    def contains(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        inside = np.zeros(xa.shape, dtype=bool)
        vertices = np.asarray(self.points, dtype=float)
        for index in range(len(vertices)):
            x0, y0 = vertices[index - 1]
            x1, y1 = vertices[index]
            crossing = (y0 > ya) != (y1 > ya)
            denominator = y1 - y0
            x_crossing = (x1 - x0) * (ya - y0) / (
                denominator + (denominator == 0.0)
            ) + x0
            inside ^= crossing & (xa < x_crossing)
        # Include polygon edges; this matters when selecting Dirichlet nodes.
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, *(abs(v) for v in self.bounds))
        for index in range(len(vertices)):
            start = vertices[index - 1]
            end = vertices[index]
            delta = end - start
            length2 = float(np.dot(delta, delta))
            fraction = np.clip(
                ((xa - start[0]) * delta[0] + (ya - start[1]) * delta[1]) / length2,
                0.0,
                1.0,
            )
            distance2 = (xa - start[0] - fraction * delta[0]) ** 2 + (
                ya - start[1] - fraction * delta[1]
            ) ** 2
            inside |= distance2 <= tolerance**2
        return inside


@dataclass(frozen=True, slots=True)
class Permittivity:
    """Real symmetric relative-permittivity tensor."""

    tensor: tuple[tuple[float, ...], ...]

    @classmethod
    def from_input(
        cls,
        value: float | Sequence[float] | Sequence[Sequence[float]],
        dim: int,
    ) -> "Permittivity":
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            matrix = np.eye(dim) * float(array)
        elif array.shape == (dim,):
            matrix = np.diag(array)
        elif array.shape == (dim, dim):
            matrix = array
        else:
            raise GeometryError(
                f"permittivity must be a scalar, length-{dim} diagonal, or {dim}x{dim} tensor."
            )
        if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
            raise GeometryError("permittivity must be finite and symmetric.")
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.any(eigenvalues <= 0.0):
            raise GeometryError("permittivity must be positive definite.")
        return cls(tuple(tuple(float(value) for value in row) for row in matrix))

    @property
    def array(self) -> FloatArray:
        return np.asarray(self.tensor, dtype=float)

    @property
    def dk_scale(self) -> float:
        return float(np.sqrt(np.linalg.eigvalsh(self.array).max()))


@dataclass(frozen=True, slots=True)
class MaterialRegion:
    id: int
    name: str
    shape: Shape
    permittivity: Permittivity


@dataclass(frozen=True, slots=True)
class PotentialRegion:
    id: int
    name: str
    shape: Shape | str
    value: float


@dataclass(frozen=True, slots=True)
class ChargeRegion:
    id: int
    name: str
    shape: Shape
    density: float


class GeometryModel:
    """Mutable continuous scene shared by the public solver."""

    def __init__(
        self,
        dim: int,
        x_span: tuple[float, float],
        y_span: tuple[float, float] | None,
        background: Permittivity,
    ) -> None:
        self.dim = dim
        self.x_span = _span(x_span, "domain x")
        self.y_span = None if dim == 1 else _span(y_span or (), "domain y")
        self.background = background
        self.materials: list[MaterialRegion] = []
        self.potentials: list[PotentialRegion] = []
        self.charges: list[ChargeRegion] = []
        self.revision = 0
        self._next_id = 1
        self._listeners: list[Callable[[], None]] = []

    def add_change_listener(self, callback: Callable[[], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def _changed(self) -> None:
        self.revision += 1
        for callback in tuple(self._listeners):
            callback()

    def _new_name(self, requested: str | None, prefix: str) -> tuple[int, str]:
        identity = self._next_id
        self._next_id += 1
        name = requested or f"{prefix}_{identity}"
        names = {item.name for item in (*self.materials, *self.potentials, *self.charges)}
        if not name or name in names:
            raise GeometryError(f"geometry name {name!r} is empty or already used.")
        return identity, name

    def validate_shape(self, shape: Shape) -> None:
        if self.dim == 1:
            if not isinstance(shape, Interval):
                raise GeometryError("1D geometry requires Interval regions.")
            if shape.x[0] < self.x_span[0] or shape.x[1] > self.x_span[1]:
                raise GeometryError("interval lies outside the solver domain.")
            return
        if isinstance(shape, Interval):
            raise GeometryError("2D geometry requires Rectangle, Circle, or Polygon regions.")
        xmin, xmax, ymin, ymax = shape.bounds
        assert self.y_span is not None
        tolerance = 1e-13 * max(1.0, self.x_span[1] - self.x_span[0], self.y_span[1] - self.y_span[0])
        if (
            xmin < self.x_span[0] - tolerance
            or xmax > self.x_span[1] + tolerance
            or ymin < self.y_span[0] - tolerance
            or ymax > self.y_span[1] + tolerance
        ):
            raise GeometryError("shape lies outside the solver domain.")

    def add_material(
        self,
        shape: Shape,
        permittivity: Permittivity,
        *,
        name: str | None = None,
    ) -> MaterialRegion:
        self.validate_shape(shape)
        identity, actual_name = self._new_name(name, "material")
        region = MaterialRegion(identity, actual_name, shape, permittivity)
        self.materials.append(region)
        self._changed()
        return region

    def add_potential(
        self,
        shape: Shape | str,
        value: float,
        *,
        name: str | None = None,
    ) -> PotentialRegion:
        if isinstance(shape, str):
            allowed = {"left", "right"} if self.dim == 1 else {"left", "right", "bottom", "top", "outer"}
            if shape.lower() not in allowed:
                raise GeometryError(f"unknown boundary selector {shape!r}; expected one of {sorted(allowed)}.")
            shape = shape.lower()
        else:
            self.validate_shape(shape)
        numeric = float(value)
        if not np.isfinite(numeric):
            raise GeometryError("fixed potential must be finite.")
        identity, actual_name = self._new_name(name, "potential")
        region = PotentialRegion(identity, actual_name, shape, numeric)
        self.potentials.append(region)
        self._changed()
        return region

    def add_charge(
        self,
        shape: Shape,
        density: float,
        *,
        name: str | None = None,
    ) -> ChargeRegion:
        self.validate_shape(shape)
        numeric = float(density)
        if not np.isfinite(numeric):
            raise GeometryError("charge density must be finite.")
        identity, actual_name = self._new_name(name, "charge")
        region = ChargeRegion(identity, actual_name, shape, numeric)
        self.charges.append(region)
        self._changed()
        return region

    def remove(self, item: MaterialRegion | PotentialRegion | ChargeRegion) -> None:
        collection = (
            self.materials
            if isinstance(item, MaterialRegion)
            else self.potentials
            if isinstance(item, PotentialRegion)
            else self.charges
        )
        try:
            collection.remove(item)  # type: ignore[arg-type]
        except ValueError as exc:
            raise GeometryError("geometry handle does not belong to this model.") from exc
        self._changed()

    def all_area_shapes(self) -> tuple[Shape, ...]:
        return tuple(
            item.shape
            for item in (*self.materials, *self.potentials, *self.charges)
            if not isinstance(item.shape, str)
        )

    def material_indices_at(self, coordinates: FloatArray) -> NDArray[np.int32]:
        count = coordinates.shape[0]
        tags = np.ones(count, dtype=np.int32)
        for index, region in enumerate(self.materials, start=2):
            if self.dim == 1:
                mask = region.shape.contains(coordinates[:, 0])  # type: ignore[union-attr]
            else:
                mask = region.shape.contains(coordinates[:, 0], coordinates[:, 1])  # type: ignore[union-attr]
            tags[np.asarray(mask, dtype=bool)] = index
        return tags

    def charge_at(self, coordinates: FloatArray) -> FloatArray:
        values = np.zeros(coordinates.shape[0], dtype=float)
        for region in self.charges:
            if self.dim == 1:
                mask = region.shape.contains(coordinates[:, 0])  # type: ignore[union-attr]
            else:
                mask = region.shape.contains(coordinates[:, 0], coordinates[:, 1])  # type: ignore[union-attr]
            values[np.asarray(mask, dtype=bool)] = region.density
        return values

    @property
    def material_table(self) -> dict[int, Permittivity]:
        return {
            1: self.background,
            **{index: region.permittivity for index, region in enumerate(self.materials, start=2)},
        }


__all__ = [
    "ChargeRegion",
    "Circle",
    "GeometryModel",
    "Interval",
    "MaterialRegion",
    "Permittivity",
    "Polygon",
    "PotentialRegion",
    "Rectangle",
    "Shape",
]
