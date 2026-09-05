"""Continuous geometry for one periodic cell.

The two-dimensional solver uses physical ``(x, z)`` coordinates and is
periodic only in ``z``.  Three-dimensional primitives are kept here as shared
public data types for the companion 3D mesher.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import ConfigurationError, GeometryError
from .materials import Material


def physical_span(value: float | Sequence[float], name: str) -> tuple[float, float]:
    if np.isscalar(value):
        span = (0.0, float(value))  # type: ignore[arg-type]
    else:
        try:
            lower, upper = value
        except (TypeError, ValueError) as exc:
            raise GeometryError(f"{name} must be a positive extent or a two-value span.") from exc
        span = (float(lower), float(upper))
    if not np.isfinite(span).all() or span[1] <= span[0]:
        raise GeometryError(f"{name} must be finite and satisfy maximum > minimum.")
    return span


class Shape2D(Protocol):
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...

    def contains(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]: ...


class Shape3D(Protocol):
    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]: ...

    def contains(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]: ...


@dataclass(frozen=True, slots=True)
class Rectangle:
    x: tuple[float, float]
    z: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", physical_span(self.x, "rectangle x"))
        object.__setattr__(self, "z", physical_span(self.z, "rectangle z"))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x[0], self.x[1], self.z[0], self.z[1]

    def contains(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        xa, za = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        return (xa >= self.x[0]) & (xa <= self.x[1]) & (za >= self.z[0]) & (za <= self.z[1])


@dataclass(frozen=True, slots=True)
class Circle:
    center: tuple[float, float]
    radius: float
    inner_radius: float | None = None

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        radius = float(self.radius)
        inner = None if self.inner_radius is None else float(self.inner_radius)
        if len(center) != 2 or not np.isfinite(center).all():
            raise GeometryError("circle center must contain two finite coordinates.")
        if not isfinite(radius) or radius <= 0.0:
            raise GeometryError("circle radius must be finite and positive.")
        if inner is not None and (not isfinite(inner) or inner <= 0.0 or inner >= radius):
            raise GeometryError("circle inner_radius must satisfy 0 < inner_radius < radius.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "inner_radius", inner)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        x, z = self.center
        return x - self.radius, x + self.radius, z - self.radius, z + self.radius

    def contains(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        xa, za = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        squared = (xa - self.center[0]) ** 2 + (za - self.center[1]) ** 2
        result = squared <= self.radius**2
        if self.inner_radius is not None:
            result &= squared >= self.inner_radius**2
        return result


@dataclass(frozen=True, slots=True)
class Polygon:
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        points = tuple((float(x), float(z)) for x, z in self.points)
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
        points = np.asarray(self.points)
        return (
            float(points[:, 0].min()),
            float(points[:, 0].max()),
            float(points[:, 1].min()),
            float(points[:, 1].max()),
        )

    def contains(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        xa, za = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        inside = np.zeros(xa.shape, dtype=bool)
        points = np.asarray(self.points, dtype=float)
        for index in range(len(points)):
            x0, z0 = points[index - 1]
            x1, z1 = points[index]
            crossing = (z0 > za) != (z1 > za)
            denominator = z1 - z0
            x_crossing = (x1 - x0) * (za - z0) / (denominator + (denominator == 0.0)) + x0
            inside ^= crossing & (xa < x_crossing)
        return inside


@dataclass(frozen=True, slots=True)
class Box:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", physical_span(self.x, "box x"))
        object.__setattr__(self, "y", physical_span(self.y, "box y"))
        object.__setattr__(self, "z", physical_span(self.z, "box z"))

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return self.x[0], self.x[1], self.y[0], self.y[1], self.z[0], self.z[1]

    def contains(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        xa, ya, za = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
        return (
            (xa >= self.x[0]) & (xa <= self.x[1])
            & (ya >= self.y[0]) & (ya <= self.y[1])
            & (za >= self.z[0]) & (za <= self.z[1])
        )


@dataclass(frozen=True, slots=True)
class Cylinder:
    center: tuple[float, float]
    radius: float
    z: tuple[float, float]

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        radius = float(self.radius)
        if len(center) != 2 or not np.isfinite(center).all() or not isfinite(radius) or radius <= 0.0:
            raise GeometryError("cylinder requires a finite (x, y) center and positive radius.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "z", physical_span(self.z, "cylinder z"))

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        x, y = self.center
        return x - self.radius, x + self.radius, y - self.radius, y + self.radius, self.z[0], self.z[1]

    def contains(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        xa, ya, za = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
        return ((xa - self.center[0]) ** 2 + (ya - self.center[1]) ** 2 <= self.radius**2) & (za >= self.z[0]) & (za <= self.z[1])


@dataclass(frozen=True, slots=True)
class Sphere:
    center: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        radius = float(self.radius)
        if len(center) != 3 or not np.isfinite(center).all() or not isfinite(radius) or radius <= 0.0:
            raise GeometryError("sphere requires a finite center and positive radius.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        x, y, z = self.center
        r = self.radius
        return x - r, x + r, y - r, y + r, z - r, z + r

    def contains(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        xa, ya, za = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
        return (xa - self.center[0]) ** 2 + (ya - self.center[1]) ** 2 + (za - self.center[2]) ** 2 <= self.radius**2


@dataclass(frozen=True, slots=True)
class Region:
    id: int
    name: str
    shape: Shape2D | Shape3D
    material: Material


@dataclass(frozen=True, slots=True)
class BoundaryRegion:
    id: int
    name: str
    shape: Shape2D | Shape3D
    kind: str


@dataclass(frozen=True, slots=True)
class MeshRefinement:
    id: int
    name: str
    shape: Shape2D | Shape3D
    max_element_size: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_element_size) or self.max_element_size <= 0.0:
            raise GeometryError("max_element_size must be finite and positive.")


@dataclass(frozen=True, slots=True)
class PMLSpec:
    thickness: float
    order: int = 3
    sigma_max: float = 5.0
    direction: str = "x"

    def __post_init__(self) -> None:
        if not np.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ConfigurationError("PML thickness must be finite and positive.")
        if isinstance(self.order, bool) or int(self.order) != self.order or self.order < 1:
            raise ConfigurationError("PML order must be a positive integer.")
        if not np.isfinite(self.sigma_max) or self.sigma_max < 0.0:
            raise ConfigurationError("PML sigma_max must be finite and nonnegative.")
        if self.direction not in ("x-", "x+", "x", "y-", "y+", "y", "all"):
            raise ConfigurationError("invalid transverse PML direction.")

    def stretch(self, depth: ArrayLike) -> NDArray[np.complex128]:
        values = np.clip(np.asarray(depth, dtype=float), 0.0, self.thickness)
        return np.asarray(
            1.0 - 1j * self.sigma_max * (values / self.thickness) ** self.order,
            dtype=np.complex128,
        )


class _GeometryBase:
    def __init__(self, background: Material) -> None:
        self.background = background
        self.regions: list[Region] = []
        self.boundaries: list[BoundaryRegion] = []
        self.refinements: list[MeshRefinement] = []
        self.pmls: list[PMLSpec] = []
        self.outer_boundary = "pec"
        self.revision = 0
        self._next_id = 1
        self._listeners: list[Callable[[], None]] = []

    def add_change_listener(self, listener: Callable[[], None]) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def _changed(self) -> None:
        self.revision += 1
        for listener in tuple(self._listeners):
            listener()

    def _identity(self, name: str | None, prefix: str) -> tuple[int, str]:
        identifier = self._next_id
        self._next_id += 1
        selected = name or f"{prefix}_{identifier}"
        if not selected or any(item.name == selected for item in (*self.regions, *self.boundaries, *self.refinements)):
            raise GeometryError(f"Geometry name {selected!r} is empty or already used.")
        return identifier, selected

    def set_outer_boundary(self, kind: str) -> None:
        normalized = str(kind).strip().lower()
        if normalized not in ("pec", "pmc"):
            raise GeometryError("outer boundary must be 'pec' or 'pmc'.")
        if normalized != self.outer_boundary:
            self.outer_boundary = normalized
            self._changed()

    def remove(self, handle: Region | BoundaryRegion | MeshRefinement | PMLSpec) -> None:
        for collection in (self.regions, self.boundaries, self.refinements, self.pmls):
            if handle in collection:
                collection.remove(handle)  # type: ignore[arg-type]
                self._changed()
                return
        raise GeometryError("The geometry handle does not belong to this model.")


class GeometryModel2D(_GeometryBase):
    def __init__(self, x_span: float | Sequence[float], z_span: float | Sequence[float], background: Material) -> None:
        super().__init__(background)
        self.x_span = physical_span(x_span, "x_range")
        self.z_span = physical_span(z_span, "z_range")

    def _inside(self, bounds: tuple[float, float, float, float]) -> bool:
        return bounds[0] >= self.x_span[0] and bounds[1] <= self.x_span[1] and bounds[2] >= self.z_span[0] and bounds[3] <= self.z_span[1]

    def add_region(self, shape: Shape2D, material: Material, *, name: str | None = None) -> Region:
        if not self._inside(shape.bounds):
            raise GeometryError("Material region lies outside the periodic cell; split seam-crossing geometry explicitly.")
        identifier, selected = self._identity(name, "region")
        result = Region(identifier, selected, shape, material)
        self.regions.append(result)
        self._changed()
        return result

    def add_boundary(self, shape: Shape2D, kind: str, *, name: str | None = None) -> BoundaryRegion:
        normalized = str(kind).strip().lower()
        if normalized not in ("pec", "pmc"):
            raise GeometryError("internal boundary kind must be 'pec' or 'pmc'.")
        if not self._inside(shape.bounds):
            raise GeometryError("Boundary object lies outside the periodic cell.")
        identifier, selected = self._identity(name, normalized)
        result = BoundaryRegion(identifier, selected, shape, normalized)
        self.boundaries.append(result)
        self._changed()
        return result

    def add_mesh_refinement(self, shape: Shape2D, max_element_size: float, *, name: str | None = None) -> MeshRefinement:
        if not self._inside(shape.bounds):
            raise GeometryError("Mesh refinement lies outside the periodic cell.")
        identifier, selected = self._identity(name, "refinement")
        result = MeshRefinement(identifier, selected, shape, float(max_element_size))
        self.refinements.append(result)
        self._changed()
        return result

    def add_pml(self, spec: PMLSpec) -> PMLSpec:
        if spec.direction not in ("x-", "x+", "x"):
            raise GeometryError("A 2D x-z periodic cell permits PML only in x.")
        requested = {"x-", "x+"} if spec.direction == "x" else {spec.direction}
        occupied = {side for pml in self.pmls for side in (("x-", "x+") if pml.direction == "x" else (pml.direction,))}
        if requested & occupied:
            raise GeometryError("A PML already covers one of the requested x boundaries.")
        widths = {side: spec.thickness for side in requested}
        for pml in self.pmls:
            for side in (("x-", "x+") if pml.direction == "x" else (pml.direction,)):
                widths[side] = pml.thickness
        if widths.get("x-", 0.0) + widths.get("x+", 0.0) >= self.x_span[1] - self.x_span[0]:
            raise GeometryError("The x PMLs leave no physical interior.")
        self.pmls.append(spec)
        self._changed()
        return spec

    def material_at(self, x: ArrayLike, z: ArrayLike) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        xa, za = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        leading = (3, *([1] * xa.ndim))
        epsilon = np.broadcast_to(np.asarray(self.background.eps_r).reshape(leading), (3, *xa.shape)).copy()
        mu = np.broadcast_to(np.asarray(self.background.mu_r).reshape(leading), (3, *xa.shape)).copy()
        for region in self.regions:
            mask = region.shape.contains(xa, za)  # type: ignore[union-attr]
            epsilon[:, mask] = np.asarray(region.material.eps_r)[:, None]
            mu[:, mask] = np.asarray(region.material.mu_r)[:, None]
        return epsilon, mu

    def transformed_material_at(self, x: ArrayLike, z: ArrayLike) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        xa, za = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        epsilon, mu = self.material_at(xa, za)
        sx = np.ones(xa.shape, dtype=np.complex128)
        for pml in self.pmls:
            if pml.direction in ("x-", "x"):
                sx *= pml.stretch(np.maximum(self.x_span[0] + pml.thickness - xa, 0.0))
            if pml.direction in ("x+", "x"):
                sx *= pml.stretch(np.maximum(xa - (self.x_span[1] - pml.thickness), 0.0))
        factors = np.stack((1.0 / sx, sx, sx))
        return factors * epsilon, factors * mu

    def pml_mask(self, x: ArrayLike) -> NDArray[np.bool_]:
        values = np.asarray(x, float)
        result = np.zeros(values.shape, dtype=bool)
        for pml in self.pmls:
            if pml.direction in ("x-", "x"):
                result |= values < self.x_span[0] + pml.thickness
            if pml.direction in ("x+", "x"):
                result |= values > self.x_span[1] - pml.thickness
        return result


class GeometryModel3D(_GeometryBase):
    """Continuous ``x-y-z`` geometry for one cell periodic in ``z``."""

    def __init__(
        self,
        x_span: float | Sequence[float],
        y_span: float | Sequence[float],
        z_span: float | Sequence[float],
        background: Material,
    ) -> None:
        super().__init__(background)
        self.x_span = physical_span(x_span, "x_range")
        self.y_span = physical_span(y_span, "y_range")
        self.z_span = physical_span(z_span, "z_range")

    def _inside(
        self, bounds: tuple[float, float, float, float, float, float]
    ) -> bool:
        return (
            bounds[0] >= self.x_span[0]
            and bounds[1] <= self.x_span[1]
            and bounds[2] >= self.y_span[0]
            and bounds[3] <= self.y_span[1]
            and bounds[4] >= self.z_span[0]
            and bounds[5] <= self.z_span[1]
        )

    def add_region(
        self, shape: Shape3D, material: Material, *, name: str | None = None
    ) -> Region:
        if not self._inside(shape.bounds):
            raise GeometryError(
                "Material region lies outside the periodic cell; split "
                "seam-crossing geometry explicitly."
            )
        identifier, selected = self._identity(name, "region")
        result = Region(identifier, selected, shape, material)
        self.regions.append(result)
        self._changed()
        return result

    def add_boundary(
        self, shape: Shape3D, kind: str, *, name: str | None = None
    ) -> BoundaryRegion:
        normalized = str(kind).strip().lower()
        if normalized not in ("pec", "pmc"):
            raise GeometryError("internal boundary kind must be 'pec' or 'pmc'.")
        if not self._inside(shape.bounds):
            raise GeometryError("Boundary object lies outside the periodic cell.")
        identifier, selected = self._identity(name, normalized)
        result = BoundaryRegion(identifier, selected, shape, normalized)
        self.boundaries.append(result)
        self._changed()
        return result

    def add_mesh_refinement(
        self,
        shape: Shape3D,
        max_element_size: float,
        *,
        name: str | None = None,
    ) -> MeshRefinement:
        if not self._inside(shape.bounds):
            raise GeometryError("Mesh refinement lies outside the periodic cell.")
        identifier, selected = self._identity(name, "refinement")
        result = MeshRefinement(
            identifier, selected, shape, float(max_element_size)
        )
        self.refinements.append(result)
        self._changed()
        return result

    @staticmethod
    def _pml_sides(direction: str) -> set[str]:
        if direction == "x":
            return {"x-", "x+"}
        if direction == "y":
            return {"y-", "y+"}
        if direction == "all":
            return {"x-", "x+", "y-", "y+"}
        return {direction}

    def add_pml(self, spec: PMLSpec) -> PMLSpec:
        requested = self._pml_sides(spec.direction)
        occupied = {
            side for pml in self.pmls for side in self._pml_sides(pml.direction)
        }
        if requested & occupied:
            raise GeometryError("A PML already covers one of the requested boundaries.")
        widths = {side: spec.thickness for side in requested}
        for pml in self.pmls:
            for side in self._pml_sides(pml.direction):
                widths[side] = pml.thickness
        if widths.get("x-", 0.0) + widths.get("x+", 0.0) >= (
            self.x_span[1] - self.x_span[0]
        ):
            raise GeometryError("The x PMLs leave no physical interior.")
        if widths.get("y-", 0.0) + widths.get("y+", 0.0) >= (
            self.y_span[1] - self.y_span[0]
        ):
            raise GeometryError("The y PMLs leave no physical interior.")
        self.pmls.append(spec)
        self._changed()
        return spec

    def material_at(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        xa, ya, za = np.broadcast_arrays(
            np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        )
        leading = (3, *([1] * xa.ndim))
        epsilon = np.broadcast_to(
            np.asarray(self.background.eps_r).reshape(leading), (3, *xa.shape)
        ).copy()
        mu = np.broadcast_to(
            np.asarray(self.background.mu_r).reshape(leading), (3, *xa.shape)
        ).copy()
        for region in self.regions:
            mask = region.shape.contains(xa, ya, za)  # type: ignore[union-attr]
            epsilon[:, mask] = np.asarray(region.material.eps_r)[:, None]
            mu[:, mask] = np.asarray(region.material.mu_r)[:, None]
        return epsilon, mu

    def transformed_material_at(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        xa, ya, za = np.broadcast_arrays(
            np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        )
        epsilon, mu = self.material_at(xa, ya, za)
        sx = np.ones(xa.shape, dtype=np.complex128)
        sy = np.ones(xa.shape, dtype=np.complex128)
        for pml in self.pmls:
            sides = self._pml_sides(pml.direction)
            if "x-" in sides:
                sx *= pml.stretch(
                    np.maximum(self.x_span[0] + pml.thickness - xa, 0.0)
                )
            if "x+" in sides:
                sx *= pml.stretch(
                    np.maximum(xa - (self.x_span[1] - pml.thickness), 0.0)
                )
            if "y-" in sides:
                sy *= pml.stretch(
                    np.maximum(self.y_span[0] + pml.thickness - ya, 0.0)
                )
            if "y+" in sides:
                sy *= pml.stretch(
                    np.maximum(ya - (self.y_span[1] - pml.thickness), 0.0)
                )
        factors = np.stack((sy / sx, sx / sy, sx * sy))
        return factors * epsilon, factors * mu

    def pml_mask(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]:
        xa, ya = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        result = np.zeros(xa.shape, dtype=bool)
        for pml in self.pmls:
            sides = self._pml_sides(pml.direction)
            if "x-" in sides:
                result |= xa < self.x_span[0] + pml.thickness
            if "x+" in sides:
                result |= xa > self.x_span[1] - pml.thickness
            if "y-" in sides:
                result |= ya < self.y_span[0] + pml.thickness
            if "y+" in sides:
                result |= ya > self.y_span[1] - pml.thickness
        return result


__all__ = [
    "BoundaryRegion", "Box", "Circle", "Cylinder", "GeometryModel2D",
    "GeometryModel3D",
    "MeshRefinement", "PMLSpec", "Polygon", "Rectangle", "Region",
    "Shape2D", "Shape3D", "Sphere", "physical_span",
]
