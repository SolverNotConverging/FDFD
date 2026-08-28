"""Continuous geometry recorded before FEM discretization."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from collections.abc import Callable
from typing import Protocol, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .boundaries import validate_surface_impedance
from .exceptions import ConfigurationError, GeometryError
from .materials import Material, MaterialInput


FloatArray: TypeAlias = NDArray[np.float64]


_PML_SIDES: dict[str, frozenset[str]] = {
    "x-": frozenset(("x-",)),
    "x+": frozenset(("x+",)),
    "x": frozenset(("x-", "x+")),
    "y-": frozenset(("y-",)),
    "y+": frozenset(("y+",)),
    "y": frozenset(("y-", "y+")),
    "all": frozenset(("x-", "x+", "y-", "y+")),
}

_RESERVED_BOUNDARY_NAMES = frozenset(
    ("pec", "pmc", "impedance", "outer_pec", "outer_pmc")
)


def _pml_sides(direction: str) -> frozenset[str]:
    """Expand a compact PML direction into the exterior sides it owns."""

    return _PML_SIDES[direction]


def physical_span(value: float | Sequence[float], name: str) -> tuple[float, float]:
    """Return a finite increasing physical span.

    A positive scalar denotes a domain from zero to that value, matching the
    older mode-solver constructors.  Geometry primitives should use explicit
    ``(minimum, maximum)`` pairs.
    """

    if np.isscalar(value):
        width = float(value)  # type: ignore[arg-type]
        result = (0.0, width)
    else:
        try:
            left, right = value
        except (TypeError, ValueError) as exc:
            raise GeometryError(f"{name} must be a positive width or a (minimum, maximum) pair.") from exc
        result = (float(left), float(right))
    if not np.isfinite(result).all() or result[1] <= result[0]:
        raise GeometryError(f"{name} must be finite and satisfy maximum > minimum.")
    return result


class Shape2D(Protocol):
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...

    def contains(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]: ...


@dataclass(frozen=True, slots=True)
class Interval:
    x: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", physical_span(self.x, "interval x"))

    def contains(self, x: ArrayLike) -> NDArray[np.bool_]:
        values = np.asarray(x, dtype=float)
        return (values >= self.x[0]) & (values <= self.x[1])


@dataclass(frozen=True, slots=True)
class Rectangle:
    x: tuple[float, float]
    y: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", physical_span(self.x, "rectangle x"))
        object.__setattr__(self, "y", physical_span(self.y, "rectangle y"))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x[0], self.x[1], self.y[0], self.y[1]

    def contains(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return (xa >= self.x[0]) & (xa <= self.x[1]) & (ya >= self.y[0]) & (ya <= self.y[1])


@dataclass(frozen=True, slots=True)
class Circle:
    center: tuple[float, float]
    radius: float
    inner_radius: float | None = None

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        if len(center) != 2 or not np.isfinite(center).all():
            raise GeometryError("circle center must contain two finite physical coordinates.")
        radius = float(self.radius)
        inner = None if self.inner_radius is None else float(self.inner_radius)
        if not isfinite(radius) or radius <= 0.0:
            raise GeometryError("circle radius must be finite and positive.")
        if inner is not None and (not isfinite(inner) or inner <= 0.0 or inner >= radius):
            raise GeometryError("circle inner_radius must satisfy 0 < inner_radius < radius.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "inner_radius", inner)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        return cx - self.radius, cx + self.radius, cy - self.radius, cy + self.radius

    def contains(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        squared = (xa - self.center[0]) ** 2 + (ya - self.center[1]) ** 2
        result = squared <= self.radius**2
        if self.inner_radius is not None:
            result &= squared >= self.inner_radius**2
        return result


@dataclass(frozen=True, slots=True)
class Polygon:
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        points = tuple((float(x), float(y)) for x, y in self.points)
        if len(points) < 3 or not np.isfinite(points).all():
            raise GeometryError("polygon requires at least three finite points.")
        area = 0.5 * sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
        if abs(area) <= np.finfo(float).eps:
            raise GeometryError("polygon points must enclose a nonzero area.")
        object.__setattr__(self, "points", points)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        coordinates = np.asarray(self.points)
        return (
            float(coordinates[:, 0].min()),
            float(coordinates[:, 0].max()),
            float(coordinates[:, 1].min()),
            float(coordinates[:, 1].max()),
        )

    def contains(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.bool_]:
        # Vectorized even/odd ray crossing.  Boundary tolerance is handled by
        # the conforming mesh, so this evaluator is chiefly for material tags.
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        inside = np.zeros(xa.shape, dtype=bool)
        vertices = np.asarray(self.points, dtype=float)
        for index in range(len(vertices)):
            x0, y0 = vertices[index - 1]
            x1, y1 = vertices[index]
            crossing = (y0 > ya) != (y1 > ya)
            denominator = y1 - y0
            x_crossing = (x1 - x0) * (ya - y0) / (denominator + (denominator == 0.0)) + x0
            inside ^= crossing & (xa < x_crossing)
        return inside


@dataclass(frozen=True, slots=True)
class Region:
    id: int
    name: str
    shape: Interval | Shape2D
    material: Material


@dataclass(frozen=True, slots=True)
class BoundaryRegion:
    id: int
    name: str
    shape: Interval | Shape2D
    kind: str
    impedance: complex | None = None


@dataclass(frozen=True, slots=True)
class MeshRefinement:
    """A geometry-only 2D mesh-size control.

    Refinement regions participate in OCC fragmentation so their boundary is
    represented exactly, but they never change material or boundary tags.
    ``transition_width`` controls the physical distance over which Gmsh grades
    from ``max_element_size`` back to the surrounding target size.
    """

    id: int
    name: str
    shape: Shape2D
    max_element_size: float
    transition_width: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_element_size, (bool, np.bool_)):
            raise GeometryError("refinement max_element_size must be finite and positive.")
        if isinstance(self.transition_width, (bool, np.bool_)):
            raise GeometryError("refinement transition_width must be finite and nonnegative.")
        maximum = float(self.max_element_size)
        transition = float(self.transition_width)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise GeometryError("refinement max_element_size must be finite and positive.")
        if not np.isfinite(transition) or transition < 0.0:
            raise GeometryError("refinement transition_width must be finite and nonnegative.")
        object.__setattr__(self, "max_element_size", maximum)
        object.__setattr__(self, "transition_width", transition)


@dataclass(frozen=True, slots=True)
class PMLSpec:
    thickness: float
    order: int = 3
    sigma_max: float = 5.0
    direction: str = "all"

    def __post_init__(self) -> None:
        if not np.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ConfigurationError("PML thickness must be finite and positive.")
        if isinstance(self.order, bool) or int(self.order) != self.order or self.order < 1:
            raise ConfigurationError("PML order must be a positive integer.")
        if not np.isfinite(self.sigma_max) or self.sigma_max < 0.0:
            raise ConfigurationError("PML sigma_max must be finite and nonnegative.")
        if self.direction not in ("x-", "x+", "x", "y-", "y+", "y", "all"):
            raise ConfigurationError(
                "PML direction must be 'x-', 'x+', 'x', 'y-', 'y+', 'y', or 'all'."
            )

    def stretch(self, depth: ArrayLike) -> NDArray[np.complex128]:
        clipped = np.clip(np.asarray(depth, dtype=float), 0.0, self.thickness)
        return np.asarray(
            1.0 - 1j * self.sigma_max * (clipped / self.thickness) ** self.order,
            dtype=np.complex128,
        )


class _GeometryBase:
    def __init__(self) -> None:
        self.regions: list[Region] = []
        self.boundaries: list[BoundaryRegion] = []
        self.pmls: list[PMLSpec] = []
        self.revision = 0
        self._next_id = 1
        self._change_listeners: list[Callable[[], None]] = []

    def add_change_listener(self, callback: Callable[[], None]) -> None:
        """Notify an owning solver whenever the continuous scene changes."""

        if not callable(callback):
            raise TypeError("geometry change listener must be callable.")
        if callback not in self._change_listeners:
            self._change_listeners.append(callback)

    def _name(self, requested: str | None, prefix: str) -> str:
        name = requested or f"{prefix}_{self._next_id}"
        refinements = getattr(self, "refinements", ())
        if not name or any(
            item.name == name for item in (*self.regions, *self.boundaries, *refinements)
        ):
            raise GeometryError(f"Geometry name {name!r} is empty or already in use.")
        return name

    def _id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def _changed(self) -> None:
        self.revision += 1
        for callback in tuple(self._change_listeners):
            callback()

    def remove(self, item: Region | BoundaryRegion | MeshRefinement) -> None:
        if isinstance(item, Region):
            collection = self.regions
        elif isinstance(item, BoundaryRegion):
            collection = self.boundaries
        else:
            collection = getattr(self, "refinements", [])
        try:
            collection.remove(item)  # type: ignore[arg-type]
        except ValueError as exc:
            raise GeometryError("The geometry handle does not belong to this model.") from exc
        self._changed()

    def add_pml(self, spec: PMLSpec) -> PMLSpec:
        self.pmls.append(spec)
        self._changed()
        return spec

    def set_outer_boundary(self, kind: str) -> None:
        normalized = str(kind).lower()
        if normalized not in ("pec", "pmc"):
            raise GeometryError("outer boundary must be 'pec' or 'pmc'.")
        if self.outer_boundary != normalized:
            self.outer_boundary = normalized
            self._changed()


class GeometryModel1D(_GeometryBase):
    def __init__(self, x_span: float | Sequence[float], background: Material) -> None:
        super().__init__()
        self.x_span = physical_span(x_span, "x_range")
        self.background = background
        self.outer_boundary = "pec"

    def add_region(self, interval: Interval, material: Material, *, name: str | None = None) -> Region:
        if interval.x[0] < self.x_span[0] or interval.x[1] > self.x_span[1]:
            raise GeometryError("Layer lies outside the solver domain.")
        region = Region(self._id(), self._name(name, "layer"), interval, material)
        self.regions.append(region)
        self._changed()
        return region

    def add_pml(self, spec: PMLSpec) -> PMLSpec:
        if spec.direction not in ("x-", "x+", "x", "all"):
            raise GeometryError("A 1D mode PML only accepts x directions.")
        width = self.x_span[1] - self.x_span[0]
        requested = _pml_sides(spec.direction) & {"x-", "x+"}
        covered = {
            side
            for existing in self.pmls
            for side in (_pml_sides(existing.direction) & {"x-", "x+"})
        }
        repeated = requested & covered
        if repeated:
            names = ", ".join(sorted(repeated))
            raise GeometryError(f"A PML already covers the 1D exterior side(s): {names}.")

        widths = {
            side: existing.thickness
            for existing in self.pmls
            for side in (_pml_sides(existing.direction) & {"x-", "x+"})
        }
        widths.update({side: spec.thickness for side in requested})
        if widths.get("x-", 0.0) + widths.get("x+", 0.0) >= width:
            raise GeometryError("The PML leaves no non-PML 1D interior.")
        return super().add_pml(spec)

    def add_boundary(self, interval: Interval, kind: str, *, impedance: complex | None = None, name: str | None = None) -> BoundaryRegion:
        normalized = str(kind).strip().lower()
        if normalized not in ("pec", "pmc", "impedance"):
            raise GeometryError("boundary kind must be 'pec', 'pmc', or 'impedance'.")
        if interval.x[0] < self.x_span[0] or interval.x[1] > self.x_span[1]:
            raise GeometryError("Boundary interval lies outside the solver domain.")
        if name in _RESERVED_BOUNDARY_NAMES:
            raise GeometryError(f"Boundary name {name!r} is reserved for mesh tags.")
        if normalized == "impedance":
            if impedance is None:
                raise GeometryError("An impedance boundary requires a surface impedance.")
            impedance = validate_surface_impedance(impedance)
        elif impedance is not None:
            raise GeometryError("Only an impedance boundary may define surface impedance.")
        boundary = BoundaryRegion(
            self._id(), self._name(name, normalized), interval, normalized, impedance
        )
        self.boundaries.append(boundary)
        self._changed()
        return boundary

    def material_at(self, x: ArrayLike) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        coordinates = np.asarray(x, dtype=float)
        eps = np.broadcast_to(np.asarray(self.background.eps_r)[:, None], (3, coordinates.size)).copy()
        mu = np.broadcast_to(np.asarray(self.background.mu_r)[:, None], (3, coordinates.size)).copy()
        flat = coordinates.ravel()
        for region in self.regions:
            mask = region.shape.contains(flat)  # type: ignore[union-attr]
            eps[:, mask] = np.asarray(region.material.eps_r)[:, None]
            mu[:, mask] = np.asarray(region.material.mu_r)[:, None]
        return eps.reshape((3, *coordinates.shape)), mu.reshape((3, *coordinates.shape))

    def transformed_material_at(self, x: ArrayLike) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        coordinates = np.asarray(x, dtype=float)
        eps, mu = self.material_at(coordinates)
        sx = np.ones(coordinates.shape, dtype=np.complex128)
        for pml in self.pmls:
            if pml.direction in ("x-", "x", "all"):
                sx *= pml.stretch(
                    np.maximum(self.x_span[0] + pml.thickness - coordinates, 0.0)
                )
            if pml.direction in ("x+", "x", "all"):
                sx *= pml.stretch(
                    np.maximum(coordinates - (self.x_span[1] - pml.thickness), 0.0)
                )
        factors = np.stack((1.0 / sx, sx, sx))
        return factors * eps, factors * mu


class GeometryModel2D(_GeometryBase):
    def __init__(
        self,
        x_span: float | Sequence[float],
        y_span: float | Sequence[float],
        background: Material,
    ) -> None:
        super().__init__()
        self.x_span = physical_span(x_span, "x_range")
        self.y_span = physical_span(y_span, "y_range")
        self.background = background
        self.outer_boundary = "pec"
        self.refinements: list[MeshRefinement] = []

    def add_region(self, shape: Shape2D, material: Material, *, name: str | None = None) -> Region:
        xmin, xmax, ymin, ymax = shape.bounds
        if xmin < self.x_span[0] or xmax > self.x_span[1] or ymin < self.y_span[0] or ymax > self.y_span[1]:
            raise GeometryError("Material region lies outside the solver domain.")
        region = Region(self._id(), self._name(name, "region"), shape, material)
        self.regions.append(region)
        self._changed()
        return region

    def add_pml(self, spec: PMLSpec) -> PMLSpec:
        xwidth = self.x_span[1] - self.x_span[0]
        ywidth = self.y_span[1] - self.y_span[0]
        requested = _pml_sides(spec.direction)
        covered = {
            side for existing in self.pmls for side in _pml_sides(existing.direction)
        }
        repeated = requested & covered
        if repeated:
            names = ", ".join(sorted(repeated))
            raise GeometryError(f"A PML already covers the exterior side(s): {names}.")

        widths = {
            side: existing.thickness
            for existing in self.pmls
            for side in _pml_sides(existing.direction)
        }
        widths.update({side: spec.thickness for side in requested})
        if widths.get("x-", 0.0) + widths.get("x+", 0.0) >= xwidth:
            raise GeometryError("The x-directed PMLs leave no interior.")
        if widths.get("y-", 0.0) + widths.get("y+", 0.0) >= ywidth:
            raise GeometryError("The y-directed PMLs leave no interior.")
        return super().add_pml(spec)

    def add_boundary(self, shape: Shape2D, kind: str, *, impedance: complex | None = None, name: str | None = None) -> BoundaryRegion:
        normalized = str(kind).strip().lower()
        if normalized not in ("pec", "pmc", "impedance"):
            raise GeometryError("boundary kind must be 'pec', 'pmc', or 'impedance'.")
        xmin, xmax, ymin, ymax = shape.bounds
        if (
            xmin < self.x_span[0]
            or xmax > self.x_span[1]
            or ymin < self.y_span[0]
            or ymax > self.y_span[1]
        ):
            raise GeometryError("Boundary region lies outside the solver domain.")
        if name in _RESERVED_BOUNDARY_NAMES:
            raise GeometryError(f"Boundary name {name!r} is reserved for mesh tags.")
        if normalized == "impedance":
            if impedance is None:
                raise GeometryError("An impedance boundary requires a surface impedance.")
            impedance = validate_surface_impedance(impedance)
        elif impedance is not None:
            raise GeometryError("Only an impedance boundary may define surface impedance.")
        boundary = BoundaryRegion(
            self._id(), self._name(name, normalized), shape, normalized, impedance
        )
        self.boundaries.append(boundary)
        self._changed()
        return boundary

    def add_mesh_refinement(
        self,
        shape: Shape2D,
        max_element_size: float,
        *,
        transition_width: float = 0.0,
        name: str | None = None,
    ) -> MeshRefinement:
        """Add a non-physical local mesh-size region to the continuous scene."""

        xmin, xmax, ymin, ymax = shape.bounds
        if (
            xmin < self.x_span[0]
            or xmax > self.x_span[1]
            or ymin < self.y_span[0]
            or ymax > self.y_span[1]
        ):
            raise GeometryError("Mesh refinement region lies outside the solver domain.")
        refinement = MeshRefinement(
            self._id(),
            self._name(name, "mesh_refinement"),
            shape,
            max_element_size,
            transition_width,
        )
        self.refinements.append(refinement)
        self._changed()
        return refinement

    def material_at(self, x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        eps = np.broadcast_to(np.asarray(self.background.eps_r).reshape(3, *([1] * xa.ndim)), (3, *xa.shape)).copy()
        mu = np.broadcast_to(np.asarray(self.background.mu_r).reshape(3, *([1] * xa.ndim)), (3, *xa.shape)).copy()
        for region in self.regions:
            mask = region.shape.contains(xa, ya)  # type: ignore[union-attr]
            eps[:, mask] = np.asarray(region.material.eps_r)[:, None]
            mu[:, mask] = np.asarray(region.material.mu_r)[:, None]
        return eps, mu

    def transformed_material_at(self, x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        eps, mu = self.material_at(xa, ya)
        sx = np.ones(xa.shape, dtype=np.complex128)
        sy = np.ones(ya.shape, dtype=np.complex128)
        for pml in self.pmls:
            direction = pml.direction
            if direction in ("x-", "x", "all"):
                sx *= pml.stretch(np.maximum(self.x_span[0] + pml.thickness - xa, 0.0))
            if direction in ("x+", "x", "all"):
                sx *= pml.stretch(np.maximum(xa - (self.x_span[1] - pml.thickness), 0.0))
            if direction in ("y-", "y", "all"):
                sy *= pml.stretch(np.maximum(self.y_span[0] + pml.thickness - ya, 0.0))
            if direction in ("y+", "y", "all"):
                sy *= pml.stretch(np.maximum(ya - (self.y_span[1] - pml.thickness), 0.0))
        factors = np.stack((sy / sx, sx / sy, sx * sy))
        return factors * eps, factors * mu

    def pml_interfaces(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        xs: set[float] = set()
        ys: set[float] = set()
        for pml in self.pmls:
            if pml.direction in ("x-", "x", "all"):
                xs.add(self.x_span[0] + pml.thickness)
            if pml.direction in ("x+", "x", "all"):
                xs.add(self.x_span[1] - pml.thickness)
            if pml.direction in ("y-", "y", "all"):
                ys.add(self.y_span[0] + pml.thickness)
            if pml.direction in ("y+", "y", "all"):
                ys.add(self.y_span[1] - pml.thickness)
        return tuple(sorted(xs)), tuple(sorted(ys))


def material(epsilon: MaterialInput, mu: MaterialInput) -> Material:
    """Small convenience used by the public placement methods."""

    return Material(epsilon, mu)


__all__ = [
    "BoundaryRegion",
    "Circle",
    "GeometryModel1D",
    "GeometryModel2D",
    "Interval",
    "MeshRefinement",
    "PMLSpec",
    "Polygon",
    "Rectangle",
    "Region",
    "material",
    "physical_span",
]
