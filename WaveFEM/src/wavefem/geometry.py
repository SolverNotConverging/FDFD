"""Geometry and material-region bookkeeping for :mod:`wavefem`.

The geometry layer deliberately keeps the *actual* device material separate
from the z-invariant background guide.  The scattered-field source is formed
from their difference, so losing that distinction would be a physics error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import ConfigurationError
from .materials import Material


FloatArray = NDArray[np.float64]
ZExtent = tuple[float, float] | Literal["all"]


def _span(value: Sequence[float], label: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ConfigurationError(f"{label} must contain exactly two values.")
    lo, hi = float(value[0]), float(value[1])
    if not np.isfinite([lo, hi]).all() or not lo < hi:
        raise ConfigurationError(
            f"{label} must be finite and strictly increasing; got {value!r}."
        )
    return lo, hi


def _coerce_material(value: Material | complex | float) -> Material:
    return value if isinstance(value, Material) else Material(eps_r=value)


@dataclass(frozen=True, slots=True)
class Rectangle:
    """Axis-aligned material rectangle in the x-z solve plane."""

    x: tuple[float, float]
    z: tuple[float, float]

    def contains(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        xa, za = np.asarray(x), np.asarray(z)
        return (
            (xa >= self.x[0])
            & (xa <= self.x[1])
            & (za >= self.z[0])
            & (za <= self.z[1])
        )


@dataclass(frozen=True, slots=True)
class Circle:
    """Circular material region in the x-z solve plane."""

    center: tuple[float, float]
    radius: float

    def contains(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        xa, za = np.asarray(x), np.asarray(z)
        return (xa - self.center[0]) ** 2 + (za - self.center[1]) ** 2 <= self.radius**2


@dataclass(frozen=True, slots=True)
class Polygon:
    """Simple polygon represented by ordered ``(x, z)`` vertices."""

    points: tuple[tuple[float, float], ...]

    def contains(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        # Vectorized even-odd ray test.  Gmsh makes the interface conforming;
        # this predicate is used only to assign quadrature-point materials.
        xa, za = np.broadcast_arrays(np.asarray(x), np.asarray(z))
        inside = np.zeros(xa.shape, dtype=bool)
        xj, zj = self.points[-1]
        for xi, zi in self.points:
            crosses = (zi > za) != (zj > za)
            x_cross = (xj - xi) * (za - zi) / (zj - zi + np.finfo(float).tiny) + xi
            inside ^= crosses & (xa < x_cross)
            xj, zj = xi, zi
        return inside


Shape = Rectangle | Circle | Polygon


@dataclass(frozen=True, slots=True)
class Region:
    """A named shape/material assignment.

    ``background=True`` means the region belongs to the unperturbed straight
    guide and therefore contributes to both actual and background profiles.
    Other regions modify only the actual device, in insertion order.
    """

    name: str
    shape: Shape
    material: Material
    background: bool
    physical_tag: int

    def contains(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        return self.shape.contains(x, z)


@dataclass(slots=True)
class GeometryModel:
    """Solve-domain geometry with explicit actual/background profiles."""

    x_span: tuple[float, float]
    z_span: tuple[float, float]
    exterior: Material
    regions: list[Region] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x_span = _span(self.x_span, "x_span")
        self.z_span = _span(self.z_span, "z_span")
        self.exterior = _coerce_material(self.exterior)

    def _next_name(self, prefix: str, name: str | None) -> str:
        candidate = name or f"{prefix}_{len(self.regions) + 1}"
        if not candidate or any(region.name == candidate for region in self.regions):
            raise ConfigurationError(f"Material-region name {candidate!r} is not unique.")
        return candidate

    def _inside_domain(self, x: tuple[float, float], z: tuple[float, float]) -> None:
        tol = 64.0 * np.finfo(float).eps * max(
            1.0, *(abs(v) for v in (*self.x_span, *self.z_span))
        )
        if (
            x[0] < self.x_span[0] - tol
            or x[1] > self.x_span[1] + tol
            or z[0] < self.z_span[0] - tol
            or z[1] > self.z_span[1] + tol
        ):
            raise ConfigurationError(
                f"Region bounds x={x}, z={z} extend outside domain "
                f"x={self.x_span}, z={self.z_span}."
            )

    def add_rectangle(
        self,
        *,
        x: Sequence[float],
        z: ZExtent,
        material: Material | complex | float,
        background: bool = False,
        name: str | None = None,
    ) -> Region:
        """Add an axis-aligned rectangle.

        Background-guide regions must use ``z="all"`` so the material is
        invariant along the nominal propagation direction.
        """

        xs = _span(x, "rectangle x")
        if background and z != "all":
            raise ConfigurationError(
                "A background-guide region must be z-invariant; use z='all'."
            )
        zs = self.z_span if z == "all" else _span(z, "rectangle z")
        self._inside_domain(xs, zs)
        return self._append(
            self._next_name("rectangle", name),
            Rectangle(xs, zs),
            material,
            background,
        )

    def add_circle(
        self,
        *,
        center: Sequence[float],
        radius: float,
        material: Material | complex | float,
        background: bool = False,
        name: str | None = None,
    ) -> Region:
        """Add a circle; compact circles cannot define a straight guide."""

        if background:
            raise ConfigurationError(
                "A finite circle is not z-invariant and cannot be a background guide."
            )
        if len(center) != 2 or not np.isfinite(center).all():
            raise ConfigurationError("circle center must be two finite values (x, z).")
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ConfigurationError("circle radius must be finite and positive.")
        cx, cz = float(center[0]), float(center[1])
        self._inside_domain((cx - radius, cx + radius), (cz - radius, cz + radius))
        return self._append(
            self._next_name("circle", name),
            Circle((cx, cz), radius),
            material,
            False,
        )

    def add_polygon(
        self,
        *,
        points: Iterable[Sequence[float]],
        material: Material | complex | float,
        background: bool = False,
        name: str | None = None,
    ) -> Region:
        """Add a simple polygon; compact polygons modify the actual device."""

        if background:
            raise ConfigurationError(
                "A finite polygon is not z-invariant and cannot be a background guide."
            )
        pts = tuple((float(p[0]), float(p[1])) for p in points)
        if len(pts) < 3 or not np.isfinite(pts).all():
            raise ConfigurationError("polygon requires at least three finite (x, z) points.")
        xs, zs = zip(*pts, strict=True)
        self._inside_domain((min(xs), max(xs)), (min(zs), max(zs)))
        return self._append(
            self._next_name("polygon", name), Polygon(pts), material, False
        )

    def _append(
        self,
        name: str,
        shape: Shape,
        material: Material | complex | float,
        background: bool,
    ) -> Region:
        # 1 is reserved for the exterior; insertion order makes tags stable.
        region = Region(name, shape, _coerce_material(material), background, len(self.regions) + 2)
        self.regions.append(region)
        return region

    @property
    def background_regions(self) -> tuple[Region, ...]:
        return tuple(region for region in self.regions if region.background)

    @property
    def perturbations(self) -> tuple[Region, ...]:
        return tuple(region for region in self.regions if not region.background)

    def material_at(
        self, x: ArrayLike, z: ArrayLike, *, profile: Literal["actual", "background"]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Evaluate scalar ``(eps_r, mu_r)`` arrays at arbitrary points."""

        xa, za = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
        eps = np.full(xa.shape, complex(self.exterior.eps_r), dtype=np.complex128)
        mu = np.full(xa.shape, complex(self.exterior.mu_r), dtype=np.complex128)
        selected = self.background_regions
        if profile == "actual":
            selected += self.perturbations
        elif profile != "background":
            raise ValueError("profile must be 'actual' or 'background'.")
        for region in selected:
            mask = region.contains(xa, za)
            eps[mask] = complex(region.material.eps_r)
            mu[mask] = complex(region.material.mu_r)
        return eps, mu

    def region_tag_at(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.int32]:
        """Return stable actual-material physical tags at the points."""

        xa, za = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
        tags = np.ones(xa.shape, dtype=np.int32)
        for region in (*self.background_regions, *self.perturbations):
            tags[region.contains(xa, za)] = region.physical_tag
        return tags

    @property
    def physical_names(self) -> dict[int, str]:
        return {1: "exterior", **{r.physical_tag: r.name for r in self.regions}}
