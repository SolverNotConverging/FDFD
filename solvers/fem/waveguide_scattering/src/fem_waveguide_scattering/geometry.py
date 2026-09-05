"""Geometry and material-region bookkeeping for :mod:`fem_waveguide_scattering`.

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
Profile = Literal["actual", "background"]


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


@dataclass(frozen=True, slots=True)
class PECSheet:
    """An ideal zero-thickness PEC sheet parallel to the z axis.

    A background sheet belongs to the unperturbed guide and, before slots are
    cut, to the actual device as well.  Background sheets must span the whole
    solve domain in z so that the lead eigenproblem remains invariant.
    """

    name: str
    x: float
    z: tuple[float, float]
    background: bool


@dataclass(frozen=True, slots=True)
class PECSlot:
    """A finite actual-device opening cut from one background PEC sheet."""

    name: str
    sheet_name: str
    z: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PECSegment:
    """One closed PEC line segment in an actual or background profile."""

    name: str
    x: float
    z: tuple[float, float]


@dataclass(slots=True)
class GeometryModel:
    """Solve-domain geometry with explicit actual/background profiles."""

    x_span: tuple[float, float]
    z_span: tuple[float, float]
    exterior: Material
    regions: list[Region] = field(default_factory=list)
    pec_sheets: list[PECSheet] = field(default_factory=list)
    pec_slots: list[PECSlot] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x_span = _span(self.x_span, "x_span")
        self.z_span = _span(self.z_span, "z_span")
        self.exterior = _coerce_material(self.exterior)

    def _next_name(self, prefix: str, name: str | None) -> str:
        candidate = name or f"{prefix}_{len(self.regions) + 1}"
        pec_names = {
            *(sheet.name for sheet in self.pec_sheets),
            *(slot.name for slot in self.pec_slots),
        }
        if (
            not candidate
            or any(region.name == candidate for region in self.regions)
            or candidate in pec_names
        ):
            raise ConfigurationError(f"Material-region name {candidate!r} is not unique.")
        return candidate

    @staticmethod
    def _coordinate_tolerance(*values: float) -> float:
        return 64.0 * np.finfo(float).eps * max(
            1.0, *(abs(value) for value in values)
        )

    def _next_pec_name(self, prefix: str, name: str | None) -> str:
        count = len(self.pec_sheets) if prefix == "pec" else len(self.pec_slots)
        candidate = name or f"{prefix}_{count + 1}"
        existing = {
            *(region.name for region in self.regions),
            *(sheet.name for sheet in self.pec_sheets),
            *(slot.name for slot in self.pec_slots),
        }
        if not candidate or candidate in existing:
            raise ConfigurationError(f"PEC geometry name {candidate!r} is not unique.")
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

    def add_pec(
        self,
        *,
        x: float,
        z: ZExtent = "all",
        background: bool = False,
        name: str | None = None,
    ) -> PECSheet:
        """Add an ideal, mesh-conforming, constant-x PEC sheet.

        ``background=True`` makes the sheet part of both the unperturbed guide
        and the actual device.  Such a sheet must use ``z="all"``.  Finite
        openings are then introduced with :meth:`add_slot`.  With
        ``background=False``, a finite z span describes an actual-only plate.
        """

        if isinstance(x, (bool, np.bool_)):
            raise ConfigurationError("PEC x must be a finite real coordinate.")
        if not isinstance(background, (bool, np.bool_)):
            raise ConfigurationError("PEC background must be a boolean.")
        try:
            x_value = float(x)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError("PEC x must be a finite real coordinate.") from exc
        if not np.isfinite(x_value):
            raise ConfigurationError("PEC x must be a finite real coordinate.")
        tolerance = self._coordinate_tolerance(x_value, *self.x_span)
        if not (
            self.x_span[0] + tolerance
            < x_value
            < self.x_span[1] - tolerance
        ):
            raise ConfigurationError(
                "An internal PEC x coordinate must lie strictly inside x_span; "
                f"got x={x_value:g} for x_span={self.x_span}."
            )
        if background and z != "all":
            raise ConfigurationError(
                "A background PEC sheet must be z-invariant; use z='all'."
            )
        z_span = self.z_span if z == "all" else _span(z, "PEC z")
        self._inside_domain((x_value, x_value), z_span)

        for existing in self.pec_sheets:
            same_x = abs(existing.x - x_value) <= tolerance
            overlap = min(existing.z[1], z_span[1]) - max(existing.z[0], z_span[0])
            if same_x and overlap > tolerance:
                raise ConfigurationError(
                    f"PEC sheet x={x_value:g}, z={z_span} overlaps {existing.name!r}."
                )

        sheet = PECSheet(
            self._next_pec_name("pec", name),
            x_value,
            z_span,
            bool(background),
        )
        self.pec_sheets.append(sheet)
        return sheet

    def _resolve_pec_sheet(self, pec: PECSheet | str) -> PECSheet:
        if isinstance(pec, PECSheet):
            matches = [sheet for sheet in self.pec_sheets if sheet is pec]
        elif isinstance(pec, str):
            matches = [sheet for sheet in self.pec_sheets if sheet.name == pec]
        else:
            raise ConfigurationError(
                "pec must be a PECSheet returned by add_pec or its name."
            )
        if len(matches) != 1:
            identifier = pec.name if isinstance(pec, PECSheet) else pec
            raise ConfigurationError(
                f"No PEC sheet named {identifier!r} belongs to this geometry."
            )
        return matches[0]

    def add_slot(
        self,
        *,
        pec: PECSheet | str,
        z: Sequence[float],
        name: str | None = None,
    ) -> PECSlot:
        """Cut a compact actual-only slot from a z-invariant background PEC."""

        sheet = self._resolve_pec_sheet(pec)
        if not sheet.background:
            raise ConfigurationError("A slot can only be cut from a background PEC sheet.")
        tolerance = self._coordinate_tolerance(sheet.x, *sheet.z)
        slot_span = _span(z, "slot z")
        if not (
            sheet.z[0] + tolerance < slot_span[0]
            and slot_span[1] < sheet.z[1] - tolerance
        ):
            raise ConfigurationError(
                f"Slot z={slot_span} must be compact and strictly inside PEC "
                f"sheet z={sheet.z}."
            )
        for existing in self.pec_slots:
            if existing.sheet_name != sheet.name:
                continue
            overlap = min(existing.z[1], slot_span[1]) - max(
                existing.z[0], slot_span[0]
            )
            if overlap >= -tolerance:
                raise ConfigurationError(
                    f"Slot z={slot_span} overlaps or touches slot {existing.name!r}."
                )
        slot = PECSlot(
            self._next_pec_name("slot", name), sheet.name, slot_span
        )
        self.pec_slots.append(slot)
        return slot

    def slots_in(self, pec: PECSheet | str) -> tuple[PECSlot, ...]:
        """Return the slots cut from ``pec``, sorted by increasing z."""

        sheet = self._resolve_pec_sheet(pec)
        return tuple(
            sorted(
                (slot for slot in self.pec_slots if slot.sheet_name == sheet.name),
                key=lambda slot: slot.z,
            )
        )

    def pec_segments(self, *, profile: Profile) -> tuple[PECSegment, ...]:
        """Return non-overlapping PEC segments for one material profile.

        Background sheets are returned whole in the background profile.  In
        the actual profile each finite slot is subtracted, producing the PEC
        segments on either side of the opening.
        """

        if profile not in ("actual", "background"):
            raise ValueError("profile must be 'actual' or 'background'.")
        segments: list[PECSegment] = []
        for sheet in self.pec_sheets:
            if profile == "background":
                if sheet.background:
                    segments.append(PECSegment(sheet.name, sheet.x, sheet.z))
                continue
            cursor = sheet.z[0]
            for slot in self.slots_in(sheet):
                if cursor < slot.z[0]:
                    segments.append(
                        PECSegment(sheet.name, sheet.x, (cursor, slot.z[0]))
                    )
                cursor = slot.z[1]
            if cursor < sheet.z[1]:
                segments.append(PECSegment(sheet.name, sheet.x, (cursor, sheet.z[1])))
        return tuple(segments)

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
