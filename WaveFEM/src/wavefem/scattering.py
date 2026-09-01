"""High-level 2.5D scattered-field waveguide simulation workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, cos, radians, sin
from os import PathLike
from operator import index as integer_index
from typing import Callable, Literal, Mapping, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import EPSILON_0, ETA_0, MU_0
from .exceptions import ConfigurationError, ModeProjectionError, ModeSolverError
from .fem import MaxwellParameters, assemble_mixed_system, evaluate_diagonal_coefficient
from .frequency import Frequency, resolve_frequency
from .geometry import (
    Circle,
    GeometryModel,
    PECSlot,
    PECSheet,
    Polygon,
    Rectangle,
    Region,
)
from .incident import IncidentMode
from .materials import Material
from .mesh import Mesh2D, generate_mesh
from .modes import CrossSection, Mode, ModeSet, ModeSolver
from .monitors import sample_horizontal_monitor, sample_vertical_monitor
from .operators import electric_field_vector, modified_curl
from .pml import PML, PMLLayout
from .projection import ElectromagneticProjector, ModalTrace, modal_power_from_gram
from .results import ScatteringResult
from .scene import Scene2D, SceneLine
from .sources import solve_scattered_pec
from .sweep import FrequencySweepResult


MaterialFunction = Callable[..., object]


def _oblique_parameters(
    *, angle: float | None, ky: float | None
) -> tuple[float | None, float]:
    """Return the requested angle and physical invariant-direction wavenumber."""

    if angle is not None and ky is not None:
        raise ConfigurationError("Specify angle or ky, not both.")
    if angle is None:
        if ky is None:
            return 0.0, 0.0
        try:
            ky_value = complex(ky)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError("ky must be finite and real.") from exc
        if (
            isinstance(ky, (bool, np.bool_))
            or not np.isfinite((ky_value.real, ky_value.imag)).all()
            or ky_value.imag != 0.0
        ):
            raise ConfigurationError("ky must be finite and real.")
        return None, float(ky_value.real)

    try:
        angle_value = complex(angle)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("angle must be a finite real value in degrees.") from exc
    if (
        isinstance(angle, (bool, np.bool_))
        or not np.isfinite((angle_value.real, angle_value.imag)).all()
        or angle_value.imag != 0.0
    ):
        raise ConfigurationError("angle must be a finite real value in degrees.")
    angle_real = float(angle_value.real)
    if not -90.0 < angle_real < 90.0:
        raise ConfigurationError(
            "angle must lie strictly between -90 and 90 degrees."
        )
    # The physical ky is resolved only after the incident modal family is
    # selected, because its total effective index is mode-dependent.
    return angle_real, 0.0


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Numerical controls for the sparse direct MVP solver."""

    linear_solver: Literal["direct"] = "direct"
    tolerance: float = 1e-10
    quadrature_order: int = 4
    projection_condition_limit: float = 1e12

    def __post_init__(self) -> None:
        if self.linear_solver != "direct":
            raise ConfigurationError("The initial solver supports linear_solver='direct' only.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ConfigurationError("Solver tolerance must be finite and positive.")
        try:
            valid_quadrature = (
                not isinstance(self.quadrature_order, bool)
                and int(self.quadrature_order) == self.quadrature_order
                and self.quadrature_order >= 2
            )
        except (TypeError, ValueError, OverflowError):
            valid_quadrature = False
        if not valid_quadrature:
            raise ConfigurationError("quadrature_order must be an integer of at least two.")
        if not np.isfinite(self.projection_condition_limit) or self.projection_condition_limit <= 1.0:
            raise ConfigurationError("projection_condition_limit must exceed one.")


def _shape_bounds(region: Region) -> tuple[tuple[float, float], tuple[float, float]]:
    shape = region.shape
    if isinstance(shape, Rectangle):
        return shape.x, shape.z
    if isinstance(shape, Circle):
        cx, cz = shape.center
        return (cx - shape.radius, cx + shape.radius), (cz - shape.radius, cz + shape.radius)
    if isinstance(shape, Polygon):
        x, z = zip(*shape.points, strict=True)
        return (min(x), max(x)), (min(z), max(z))
    raise TypeError(f"Unsupported shape {type(shape).__name__}.")


class Scattering2D:
    """Full-vector 2.5D FEM scattering simulation in SI units.

    The unknown is the outgoing scattered field.  The unperturbed incident
    mode is analytic in the straight lead, and the equivalent source is
    compactly supported where the physical actual/background materials differ.
    """

    def __init__(
        self,
        *,
        frequency: float | None = None,
        omega: float | None = None,
        wavelength: float | None = None,
        angle: float | None = None,
        ky: float | None = None,
        x_span: Sequence[float],
        z_span: Sequence[float],
        background_eps: complex | float = 1.0,
        background_mu: complex | float = 1.0,
        transverse_boundary: Literal["pec", "pmc"] | None = None,
        solver_options: SolverOptions | None = None,
    ) -> None:
        self.frequency: Frequency = resolve_frequency(
            wavelength=wavelength, frequency=frequency, omega=omega
        )
        self.angle, self.ky = _oblique_parameters(
            angle=angle, ky=ky
        )
        exterior = Material(eps_r=background_eps, mu_r=background_mu)
        if not exterior.is_passive:
            raise ConfigurationError(
                "Integrated Scattering2D power accounting supports passive materials only."
            )
        self.geometry = GeometryModel(tuple(x_span), tuple(z_span), exterior)
        if transverse_boundary not in (None, "pec", "pmc"):
            raise ConfigurationError("transverse_boundary must be None, 'pec', or 'pmc'.")
        if transverse_boundary == "pmc":
            raise NotImplementedError(
                "Scattering2D does not yet implement a PMC transverse outer boundary; "
                "use transverse_boundary='pec' or an x-directed PML."
            )
        self.transverse_boundary = transverse_boundary
        self.solver_options = solver_options or SolverOptions()
        self.pml = PMLLayout()
        self.mesh_data: Mesh2D | None = None
        self.modes: ModeSet | None = None
        self.incident: IncidentMode | None = None
        self._incident_mode_index: int | None = None
        self._angle_mode_request: tuple[int, int] | None = None
        self._angle_modes_resolved = self.angle is None or self.angle == 0.0
        self.left_monitor: float | None = None
        self.right_monitor: float | None = None
        self._mesh_size: float | None = None
        self._material_actual: MaterialFunction | None = None
        self._material_background: MaterialFunction | None = None

    @classmethod
    def from_material_function(
        cls,
        *,
        frequency: float | None = None,
        omega: float | None = None,
        wavelength: float | None = None,
        angle: float | None = None,
        ky: float | None = None,
        domain: tuple[Sequence[float], Sequence[float]],
        eps_r: MaterialFunction,
        eps_background: MaterialFunction,
        transverse_boundary: Literal["pec", "pmc"] | None = None,
        solver_options: SolverOptions | None = None,
    ) -> "Scattering2D":
        """Construct from explicit actual and background permittivity callbacks."""

        if not callable(eps_r) or not callable(eps_background):
            raise ConfigurationError("eps_r and eps_background must be callable.")
        simulation = cls(
            wavelength=wavelength,
            frequency=frequency,
            omega=omega,
            angle=angle,
            ky=ky,
            x_span=domain[0],
            z_span=domain[1],
            background_eps=1.0,
            transverse_boundary=transverse_boundary,
            solver_options=solver_options,
        )
        simulation._material_actual = eps_r
        simulation._material_background = eps_background
        return simulation

    @property
    def x_span(self) -> tuple[float, float]:
        return self.geometry.x_span

    @property
    def z_span(self) -> tuple[float, float]:
        return self.geometry.z_span

    def _clone_at_frequency(self, frequency_hz: float) -> "Scattering2D":
        """Copy the physical configuration at one ordinary frequency."""

        if self._material_actual is not None:
            assert self._material_background is not None
            oblique = (
                {"angle": self.angle}
                if self.angle is not None
                else {"ky": self.ky}
            )
            clone = type(self).from_material_function(
                frequency=frequency_hz,
                **oblique,
                domain=(self.x_span, self.z_span),
                eps_r=self._material_actual,
                eps_background=self._material_background,
                transverse_boundary=self.transverse_boundary,
                solver_options=self.solver_options,
            )
        else:
            oblique = (
                {"angle": self.angle}
                if self.angle is not None
                else {"ky": self.ky}
            )
            clone = type(self)(
                frequency=frequency_hz,
                **oblique,
                x_span=self.x_span,
                z_span=self.z_span,
                background_eps=self.geometry.exterior.eps_r,
                background_mu=self.geometry.exterior.mu_r,
                transverse_boundary=self.transverse_boundary,
                solver_options=self.solver_options,
            )
            # Regions and their shape/material records are immutable; copy the
            # list so sweep points cannot mutate the source simulation.
            clone.geometry.regions = list(self.geometry.regions)
            clone.geometry.pec_sheets = list(self.geometry.pec_sheets)
            clone.geometry.pec_slots = list(self.geometry.pec_slots)
        clone.pml = self.pml
        clone.left_monitor = self.left_monitor
        clone.right_monitor = self.right_monitor
        return clone

    def _invalidate(self) -> None:
        self.mesh_data = None
        self.modes = None
        self.incident = None
        self._incident_mode_index = None
        self._angle_mode_request = None
        self._angle_modes_resolved = self.angle is None or self.angle == 0.0
        if self.angle is not None:
            self.ky = 0.0

    def _require_region_geometry(self) -> None:
        if self._material_actual is not None:
            raise ConfigurationError(
                "Geometry primitives cannot be mixed with from_material_function callbacks."
            )

    def _validate_integrated_pec_geometry(self) -> None:
        (x0, x1), (z0, z1) = self._interior_spans()
        for sheet in self.geometry.pec_sheets:
            if sheet.background:
                continue
            if not z0 < sheet.z[0] < sheet.z[1] < z1:
                raise ConfigurationError(
                    f"Actual-only PEC sheet {sheet.name!r} must be compact and lie "
                    "strictly inside the non-PML z interior."
                )
            if self.pml.x is not None and not x0 < sheet.x < x1:
                raise ConfigurationError(
                    f"Actual-only PEC sheet {sheet.name!r} must lie strictly inside "
                    "the non-PML x interior."
                )

    def add_rectangle(
        self,
        *,
        x: Sequence[float],
        z: tuple[float, float] | Literal["all"],
        eps: complex | float,
        mu: complex | float = 1.0,
        background: bool = False,
        name: str | None = None,
    ) -> Region:
        self._require_region_geometry()
        material = Material(eps_r=eps, mu_r=mu)
        if not material.is_passive:
            raise ConfigurationError(
                "Integrated Scattering2D power accounting supports passive materials only."
            )
        region = self.geometry.add_rectangle(
            x=x,
            z=z,
            material=material,
            background=background,
            name=name,
        )
        self._invalidate()
        return region

    def add_circle(
        self,
        *,
        center: Sequence[float],
        radius: float,
        eps: complex | float,
        mu: complex | float = 1.0,
        name: str | None = None,
    ) -> Region:
        self._require_region_geometry()
        material = Material(eps_r=eps, mu_r=mu)
        if not material.is_passive:
            raise ConfigurationError(
                "Integrated Scattering2D power accounting supports passive materials only."
            )
        region = self.geometry.add_circle(
            center=center,
            radius=radius,
            material=material,
            name=name,
        )
        self._invalidate()
        return region

    def add_polygon(
        self,
        *,
        points: Sequence[Sequence[float]],
        eps: complex | float,
        mu: complex | float = 1.0,
        name: str | None = None,
    ) -> Region:
        self._require_region_geometry()
        material = Material(eps_r=eps, mu_r=mu)
        if not material.is_passive:
            raise ConfigurationError(
                "Integrated Scattering2D power accounting supports passive materials only."
            )
        region = self.geometry.add_polygon(
            points=points,
            material=material,
            name=name,
        )
        self._invalidate()
        return region

    def add_pec(
        self,
        *,
        x: float,
        z: Sequence[float] | Literal["all"] = "all",
        background: bool = True,
        name: str | None = None,
    ) -> PECSheet:
        """Add a background ground sheet or a compact actual-only PEC plate.

        A background sheet must use ``z='all'`` and is present in both the
        unperturbed lead and actual device until openings are cut with
        :meth:`add_slot`.  An actual-only sheet must have a finite compact
        ``z`` span.  Its scattered tangential field is prescribed as the
        negative incident trace so the total tangential field vanishes.
        """

        self._require_region_geometry()
        if not isinstance(background, (bool, np.bool_)):
            raise ConfigurationError(
                "PEC background must be a boolean."
            )
        background_value = bool(background)
        if not background_value and isinstance(z, str) and z == "all":
            raise ConfigurationError(
                "An actual-only PEC sheet must be compact; provide a finite z span."
            )
        sheet = self.geometry.add_pec(
            x=x,
            z=z,
            background=background_value,
            name=name,
        )
        self._invalidate()
        return sheet

    def add_slot(
        self,
        *,
        pec: PECSheet | str,
        z: Sequence[float],
        name: str | None = None,
    ) -> PECSlot:
        """Cut a finite slot from a background PEC sheet in the actual device.

        The background guide retains the complete PEC sheet for modal
        excitation.  In the actual solve, facets inside ``z`` are released and
        driven by the two one-sided incident magnetic reactions.
        """

        self._require_region_geometry()
        slot = self.geometry.add_slot(pec=pec, z=z, name=name)
        self._invalidate()
        return slot

    def add_pml(
        self,
        *,
        x: float | PML | None = None,
        z: float | PML | None = None,
        order: int = 3,
        target_reflection: float = 1e-8,
    ) -> None:
        def convert(value: float | PML | None) -> PML | None:
            if value is None or isinstance(value, PML):
                return value
            return PML(float(value), order=order, target_reflection=target_reflection)

        self.pml = PMLLayout(
            x=self.pml.x if x is None else convert(x),
            z=self.pml.z if z is None else convert(z),
        )
        self.pml.validate_domain(self.x_span, self.z_span)
        self._invalidate()

    def set_monitors(self, *, left: float, right: float) -> None:
        left, right = float(left), float(right)
        if not np.isfinite((left, right)).all() or not left < right:
            raise ConfigurationError("Monitor coordinates must be finite with left < right.")
        self.left_monitor, self.right_monitor = left, right
        self.mesh_data = None

    def _call_eps(self, callback: MaterialFunction, x: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
        try:
            value = callback(x, z)
        except TypeError:
            value = callback(x)
        result = np.asarray(value, dtype=np.complex128)
        try:
            return np.asarray(np.broadcast_to(result, np.broadcast_shapes(np.shape(x), np.shape(z))), dtype=np.complex128)
        except ValueError as exc:
            raise ConfigurationError("Material callback output is not compatible with x/z coordinates.") from exc

    def _physical_material(
        self, x: ArrayLike, z: ArrayLike, *, profile: Literal["actual", "background"]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        if self._material_actual is None:
            return self.geometry.material_at(x, z, profile=profile)
        callback = self._material_actual if profile == "actual" else self._material_background
        assert callback is not None
        eps = self._call_eps(callback, x, z)
        mu = np.ones_like(eps)
        if np.any(np.imag(eps) < -1e-14):
            raise ConfigurationError(
                "Integrated Scattering2D material callbacks must be passive."
            )
        return eps, mu

    def _transformed_material(
        self, x: ArrayLike, z: ArrayLike, *, profile: Literal["actual", "background"]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        eps, mu = self._physical_material(x, z, profile=profile)
        sx, sz = self.pml.stretching(
            x,
            z,
            x_span=self.x_span,
            z_span=self.z_span,
            k_reference=self.frequency.k0,
        )
        return self.pml.transform_isotropic(eps, mu, sx, sz)

    def _eps_actual(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
        return self._transformed_material(x, z, profile="actual")[0]

    def _mu_actual(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
        return self._transformed_material(x, z, profile="actual")[1]

    def _eps_background(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
        return self._transformed_material(x, z, profile="background")[0]

    def _mu_background(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
        return self._transformed_material(x, z, profile="background")[1]

    def _interior_spans(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x0, x1 = self.x_span
        z0, z1 = self.z_span
        if self.pml.x is not None:
            x0 += self.pml.x.thickness
            x1 -= self.pml.x.thickness
        if self.pml.z is not None:
            z0 += self.pml.z.thickness
            z1 -= self.pml.z.thickness
        return (x0, x1), (z0, z1)

    def _visualization_scene(self) -> Scene2D:
        """Return full-domain material and overlay data for persisted results."""

        if self.mesh_data is None:
            raise ConfigurationError("A mesh is required to build visualization data.")
        if self.left_monitor is None or self.right_monitor is None:
            raise ConfigurationError("Monitor lines are required to build visualization data.")

        points = np.asarray(self.mesh_data.mesh.p, dtype=np.float64)
        triangles = np.asarray(self.mesh_data.mesh.t, dtype=np.int64)
        centroids = points[:, triangles].mean(axis=1)
        eps_r, _ = self._physical_material(
            centroids[0], centroids[1], profile="actual"
        )

        xmin, xmax = self.x_span
        zmin, zmax = self.z_span
        (port_xmin, port_xmax), _ = self._interior_spans()
        lines: list[SceneLine] = [
            SceneLine("pec", ((xmin, zmin), (xmin, zmax)), "PEC outer boundary"),
            SceneLine("pec", ((xmax, zmin), (xmax, zmax)), "PEC outer boundary"),
            SceneLine("pec", ((xmin, zmin), (xmax, zmin)), "PEC outer boundary"),
            SceneLine("pec", ((xmin, zmax), (xmax, zmax)), "PEC outer boundary"),
            SceneLine(
                "wave_port",
                ((port_xmin, self.left_monitor), (port_xmax, self.left_monitor)),
                "left wave port",
            ),
            SceneLine(
                "wave_port",
                ((port_xmin, self.right_monitor), (port_xmax, self.right_monitor)),
                "right wave port",
            ),
        ]
        lines.extend(
            SceneLine(
                "pec",
                ((segment.x, segment.z[0]), (segment.x, segment.z[1])),
                segment.name,
            )
            for segment in self.geometry.pec_segments(profile="actual")
        )
        pml_x, pml_z = self.pml.interfaces(self.x_span, self.z_span)
        lines.extend(
            SceneLine("pml", ((x, zmin), (x, zmax)), "x-PML interface")
            for x in pml_x
        )
        lines.extend(
            SceneLine("pml", ((xmin, z), (xmax, z)), "z-PML interface")
            for z in pml_z
        )
        return Scene2D(
            points=points,
            triangles=triangles,
            eps_r=np.asarray(eps_r, dtype=np.complex128),
            x_span=self.x_span,
            z_span=self.z_span,
            lines=tuple(lines),
        )

    def _perturbation_z_bounds(self) -> tuple[float, float] | None:
        if self._material_actual is not None:
            return None
        bounds = [_shape_bounds(region)[1] for region in self.geometry.perturbations]
        bounds.extend(slot.z for slot in self.geometry.pec_slots)
        bounds.extend(
            sheet.z for sheet in self.geometry.pec_sheets if not sheet.background
        )
        if not bounds:
            return None
        return min(item[0] for item in bounds), max(item[1] for item in bounds)

    def _pec_coordinates_at(
        self,
        z: float,
        *,
        profile: Literal["actual", "background"],
    ) -> tuple[float, ...]:
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(z), *(abs(value) for value in self.z_span)
        )
        return tuple(
            sorted(
                segment.x
                for segment in self.geometry.pec_segments(profile=profile)
                if segment.z[0] - tolerance <= z <= segment.z[1] + tolerance
            )
        )

    def _choose_monitors(self) -> None:
        _, (z0, z1) = self._interior_spans()
        perturbation = self._perturbation_z_bounds()
        if self.left_monitor is None or self.right_monitor is None:
            if perturbation is None:
                width = z1 - z0
                self.left_monitor = z0 + width / 3.0
                self.right_monitor = z1 - width / 3.0
            else:
                self.left_monitor = 0.5 * (z0 + perturbation[0])
                self.right_monitor = 0.5 * (perturbation[1] + z1)
        assert self.left_monitor is not None and self.right_monitor is not None
        if not z0 < self.left_monitor < self.right_monitor < z1:
            raise ConfigurationError("Monitor lines must lie in the non-PML z interior.")
        if perturbation is not None and not (
            self.left_monitor < perturbation[0] and self.right_monitor > perturbation[1]
        ):
            raise ConfigurationError("Monitor line lies inside or intersects the perturbation.")
        sample_x = np.linspace(*self.x_span, 101)
        for label, monitor in (("left", self.left_monitor), ("right", self.right_monitor)):
            actual, _ = self._physical_material(sample_x, np.full_like(sample_x, monitor), profile="actual")
            background, _ = self._physical_material(sample_x, np.full_like(sample_x, monitor), profile="background")
            if not np.allclose(actual, background, rtol=1e-12, atol=1e-14):
                raise ConfigurationError(f"The {label} monitor is not inside a uniform lead.")
            if self._pec_coordinates_at(monitor, profile="actual") != self._pec_coordinates_at(
                monitor, profile="background"
            ):
                raise ConfigurationError(
                    f"The {label} monitor intersects a PEC boundary perturbation."
                )

    def _maximum_index(self) -> float:
        if self._material_actual is None:
            materials = [self.geometry.exterior, *(r.material for r in self.geometry.regions)]
            return max(float(np.sqrt(abs(m.eps_r * m.mu_r))) for m in materials)
        x = np.linspace(*self.x_span, 33)
        z = np.linspace(*self.z_span, 33)
        xx, zz = np.meshgrid(x, z, indexing="ij")
        eps, mu = self._physical_material(xx, zz, profile="actual")
        return float(np.max(np.sqrt(np.abs(eps * mu))))

    def mesh(
        self,
        *,
        max_element_size: float | None = None,
        wavelength_elements: int = 10,
        refine_interfaces: bool = True,
        dielectric_refinement_factor: float = 0.5,
        pec_refinement_factor: float = 0.5,
        pec_refinement_distance: float | None = None,
    ) -> Mesh2D:
        """Generate the Gmsh mesh and reveal the selected maximum edge size.

        Material, PEC, and PML boundaries are always conforming.  With
        ``refine_interfaces=True``, local wavelength sizing refines dielectric
        regions and a distance field refines the neighbourhood of actual PEC.
        """

        if not isinstance(refine_interfaces, bool):
            raise ConfigurationError("refine_interfaces must be a boolean.")
        if isinstance(dielectric_refinement_factor, bool):
            raise ConfigurationError(
                "dielectric_refinement_factor must be in the interval (0, 1]."
            )
        try:
            dielectric_factor = float(dielectric_refinement_factor)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                "dielectric_refinement_factor must be in the interval (0, 1]."
            ) from exc
        if not np.isfinite(dielectric_factor) or not 0.0 < dielectric_factor <= 1.0:
            raise ConfigurationError(
                "dielectric_refinement_factor must be in the interval (0, 1]."
            )
        if isinstance(pec_refinement_factor, bool):
            raise ConfigurationError(
                "pec_refinement_factor must be in the interval (0, 1]."
            )
        try:
            pec_factor = float(pec_refinement_factor)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                "pec_refinement_factor must be in the interval (0, 1]."
            ) from exc
        if not np.isfinite(pec_factor) or not 0.0 < pec_factor <= 1.0:
            raise ConfigurationError(
                "pec_refinement_factor must be in the interval (0, 1]."
            )
        if pec_refinement_distance is None:
            pec_distance = None
        else:
            if isinstance(pec_refinement_distance, bool):
                raise ConfigurationError(
                    "pec_refinement_distance must be finite and positive."
                )
            try:
                pec_distance = float(pec_refinement_distance)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ConfigurationError(
                    "pec_refinement_distance must be finite and positive."
                ) from exc
            if not np.isfinite(pec_distance) or pec_distance <= 0.0:
                raise ConfigurationError(
                    "pec_refinement_distance must be finite and positive."
                )
        self._validate_integrated_pec_geometry()
        try:
            valid_density = (
                not isinstance(wavelength_elements, bool)
                and int(wavelength_elements) == wavelength_elements
                and wavelength_elements >= 4
            )
        except (TypeError, ValueError, OverflowError):
            valid_density = False
        if not valid_density:
            raise ConfigurationError("wavelength_elements must be an integer of at least four.")
        self._choose_monitors()
        maximum_index = max(self._maximum_index(), 1.0)
        sizing_index = maximum_index
        if self._material_actual is None and refine_interfaces:
            exterior = self.geometry.exterior
            local_reference_index = max(
                float(np.sqrt(abs(exterior.eps_r * exterior.mu_r))),
                1.0,
            )
        else:
            # Callback materials cannot be assigned reliable local Gmsh fields,
            # so retain conservative global sizing by their sampled maximum.
            local_reference_index = maximum_index
        derived = self.frequency.wavelength / (wavelength_elements * sizing_index)
        selected = derived if max_element_size is None else min(float(max_element_size), derived)
        if not np.isfinite(selected) or selected <= 0.0:
            raise ConfigurationError("max_element_size must be finite and positive.")
        self._mesh_size = selected * (
            min(
                1.0,
                dielectric_factor * local_reference_index / maximum_index,
            )
            if refine_interfaces and self._material_actual is None
            else 1.0
        )
        for axis, pml in (("x", self.pml.x), ("z", self.pml.z)):
            if pml is not None and pml.thickness < 3.0 * selected:
                warnings.warn(
                    f"The {axis}-PML thickness spans fewer than three requested "
                    "maximum edge lengths; refine the mesh or thicken the PML.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        perturbation = self._perturbation_z_bounds()
        if perturbation is not None:
            assert self.left_monitor is not None and self.right_monitor is not None
            monitor_gap = min(
                perturbation[0] - self.left_monitor,
                self.right_monitor - perturbation[1],
            )
            if monitor_gap < 2.0 * selected:
                warnings.warn(
                    "A monitor is fewer than two requested element sizes from the "
                    "perturbation; move it deeper into the uniform lead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        x_partitions, z_partitions = self.pml.interfaces(self.x_span, self.z_span)
        z_partitions = tuple((*z_partitions, self.left_monitor, self.right_monitor))
        self.mesh_data = generate_mesh(
            self.geometry,
            max_element_size=selected,
            # PML interfaces are required facets for the closed Poynting-flux
            # control surface, irrespective of optional material-interface
            # refinement policy.  Material boundaries themselves are always
            # conforming in the Gmsh geometry.
            x_partitions=x_partitions,
            z_partitions=z_partitions,
            refine_dielectrics=refine_interfaces,
            dielectric_refinement_factor=dielectric_factor,
            refine_pec=refine_interfaces,
            pec_refinement_factor=pec_factor,
            pec_refinement_distance=pec_distance,
        )
        self.modes = None
        self.incident = None
        self._incident_mode_index = None
        self._angle_mode_request = None
        self._angle_modes_resolved = self.angle is None or self.angle == 0.0
        if self.angle is not None:
            self.ky = 0.0
        return self.mesh_data

    def _cross_section(self) -> CrossSection:
        self._validate_integrated_pec_geometry()
        if self._material_background is not None:
            raise ConfigurationError(
                "Mode solving from arbitrary material callbacks requires an explicit CrossSection."
            )
        lead_materials = [
            self.geometry.exterior,
            *(region.material for region in self.geometry.background_regions),
        ]
        if any(not material.is_lossless for material in lead_materials):
            raise ConfigurationError(
                "The integrated Scattering2D projector currently requires lossless "
                "uniform lead materials. Compact lossy perturbations are supported."
            )
        boundary = self.transverse_boundary
        if self.pml.x is not None:
            boundary = "pec"
        if boundary is None:
            raise ConfigurationError(
                "An open transverse guide requires add_pml(x=...); otherwise set "
                "transverse_boundary='pec' or 'pmc' explicitly."
            )
        kwargs: dict[str, object] = {}
        if self.pml.x is not None:
            kwargs["pml"] = self.pml.x
        cross_section = CrossSection(
            self.x_span,
            background=self.geometry.exterior,
            boundary=boundary,
            **kwargs,
        )
        for region in self.geometry.background_regions:
            if not isinstance(region.shape, Rectangle):
                raise ConfigurationError("Background guide regions must be z-invariant rectangles.")
            cross_section.add_layer(x=region.shape.x, material=region.material, name=region.name)
        for sheet in self.geometry.pec_sheets:
            if sheet.background:
                cross_section.add_pec(x=sheet.x, name=sheet.name)
        return cross_section

    def _is_bound_guided_mode(self, mode: Mode) -> bool:
        """Classify a lossless transverse-PML candidate by the cladding light line."""

        if self.pml.x is None:
            return mode.is_propagating
        physical_left = self.x_span[0] + self.pml.x.thickness
        physical_right = self.x_span[1] - self.pml.x.thickness
        inset = 1e-9 * (physical_right - physical_left)
        probes = np.asarray(
            (physical_left + inset, physical_right - inset), dtype=float
        )
        probe_z = np.full_like(probes, 0.5 * sum(self.z_span))
        eps_r, mu_r = self._physical_material(
            probes,
            probe_z,
            profile="background",
        )
        eta = self.ky / self.frequency.k0
        light_line_squared = max(float(np.max(np.real(eps_r * mu_r))) - eta**2, 0.0)
        light_line = float(np.sqrt(light_line_squared))
        margin = 1e-6 * max(1.0, light_line, abs(mode.neff))
        return (
            mode.is_propagating
            and mode.neff.real > light_line + margin
            # A weakly confined surface mode can overlap the finite PML enough
            # to acquire a small positive numerical attenuation.  Keep such a
            # mode while rejecting the much more strongly damped PML spectrum.
            and abs(mode.neff.imag) <= 1e-3 * max(1.0, abs(mode.neff.real))
        )

    def set_modes(self, modes: ModeSet) -> ModeSet:
        """Bind a compatible solved lead-mode set to this simulation.

        This is the explicit mode-injection path for
        :meth:`from_material_function`; WaveFEM never guesses a discontinuous
        one-dimensional cross-section from arbitrary two-dimensional callback
        samples.
        """

        if not isinstance(modes, ModeSet) or len(modes) == 0:
            raise ConfigurationError("modes must be a nonempty ModeSet.")
        for mode in modes:
            if not np.isclose(
                mode.omega,
                self.frequency.omega,
                rtol=1e-12,
                atol=0.0,
            ):
                raise ConfigurationError(
                    "Injected modes use a different angular frequency."
                )
            if not np.isclose(mode.ky, self.ky, rtol=1e-12, atol=1e-14):
                raise ConfigurationError("Injected modes use a different ky.")
            if not np.allclose(
                (mode.x_nodes[0], mode.x_nodes[-1]),
                self.x_span,
                rtol=1e-12,
                atol=1e-18,
            ):
                raise ConfigurationError(
                    "Injected mode cross-section does not match Scattering2D.x_span."
                )

        selected = modes
        if self.pml.x is not None:
            guided = tuple(mode for mode in modes if self._is_bound_guided_mode(mode))
            if not guided:
                raise ModeSolverError(
                    "No bound guided mode was found above the exterior light line. "
                    "Adjust neff_guess, the transverse domain/PML, or the guide profile."
                )
            if len(guided) < len(modes):
                warnings.warn(
                    f"Discarded {len(modes) - len(guided)} transverse-PML/radiation "
                    "mode candidate(s); integrated port powers include bound guided modes only.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            selected = ModeSet(
                modes=guided,
                system=modes.system,
                solve_info={
                    **modes.solve_info,
                    "candidate_count_before_guided_filter": len(modes),
                    "guided_mode_count": len(guided),
                    "guided_filter": "exterior-light-line",
                },
            )
        self.modes = selected
        self.incident = None
        self._incident_mode_index = None
        self._angle_mode_request = None
        self._angle_modes_resolved = self.angle is None or self.angle == 0.0
        return selected

    def _resolve_modes_for_angle(self, selected_index: int) -> int:
        """Resolve the selected normal-incidence family at ``self.angle``."""

        if self.angle is None or self.angle == 0.0:
            return selected_index
        if self._angle_mode_request is None or self.modes is None:
            raise ConfigurationError(
                "A nonzero angle must be resolved from modes produced by "
                "Scattering2D.solve_modes() before selecting the incident mode."
            )
        if self._material_background is not None:
            raise ConfigurationError(
                "Nonzero-angle callback simulations cannot derive an oblique mode "
                "from an injected ModeSet; use an integrated geometry-backed lead."
            )

        seed = self.modes[selected_index]
        if seed.classification != "propagating" or seed.neff.real <= 0.0:
            raise ConfigurationError(
                "The incident family used to resolve angle must be a forward "
                "propagating mode with positive effective index."
            )
        num_modes, num_elements = self._angle_mode_request
        total_neff = float(seed.neff.real)
        angle_radians = radians(self.angle)
        self.ky = float(
            self.frequency.k0 * total_neff * sin(angle_radians)
        )
        expected_z_neff = seed.neff * cos(angle_radians)
        solver = ModeSolver(
            self._cross_section(),
            omega=self.frequency.omega,
            ky=self.ky,
            num_elements=num_elements,
        )
        candidates = solver.solve(
            num_modes=num_modes,
            neff_guess=expected_z_neff,
            direction="forward",
        )
        resolved = self.set_modes(candidates)
        self._angle_mode_request = (num_modes, num_elements)
        propagating = [
            index for index, candidate in enumerate(resolved)
            if candidate.classification == "propagating" and candidate.neff.real > 0.0
        ]
        if not propagating:
            raise ModeSolverError(
                "No propagating mode remained after resolving the requested angle."
            )
        resolved_index = min(
            propagating,
            key=lambda index: abs(resolved[index].neff - expected_z_neff),
        )
        selected = resolved[resolved_index]
        eta = self.ky / self.frequency.k0
        resolved_angle = float(np.degrees(np.arctan2(eta, selected.neff.real)))
        resolved_total_neff = float(np.hypot(selected.neff.real, eta))
        if not np.isclose(resolved_angle, self.angle, rtol=0.0, atol=5e-4):
            raise ModeSolverError(
                "The selected modal family did not preserve the requested "
                "propagation angle after the oblique solve."
            )
        self.modes = ModeSet(
            modes=resolved.modes,
            system=resolved.system,
            solve_info={
                **resolved.solve_info,
                "angle_degrees": self.angle,
                "angle_seed_mode_index": selected_index,
                "angle_resolved_mode_index": resolved_index,
                "angle_total_neff": resolved_total_neff,
            },
        )
        self._angle_modes_resolved = True
        return resolved_index

    def solve_modes(
        self,
        *,
        side: Literal["left", "right"] = "left",
        num_modes: int = 4,
        neff_guess: complex | None = None,
        num_elements: int | None = None,
    ) -> ModeSet:
        if side not in ("left", "right"):
            raise ConfigurationError("side must be 'left' or 'right'.")
        if num_elements is None:
            h = self._mesh_size or self.frequency.wavelength / (10 * max(self._maximum_index(), 1.0))
            num_elements = max(40, int(ceil((self.x_span[1] - self.x_span[0]) / h)))
        if self.angle is not None:
            self.ky = 0.0
        solver = ModeSolver(
            self._cross_section(),
            omega=self.frequency.omega,
            ky=self.ky,
            num_elements=num_elements,
        )
        candidates = solver.solve(
            num_modes=num_modes,
            neff_guess=neff_guess,
            direction="forward",
        )
        selected = self.set_modes(candidates)
        if self.angle is not None and self.angle != 0.0:
            self._angle_mode_request = (int(num_modes), int(num_elements))
            self._angle_modes_resolved = False
        return selected

    def set_incident_mode(
        self,
        mode: int | Mode,
        *,
        side: Literal["left", "right"] = "left",
        reference_plane: float | None = None,
        amplitude: complex = 1.0,
    ) -> IncidentMode:
        if isinstance(mode, bool):
            raise ConfigurationError("mode must be a Mode or nonnegative integer index.")
        if isinstance(mode, int):
            if self.modes is None:
                raise ConfigurationError("solve_modes must be called before selecting a mode index.")
            if mode < 0 or mode >= len(self.modes):
                raise ConfigurationError(
                    f"mode index {mode} is outside the solved range 0..{len(self.modes) - 1}."
                )
            selected_index = mode
            selected = self.modes[selected_index]
        elif isinstance(mode, Mode):
            if self.modes is None:
                raise ConfigurationError(
                    "solve_modes must be called before binding an incident Mode to Scattering2D."
                )
            matches = [index for index, candidate in enumerate(self.modes) if candidate is mode]
            if not matches:
                raise ConfigurationError(
                    "The incident Mode must be a member of this simulation's current "
                    "ModeSet; external or stale modes cannot be projected safely."
                )
            selected_index = matches[0]
            selected = mode
        else:
            raise ConfigurationError("mode must be a Mode or integer index.")
        if self.angle is not None and self.angle != 0.0 and not self._angle_modes_resolved:
            selected_index = self._resolve_modes_for_angle(selected_index)
            assert self.modes is not None
            selected = self.modes[selected_index]
        if (
            selected.classification != "propagating"
            or selected.normalization != "unit-power"
        ):
            raise ConfigurationError(
                "Integrated S-parameter and power accounting requires a "
                "propagating, unit-power-normalized incident mode; evanescent "
                "launches are available only through the low-level field APIs."
            )
        self._choose_monitors()
        if reference_plane is None:
            reference_plane = self.left_monitor if side == "left" else self.right_monitor
        assert reference_plane is not None
        launched = IncidentMode(
            selected,
            side=side,
            reference_plane=reference_plane,
            amplitude=amplitude,
        )
        if abs(launched.amplitude) < np.sqrt(np.finfo(float).tiny):
            raise ConfigurationError(
                "Incident amplitude is zero or too small for finite power normalization."
            )
        self.incident = launched
        self._incident_mode_index = selected_index
        return self.incident

    def sweep_frequencies(
        self,
        frequencies_hz: Sequence[float],
        *,
        h5_path: str | PathLike[str] | None = "wavefem_sweep.h5",
        mesh_options: Mapping[str, object] | None = None,
        mode_options: Mapping[str, object] | None = None,
        incident_mode: int = 0,
        amplitude: complex = 1.0,
        reference_plane: float | None = None,
        mode_factory: Callable[[float], ModeSet] | None = None,
    ) -> FrequencySweepResult:
        """Solve independent points on an increasing ordinary-frequency grid.

        Each point receives its own mesh, lead-mode solve, incident field, and
        scattered-field solve.  The source simulation is not mutated.  By
        default the complete sweep is saved to ``wavefem_sweep.h5``; pass
        ``h5_path=None`` only when an in-memory sweep is explicitly desired.

        Callback-defined devices must provide ``mode_factory(frequency_hz)``
        because their lead cross-section cannot be inferred automatically.
        """

        raw_frequencies = np.asarray(frequencies_hz)
        if (
            raw_frequencies.ndim != 1
            or raw_frequencies.size == 0
            or np.iscomplexobj(raw_frequencies)
            or raw_frequencies.dtype.kind == "b"
        ):
            raise ConfigurationError(
                "frequencies_hz must be a nonempty one-dimensional real array."
            )
        try:
            frequencies = np.asarray(raw_frequencies, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                "frequencies_hz must contain finite positive values in hertz."
            ) from exc
        if (
            not np.isfinite(frequencies).all()
            or np.any(frequencies <= 0.0)
            or np.any(np.diff(frequencies) <= 0.0)
        ):
            raise ConfigurationError(
                "frequencies_hz must be finite, positive, and strictly increasing."
            )
        if isinstance(incident_mode, bool):
            raise ConfigurationError("incident_mode must be a nonnegative integer.")
        try:
            incident_mode = integer_index(incident_mode)
        except TypeError as exc:
            raise ConfigurationError(
                "incident_mode must be a nonnegative integer."
            ) from exc
        if incident_mode < 0:
            raise ConfigurationError("incident_mode must be a nonnegative integer.")
        if mode_factory is not None and not callable(mode_factory):
            raise ConfigurationError("mode_factory must be callable or None.")
        if self._material_actual is not None and mode_factory is None:
            raise ConfigurationError(
                "Callback frequency sweeps require mode_factory(frequency_hz) "
                "returning a compatible positive-z ModeSet."
            )
        try:
            mesh_kwargs = dict(mesh_options or {})
            mode_kwargs = dict(mode_options or {})
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "mesh_options and mode_options must be mappings."
            ) from exc
        if mode_factory is None:
            if mode_kwargs.get("side", "left") != "left":
                raise ConfigurationError(
                    "The integrated frequency sweep currently supports left incidence."
                )
            requested_modes = mode_kwargs.setdefault(
                "num_modes", max(1, incident_mode + 1)
            )
            try:
                normalized_modes = (
                    None
                    if isinstance(requested_modes, bool)
                    else integer_index(requested_modes)
                )
            except TypeError:
                normalized_modes = None
            if normalized_modes is None or normalized_modes <= incident_mode:
                raise ConfigurationError(
                    "mode_options['num_modes'] must exceed incident_mode."
                )
            mode_kwargs["num_modes"] = normalized_modes

        results: list[ScatteringResult] = []
        for value in frequencies:
            point = self._clone_at_frequency(float(value))
            point.mesh(**mesh_kwargs)  # type: ignore[arg-type]
            if mode_factory is None:
                point.solve_modes(**mode_kwargs)  # type: ignore[arg-type]
            else:
                point.set_modes(mode_factory(float(value)))
            point.set_incident_mode(
                incident_mode,
                side="left",
                reference_plane=reference_plane,
                amplitude=amplitude,
            )
            results.append(point.solve())

        sweep = FrequencySweepResult(
            frequencies_hz=np.array(frequencies, copy=True),
            results=tuple(results),
        )
        if h5_path is not None:
            written = sweep.save_h5(h5_path)
            sweep = replace(sweep, h5_path=written)
        return sweep

    @staticmethod
    def _poynting_z(E: NDArray[np.complex128], H: NDArray[np.complex128], weights: NDArray[np.float64]) -> float:
        return float(0.5 * np.real(np.sum(weights * (E[0] * np.conj(H[1]) - E[1] * np.conj(H[0])))))

    @staticmethod
    def _poynting_x(E: NDArray[np.complex128], H: NDArray[np.complex128], weights: NDArray[np.float64]) -> float:
        return float(0.5 * np.real(np.sum(weights * (E[1] * np.conj(H[2]) - E[2] * np.conj(H[1])))))

    def solve(
        self,
        *,
        h5_path: str | PathLike[str] | None = None,
    ) -> ScatteringResult:
        """Assemble, solve, project modes, and return fields and power diagnostics."""

        if self.mesh_data is None:
            raise ConfigurationError("mesh() must be called before solve().")
        self._validate_integrated_pec_geometry()
        if (
            self.modes is None
            or self.incident is None
            or self._incident_mode_index is None
        ):
            raise ConfigurationError("solve_modes() and set_incident_mode() are required before solve().")
        if self.incident.side != "left":
            raise NotImplementedError("The initial integrated scattering solve supports left incidence.")
        if self.pml.z is None:
            raise ConfigurationError("Outgoing scattering requires add_pml(z=...).")
        self._choose_monitors()
        assert self.left_monitor is not None and self.right_monitor is not None

        length_scale = 1.0 / self.frequency.k0
        parameters = MaxwellParameters(
            k0=self.frequency.k0,
            ky=self.ky,
            eps_r=self._eps_actual,
            mu_r=self._mu_actual,
        )
        system = assemble_mixed_system(
            self.mesh_data.mesh,
            parameters,
            intorder=self.solver_options.quadrature_order,
            length_scale=length_scale,
            internal_pec_facets=self.mesh_data.actual_pec_facets,
        )
        scattered = solve_scattered_pec(
            system,
            eps_background=self._eps_background,
            mu_background=self._mu_background,
            incident=self.incident,
            released_pec_facets=self.mesh_data.released_pec_facets,
            inserted_pec_facets=self.mesh_data.inserted_pec_facets,
            incident_magnetic=self.incident.H,
            residual_tolerance=self.solver_options.tolerance,
        )
        coefficients = scattered.field.coefficients

        left_sc = sample_vertical_monitor(
            system.basis,
            coefficients,
            z=self.left_monitor,
            ky=self.ky,
            omega=self.frequency.omega,
            mu_r=self._mu_actual,
            length_scale=length_scale,
            intorder=self.solver_options.quadrature_order,
        )
        right_sc = sample_vertical_monitor(
            system.basis,
            coefficients,
            z=self.right_monitor,
            ky=self.ky,
            omega=self.frequency.omega,
            mu_r=self._mu_actual,
            length_scale=length_scale,
            intorder=self.solver_options.quadrature_order,
        )

        def total_monitor(samples: object) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
            z_values = np.full_like(samples.x, samples.z)  # type: ignore[attr-defined]
            inc_e, inc_h = self.incident.fields(samples.x, z_values)  # type: ignore[attr-defined]
            return samples.E + inc_e, samples.H + inc_h  # type: ignore[attr-defined]

        left_E, left_H = total_monitor(left_sc)
        right_E, right_H = total_monitor(right_sc)
        forward_modes = tuple(self.modes)
        backward_modes = tuple(mode.counterpropagating() for mode in forward_modes)

        def traces(
            samples: object, mask: NDArray[np.bool_]
        ) -> tuple[ModalTrace, ...]:
            sample_x = samples.x[mask]  # type: ignore[attr-defined]
            z_values = np.full_like(sample_x, samples.z)  # type: ignore[attr-defined]
            result: list[ModalTrace] = []
            for index, mode in enumerate((*forward_modes, *backward_modes)):
                electric, magnetic = mode.fields(
                    sample_x,
                    z_values,
                    reference_plane=self.incident.reference_plane,
                )
                direction = "f" if index < len(forward_modes) else "b"
                result.append(ModalTrace(electric, magnetic, f"{direction}{index % len(forward_modes)}"))
            return tuple(result)

        (x0, x1), _ = self._interior_spans()
        left_modal_mask = (left_sc.x >= x0) & (left_sc.x <= x1)
        right_modal_mask = (right_sc.x >= x0) & (right_sc.x <= x1)
        left_projection = ElectromagneticProjector(
            traces(left_sc, left_modal_mask),
            left_sc.weights[left_modal_mask],
            impedance=ETA_0,
            condition_limit=self.solver_options.projection_condition_limit,
        ).project(left_E[:, left_modal_mask], left_H[:, left_modal_mask])
        right_projection = ElectromagneticProjector(
            traces(right_sc, right_modal_mask),
            right_sc.weights[right_modal_mask],
            impedance=ETA_0,
            condition_limit=self.solver_options.projection_condition_limit,
        ).project(right_E[:, right_modal_mask], right_H[:, right_modal_mask])
        mode_count = len(forward_modes)
        incident_index = self._incident_mode_index
        if incident_index >= mode_count:
            raise ConfigurationError("The selected incident mode is no longer in the current ModeSet.")
        prescribed_incoming = complex(self.incident.amplitude)
        projected_incoming = left_projection.amplitudes[incident_index]
        reflected_coefficients = left_projection.amplitudes[mode_count:]
        transmitted_coefficients = right_projection.amplitudes[:mode_count]
        reflected_amplitudes = reflected_coefficients / prescribed_incoming
        transmitted_amplitudes = transmitted_coefficients / prescribed_incoming
        s_parameters = {
            **{("left", index, incident_index): complex(value) for index, value in enumerate(reflected_amplitudes)},
            **{("right", index, incident_index): complex(value) for index, value in enumerate(transmitted_amplitudes)},
        }
        propagating = np.asarray(
            [index for index, mode in enumerate(forward_modes) if mode.is_propagating],
            dtype=np.int64,
        )

        forward_gram = right_projection.gram_matrix[:mode_count, :mode_count]
        backward_gram = left_projection.gram_matrix[mode_count:, mode_count:]
        if propagating.size:
            forward_diagonal = np.real(np.diag(forward_gram)[propagating])
            backward_diagonal = np.real(np.diag(backward_gram)[propagating])
            if np.any(forward_diagonal <= 0.0) or np.any(backward_diagonal >= 0.0):
                raise ModeProjectionError(
                    "A propagating modal trace has the wrong signed diagonal power."
                )
            forward_gram_diagonal_error = float(
                np.max(np.abs(forward_diagonal - 1.0))
            )
            backward_gram_diagonal_error = float(
                np.max(np.abs(backward_diagonal + 1.0))
            )
        else:
            forward_gram_diagonal_error = 0.0
            backward_gram_diagonal_error = 0.0
        reflected_power_raw = -modal_power_from_gram(
            reflected_coefficients,
            backward_gram,
            indices=propagating,
            normalize_diagonal=True,
        )
        transmitted_power_raw = modal_power_from_gram(
            transmitted_coefficients,
            forward_gram,
            indices=propagating,
            normalize_diagonal=True,
        )
        reflected_power = max(reflected_power_raw, 0.0)
        transmitted_power = max(transmitted_power_raw, 0.0)
        incident_power = float(abs(prescribed_incoming) ** 2)

        coordinates_grid = system.physical_coordinates()
        et, ey = scattered.field.interpolate()
        E_sc_grid = np.asarray(electric_field_vector(et, ey), dtype=np.complex128)
        curl_sc = np.asarray(
            modified_curl(et, ey, self.ky * length_scale), dtype=np.complex128
        ) / length_scale
        mu_diagonal = evaluate_diagonal_coefficient(
            self._mu_actual,
            coordinates_grid[0],
            coordinates_grid[1],
            name="mu_r",
        )
        H_sc_grid = curl_sc / (1j * self.frequency.omega * MU_0 * mu_diagonal)
        E_inc_grid, H_inc_grid = self.incident.fields(
            coordinates_grid[0], coordinates_grid[1]
        )
        E_total_grid = E_inc_grid + E_sc_grid
        H_total_grid = H_inc_grid + H_sc_grid

        central = (
            (coordinates_grid[0] >= x0)
            & (coordinates_grid[0] <= x1)
            & (coordinates_grid[1] >= self.left_monitor)
            & (coordinates_grid[1] <= self.right_monitor)
        )
        eps_physical, mu_physical = self._physical_material(
            coordinates_grid[0], coordinates_grid[1], profile="actual"
        )
        density = 0.5 * self.frequency.omega * (
            EPSILON_0 * np.imag(eps_physical) * np.sum(np.abs(E_total_grid) ** 2, axis=0)
            + MU_0 * np.imag(mu_physical) * np.sum(np.abs(H_total_grid) ** 2, axis=0)
        )
        absorbed_raw = float(np.real(np.sum(system.basis.dx * length_scale**2 * density * central)))
        absorbed_power = max(absorbed_raw, 0.0)

        left_face_mask = (left_sc.x >= x0) & (left_sc.x <= x1)
        right_face_mask = (right_sc.x >= x0) & (right_sc.x <= x1)
        outward = -self._poynting_z(
            left_E[:, left_face_mask],
            left_H[:, left_face_mask],
            left_sc.weights[left_face_mask],
        )
        outward += self._poynting_z(
            right_E[:, right_face_mask],
            right_H[:, right_face_mask],
            right_sc.weights[right_face_mask],
        )
        if self.pml.x is not None:
            left_side = sample_horizontal_monitor(
                system.basis,
                coefficients,
                x=x0,
                ky=self.ky,
                omega=self.frequency.omega,
                mu_r=self._mu_actual,
                length_scale=length_scale,
                intorder=self.solver_options.quadrature_order,
            )
            right_side = sample_horizontal_monitor(
                system.basis,
                coefficients,
                x=x1,
                ky=self.ky,
                omega=self.frequency.omega,
                mu_r=self._mu_actual,
                length_scale=length_scale,
                intorder=self.solver_options.quadrature_order,
            )

            def side_total(samples: object) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.bool_]]:
                inc_e, inc_h = self.incident.fields(np.full_like(samples.z, samples.x), samples.z)  # type: ignore[attr-defined]
                mask = (samples.z >= self.left_monitor) & (samples.z <= self.right_monitor)  # type: ignore[attr-defined]
                return samples.E + inc_e, samples.H + inc_h, mask  # type: ignore[attr-defined]

            side_e, side_h, mask = side_total(left_side)
            outward -= self._poynting_x(side_e[:, mask], side_h[:, mask], left_side.weights[mask])
            side_e, side_h, mask = side_total(right_side)
            outward += self._poynting_x(side_e[:, mask], side_h[:, mask], right_side.weights[mask])
        radiated_raw = outward + incident_power - reflected_power - transmitted_power
        radiated_power = (
            max(float(radiated_raw), 0.0) if self.pml.x is not None else 0.0
        )
        energy_residual = abs(outward + absorbed_power) / incident_power

        sample_mask = central.reshape(-1)
        coordinates_flat = coordinates_grid.reshape(2, -1)[:, sample_mask]
        result = ScatteringResult(
            coordinates=coordinates_flat,
            E_incident=E_inc_grid.reshape(3, -1)[:, sample_mask],
            E_scattered=E_sc_grid.reshape(3, -1)[:, sample_mask],
            H_incident=H_inc_grid.reshape(3, -1)[:, sample_mask],
            H_scattered=H_sc_grid.reshape(3, -1)[:, sample_mask],
            s_parameters=s_parameters,
            reflected_power=reflected_power,
            transmitted_power=transmitted_power,
            radiated_power=radiated_power,
            absorbed_power=absorbed_power,
            incident_power=incident_power,
            ndofs=system.ndofs,
            solve_info={
                **dict(scattered.field.solve_info),
                "angle_degrees": self.angle,
                "ky": self.ky,
                "length_scale": length_scale,
                "source_active_fraction": scattered.source.active_quadrature_fraction,
                "released_pec_facet_count": scattered.source.released_pec_facet_count,
                "inserted_pec_facet_count": scattered.source.inserted_pec_facet_count,
                "left_projection_residual": left_projection.relative_residual,
                "right_projection_residual": right_projection.relative_residual,
                "projected_incoming_amplitude": projected_incoming,
                "prescribed_incoming_amplitude": prescribed_incoming,
                "incoming_projection_relative_error": abs(
                    projected_incoming - prescribed_incoming
                )
                / abs(prescribed_incoming),
                "independent_energy_residual": energy_residual,
                "raw_radiated_power": radiated_raw,
                "raw_absorbed_power": absorbed_raw,
                "raw_reflected_modal_power": reflected_power_raw,
                "raw_transmitted_modal_power": transmitted_power_raw,
                "forward_port_gram_diagonal_error": forward_gram_diagonal_error,
                "backward_port_gram_diagonal_error": backward_gram_diagonal_error,
            },
            mesh_info=asdict(self.mesh_data.info),
            projection_condition_numbers={
                "left": left_projection.condition_number,
                "right": right_projection.condition_number,
            },
            reference_planes={
                "left": self.incident.reference_plane,
                "right": self.incident.reference_plane,
            },
            port_betas={
                (side, index): complex(mode.beta)
                for side in ("left", "right")
                for index, mode in enumerate(forward_modes)
            },
            frequency_hz=self.frequency.frequency,
            ky=self.ky,
            modes=forward_modes,
            scene=self._visualization_scene(),
        )
        if h5_path is not None:
            written = result.save_h5(h5_path)
            result = replace(result, h5_path=written)
        return result

    def run(
        self,
        *,
        h5_path: str | PathLike[str] = "wavefem_result.h5",
    ) -> ScatteringResult:
        """Solve and always persist a complete single-run HDF5 result."""

        if h5_path is None:
            raise ConfigurationError(
                "run() requires an HDF5 destination; use solve() for an "
                "explicitly in-memory result."
            )
        return self.solve(h5_path=h5_path)


__all__ = ["Scattering2D", "SolverOptions"]
