"""Public two-dimensional full-vector finite-element mode solver.

The geometry is continuous until :meth:`ModeSolver2D.discretize` is called.
This is the important semantic difference from the legacy Yee-grid solver:
placing an object does not rasterize it, and changing the scene invalidates the
mesh rather than modifying material arrays in place.

Fields use ``exp(+1j*omega*t - 1j*beta*z)`` throughout.  Consequently a
forward passive mode has ``Im(beta) <= 0`` and attenuation ``-Im(beta)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .assembly import (
    ModeFEMSystem2D,
    evaluate_material,
    assemble_mode_system_2d,
    solve_qep_candidates,
)
from .boundaries import (
    good_conductor_surface_impedance,
    validate_surface_impedance,
)
from .constants import C_0, ETA_0
from .exceptions import (
    BackendCapabilityError,
    ConfigurationError,
    NotDiscretizedError,
    SolverError,
    StaleDiscretizationError,
)
from .geometry import (
    BoundaryRegion,
    Circle,
    GeometryModel2D,
    MeshRefinement,
    PMLSpec,
    Polygon,
    Rectangle,
    Region,
    Shape2D,
)
from .materials import Material, MaterialInput
from .meshing import FEMMesh2D, discretize_2d
from .results import Mode, ModeSet, SampledFields


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
Direction = Literal["forward", "backward", "all"]

_COMPONENT_ORDER = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_PML_DIRECTIONS = {"x-", "x+", "x", "y-", "y+", "y", "all"}


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or not np.isscalar(value)
    ):
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(
            f"{name} must be an integer of at least {minimum}."
        ) from exc
    if result != value or result < minimum:
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    return result


def _positive_real(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or not np.isscalar(value)
    ):
        raise ConfigurationError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be finite and positive.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"{name} must be finite and positive.")
    return result


def _finite_complex(value: Any, name: str) -> complex:
    result = complex(value)
    if not np.isfinite((result.real, result.imag)).all():
        raise ConfigurationError(f"{name} must be finite.")
    return result


def _optional_refinement_factor(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ConfigurationError(f"{name} must be in (0, 1] or None.")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result <= 1.0:
        raise ConfigurationError(f"{name} must be in (0, 1] or None.")
    return result


def _forward_root(value: complex) -> complex:
    """Choose the ``+z`` branch for the package's time convention."""

    root = complex(np.sqrt(complex(value)))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(root))
    if root.real < -tolerance or (
        abs(root.real) <= tolerance and root.imag > tolerance
    ):
        root = -root
    return root


class ModeSolver2D:
    """Full-vector 2D FEM waveguide mode solver.

    Parameters are physical SI values.  A scalar ``x_range`` or ``y_range``
    denotes ``(0, extent)``; a pair denotes explicit lower and upper bounds.
    Geometry can be placed in any order before an explicit call to
    :meth:`discretize`.
    """

    def __init__(
        self,
        frequency: float,
        x_range: float | Sequence[float],
        y_range: float | Sequence[float],
        num_modes: int = 4,
        neff_guess: complex | None = None,
        *,
        guess: complex | None = None,
        background_epsilon: MaterialInput = 1.0,
        background_mu: MaterialInput = 1.0,
        boundary: str = "pec",
    ) -> None:
        self.frequency = _positive_real(frequency, "frequency")
        self.omega = 2.0 * np.pi * self.frequency
        self.k0 = self.omega / C_0
        self.k_0 = self.k0  # legacy spelling
        self.num_modes = _positive_integer(num_modes, "num_modes")
        if neff_guess is not None and guess is not None:
            raise ConfigurationError("Supply only one of neff_guess or the guess alias.")
        requested_guess = neff_guess if neff_guess is not None else guess
        self.neff_guess = (
            None
            if requested_guess is None
            else _finite_complex(requested_guess, "neff_guess")
        )
        # Retained as a read-only-by-convention compatibility view.
        self.guess = self.neff_guess

        self.geometry = GeometryModel2D(
            x_range,
            y_range,
            Material(background_epsilon, background_mu),
        )
        self.geometry.set_outer_boundary(boundary)
        self.x_span = self.geometry.x_span
        self.y_span = self.geometry.y_span
        self.x_range = self.x_span[1] - self.x_span[0]
        self.y_range = self.y_span[1] - self.y_span[0]

        self._mesh_data: FEMMesh2D | None = None
        self._system: ModeFEMSystem2D | None = None
        self._discretized_revision: int | None = None
        self._result: ModeSet | None = None
        self._discretization_settings: dict[str, object] | None = None
        self._invalidate_solution()
        self.geometry.add_change_listener(self._geometry_changed)

    # ------------------------------------------------------------------
    # Continuous geometry
    # ------------------------------------------------------------------
    def _invalidate_solution(self) -> None:
        self._result = None
        self.modes: ModeSet | None = None
        self.eigenvalues: ComplexArray | None = None
        self.eigenvectors: ComplexArray | None = None
        self.coefficients: ComplexArray | None = None
        self.neff: ComplexArray | None = None
        self.beta: ComplexArray | None = None
        self.propagation_constant: FloatArray | None = None
        self.attenuation_constant: FloatArray | None = None
        self.alpha: FloatArray | None = None
        for name in _COMPONENT_ORDER:
            setattr(self, name, None)

    def _invalidate_discretization(self) -> None:
        self._mesh_data = None
        self._system = None
        self._discretized_revision = None
        self._invalidate_solution()

    def _geometry_changed(self) -> None:
        self._invalidate_discretization()

    @staticmethod
    def _unused_subpixels(subpixels: int | None) -> None:
        if subpixels is None:
            return
        _positive_integer(subpixels, "subpixels")

    def add_rectangle(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        x_range: Sequence[float],
        y_range: Sequence[float],
        *,
        subpixels: int | None = None,
        name: str | None = None,
    ) -> Region:
        """Place a conformingly meshed rectangular material region."""

        self._unused_subpixels(subpixels)
        handle = self.geometry.add_region(
            Rectangle(tuple(x_range), tuple(y_range)),
            Material(epsilon, mu),
            name=name,
        )
        self._geometry_changed()
        return handle

    def add_circle(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        center: Sequence[float],
        r1: float,
        r2: float | None = None,
        *,
        subpixels: int | None = None,
        name: str | None = None,
    ) -> Region:
        """Place a circular or annular material region."""

        self._unused_subpixels(subpixels)
        handle = self.geometry.add_region(
            Circle(tuple(center), r1, r2),  # type: ignore[arg-type]
            Material(epsilon, mu),
            name=name,
        )
        self._geometry_changed()
        return handle

    def add_polygon(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        points: Sequence[Sequence[float]],
        *,
        name: str | None = None,
    ) -> Region:
        """Place an arbitrary simple polygon material region."""

        vertices = tuple((float(point[0]), float(point[1])) for point in points)
        handle = self.geometry.add_region(
            Polygon(vertices), Material(epsilon, mu), name=name
        )
        self._geometry_changed()
        return handle

    def add_triangle(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        p1: Sequence[float],
        p2: Sequence[float],
        p3: Sequence[float],
        *,
        subpixels: int | None = None,
        name: str | None = None,
    ) -> Region:
        """Place a triangular material region."""

        self._unused_subpixels(subpixels)
        return self.add_polygon(epsilon, mu, (p1, p2, p3), name=name)

    def add_mesh_refinement(
        self,
        shape: Shape2D,
        max_element_size: float,
        *,
        transition_width: float = 0.0,
        name: str | None = None,
    ) -> MeshRefinement:
        """Place a local mesh-size control without changing the physics.

        ``shape`` may be a :class:`Rectangle`, :class:`Circle`, or
        :class:`Polygon`.  The shape becomes an exact OCC partition during
        :meth:`discretize`; it does not create a material or boundary object.
        """

        if isinstance(transition_width, (bool, np.bool_)):
            raise ConfigurationError(
                "transition_width must be finite and nonnegative."
            )
        handle = self.geometry.add_mesh_refinement(
            shape,
            _positive_real(max_element_size, "max_element_size"),
            transition_width=(
                0.0
                if transition_width == 0.0
                else _positive_real(transition_width, "transition_width")
            ),
            name=name,
        )
        self._geometry_changed()
        return handle

    def _add_wall(
        self,
        kind: str,
        x_range: Sequence[float],
        y_range: Sequence[float],
        *,
        components: Sequence[str] | str | None,
        name: str | None,
    ) -> BoundaryRegion:
        if components is not None:
            raise BackendCapabilityError(
                "Component-selective conductor masks are a Yee-grid feature; "
                "the FEM backend applies PEC/PMC trace conditions to the complete wall."
            )
        handle = self.geometry.add_boundary(
            Rectangle(tuple(x_range), tuple(y_range)), kind, name=name
        )
        self._geometry_changed()
        return handle

    def add_pec(
        self,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        components: Sequence[str] | str | None = None,
        *,
        name: str | None = None,
    ) -> BoundaryRegion | None:
        """Add an internal PEC object, or select PEC for the outer wall."""

        if x_range is None and y_range is None:
            if components is not None or name is not None:
                raise ConfigurationError(
                    "components/name apply only to an internal PEC rectangle."
                )
            self.set_outer_boundary("pec")
            return None
        if x_range is None or y_range is None:
            raise ConfigurationError(
                "Provide both x_range and y_range for an internal PEC object."
            )

        return self._add_wall(
            "pec", x_range, y_range, components=components, name=name
        )

    def add_pmc(
        self,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        components: Sequence[str] | str | None = None,
        *,
        name: str | None = None,
    ) -> BoundaryRegion | None:
        """Add an internal PMC object, or select PMC for the outer wall."""

        if x_range is None and y_range is None:
            if components is not None or name is not None:
                raise ConfigurationError(
                    "components/name apply only to an internal PMC rectangle."
                )
            self.set_outer_boundary("pmc")
            return None
        if x_range is None or y_range is None:
            raise ConfigurationError(
                "Provide both x_range and y_range for an internal PMC object."
            )

        return self._add_wall(
            "pmc", x_range, y_range, components=components, name=name
        )

    def add_impedance_surface(
        self,
        Zs: complex | None = None,
        *,
        preset: str | None = None,
        x_range: Sequence[float],
        y_range: Sequence[float],
        name: str | None = None,
    ) -> BoundaryRegion:
        """Add an opaque conductor whose exposed facets obey scalar SIBC.

        Supply exactly one of ``Zs`` (ohms) or a good-conductor metal
        ``preset`` such as ``"Cu"``.  Presets are evaluated at this solver's
        frequency using the package's ``exp(+j*omega*t)`` convention.
        """

        if (Zs is None) == (preset is None):
            raise ConfigurationError("Provide exactly one of Zs or preset.")
        impedance = (
            good_conductor_surface_impedance(preset, self.frequency)
            if preset is not None
            else validate_surface_impedance(Zs)  # type: ignore[arg-type]
        )
        handle = self.geometry.add_boundary(
            Rectangle(tuple(x_range), tuple(y_range)),
            "impedance",
            impedance=impedance,
            name=name,
        )
        self._geometry_changed()
        return handle

    def add_pml(
        self,
        pml_width: float,
        n: int = 3,
        sigma_max: float = 5.0,
        direction: str = "all",
    ) -> PMLSpec:
        """Add a physical-width uniaxial PML to selected exterior side(s)."""

        width = _positive_real(pml_width, "pml_width")
        normalized = str(direction).strip().lower()
        if normalized == "both":
            normalized = "all"
        if normalized not in _PML_DIRECTIONS:
            choices = ", ".join(sorted(_PML_DIRECTIONS))
            raise ConfigurationError(f"direction must be one of {choices}.")
        order = _positive_integer(n, "n")
        loss = float(sigma_max)
        if not np.isfinite(loss) or loss < 0.0:
            raise ConfigurationError("sigma_max must be finite and nonnegative.")

        xwidth = self.x_span[1] - self.x_span[0]
        ywidth = self.y_span[1] - self.y_span[0]
        if normalized in ("x", "all") and 2.0 * width >= xwidth:
            raise ConfigurationError("Opposing x PMLs must leave a nonempty physical region.")
        if normalized in ("y", "all") and 2.0 * width >= ywidth:
            raise ConfigurationError("Opposing y PMLs must leave a nonempty physical region.")
        if normalized in ("x-", "x+") and width >= xwidth:
            raise ConfigurationError("The x PML must be thinner than the domain.")
        if normalized in ("y-", "y+") and width >= ywidth:
            raise ConfigurationError("The y PML must be thinner than the domain.")

        spec = self.geometry.add_pml(
            PMLSpec(width, order=order, sigma_max=loss, direction=normalized)
        )
        self._geometry_changed()
        return spec

    def add_UPML(
        self,
        pml_width: float,
        n: int = 3,
        sigma_max: float = 5.0,
        direction: str = "all",
    ) -> PMLSpec:
        """Backward-compatible spelling of :meth:`add_pml`."""

        return self.add_pml(pml_width, n=n, sigma_max=sigma_max, direction=direction)

    def set_outer_boundary(self, kind: str) -> None:
        self.geometry.set_outer_boundary(kind)
        self._geometry_changed()

    def remove(
        self, handle: Region | BoundaryRegion | MeshRefinement | PMLSpec
    ) -> None:
        """Remove a placed object and invalidate any existing mesh."""

        if isinstance(handle, PMLSpec):
            try:
                self.geometry.pmls.remove(handle)
            except ValueError as exc:
                raise ConfigurationError(
                    "The PML handle does not belong to this solver."
                ) from exc
            self.geometry._changed()
        else:
            self.geometry.remove(handle)
        self._geometry_changed()

    # ------------------------------------------------------------------
    # Explicit discretization
    # ------------------------------------------------------------------
    @property
    def discretized(self) -> bool:
        return self._mesh_data is not None and self._system is not None

    @property
    def mesh_data(self) -> FEMMesh2D:
        if self._mesh_data is None:
            raise NotDiscretizedError("Call discretize() before requesting the FEM mesh.")
        return self._mesh_data

    @property
    def mesh(self) -> FEMMesh2D:
        """Return the common FEM mesh wrapper used by ``discretize()``."""

        return self.mesh_data

    @property
    def native_mesh(self) -> Any:
        """Return the underlying scikit-fem triangular mesh."""

        return self.mesh_data.mesh

    @property
    def system(self) -> ModeFEMSystem2D:
        if self._system is None:
            raise NotDiscretizedError("Call discretize() before requesting the FEM system.")
        return self._system

    @property
    def result(self) -> ModeSet:
        if self._result is None:
            raise SolverError("No FEM modal result is available; call solve() first.")
        return self._result

    @property
    def solution(self) -> ModeSet | None:
        """ModeSet hook consumed by the backend-neutral visualizers."""

        return self._result

    def visualize(self, mode: int | Mode = 1, **kwargs: Any) -> Any:
        """Plot sampled FEM fields using the common static visualizer."""

        if self.solution is None:
            raise RuntimeError("solve() must be called before visualize().")
        from .visualization import visualize

        return visualize(self, mode=mode, **kwargs)

    def visualize_with_gui(self, **kwargs: Any) -> Any:
        """Open the same interactive field viewer exposed by legacy solvers."""

        if self.solution is None:
            raise RuntimeError("solve() must be called before visualize_with_gui().")
        from .visualization import visualize_with_gui

        return visualize_with_gui(self, **kwargs)

    def _background_refractive_index(self) -> float:
        item = self.geometry.background
        epsilon = np.asarray(item.eps_r, dtype=np.complex128)
        permeability = np.asarray(item.mu_r, dtype=np.complex128)
        return float(np.sqrt(np.max(np.abs(epsilon * permeability))))

    def discretize(
        self,
        *,
        max_element_size: float | None = None,
        resolution: tuple[int, int] | None = None,
        wavelength_elements: int = 10,
        material_aware: bool = True,
        interface_refinement: float | None = None,
        interface_refinement_width: float | None = None,
        boundary_refinement: float | None = 0.5,
        boundary_refinement_width: float | None = None,
        element_order: int = 1,
        quadrature_order: int = 4,
    ) -> FEMMesh2D:
        """Mesh the scene and assemble its analytic quadratic pencil.

        Material-aware sizing is enabled by default: higher local propagation
        wavenumber produces smaller elements while ``max_element_size`` stays
        a global characteristic-size target.  ``interface_refinement`` and
        ``boundary_refinement`` are optional size multipliers in ``(0, 1]``;
        ``None`` disables the corresponding distance field.  PEC, PMC, and
        impedance walls all participate in boundary sizing.
        """

        return self._discretize(
            max_element_size=max_element_size,
            resolution=resolution,
            wavelength_elements=wavelength_elements,
            material_aware=material_aware,
            interface_refinement=interface_refinement,
            interface_refinement_width=interface_refinement_width,
            boundary_refinement=boundary_refinement,
            boundary_refinement_width=boundary_refinement_width,
            element_order=element_order,
            quadrature_order=quadrature_order,
            refinement_scale=1.0,
        )

    def _discretize(
        self,
        *,
        max_element_size: float | None,
        resolution: tuple[int, int] | None,
        wavelength_elements: int,
        material_aware: bool,
        interface_refinement: float | None,
        interface_refinement_width: float | None,
        boundary_refinement: float | None,
        boundary_refinement_width: float | None,
        element_order: int,
        quadrature_order: int,
        refinement_scale: float,
    ) -> FEMMesh2D:

        wavelength_count = _positive_integer(
            wavelength_elements, "wavelength_elements", minimum=4
        )
        order = _positive_integer(element_order, "element_order")
        quadrature = _positive_integer(
            quadrature_order, "quadrature_order", minimum=2
        )
        if not isinstance(material_aware, (bool, np.bool_)):
            raise ConfigurationError("material_aware must be a boolean.")
        interface_factor = _optional_refinement_factor(
            interface_refinement, "interface_refinement"
        )
        boundary_factor = _optional_refinement_factor(
            boundary_refinement, "boundary_refinement"
        )
        interface_width = (
            None
            if interface_refinement_width is None
            else _positive_real(
                interface_refinement_width, "interface_refinement_width"
            )
        )
        boundary_width = (
            None
            if boundary_refinement_width is None
            else _positive_real(
                boundary_refinement_width, "boundary_refinement_width"
            )
        )
        if interface_width is not None and interface_factor is None:
            raise ConfigurationError(
                "interface_refinement_width requires interface_refinement."
            )
        if boundary_width is not None and boundary_factor is None:
            raise ConfigurationError(
                "boundary_refinement_width requires boundary_refinement."
            )
        if max_element_size is not None:
            maximum = _positive_real(max_element_size, "max_element_size")
        elif resolution is None:
            width = self.x_span[1] - self.x_span[0]
            height = self.y_span[1] - self.y_span[0]
            wavelength_size = C_0 / self.frequency / (
                wavelength_count * max(self._background_refractive_index(), 1.0)
            )
            maximum = min(wavelength_size, min(width, height) / 12.0)
        else:
            maximum = None

        unsupported = [
            boundary
            for boundary in self.geometry.boundaries
            if boundary.kind not in ("pec", "pmc", "impedance")
        ]
        if unsupported:
            kinds = ", ".join(sorted({item.kind for item in unsupported}))
            raise ConfigurationError(
                f"Unsupported internal FEM boundary kind(s): {kinds}."
            )

        self._invalidate_discretization()
        mesh_data = discretize_2d(
            self.geometry,
            max_element_size=maximum,
            resolution=resolution,
            element_order=order,
            material_aware=bool(material_aware),
            vacuum_wavenumber=self.k0,
            wavelength_elements=wavelength_count,
            interface_refinement=interface_factor,
            interface_refinement_width=interface_width,
            boundary_refinement=boundary_factor,
            boundary_refinement_width=boundary_width,
            _refinement_scale=refinement_scale,
        )
        pec_parts = [
            mesh_data.boundary_facets[name]
            for name in ("outer_pec", "pec")
            if name in mesh_data.boundary_facets
            and mesh_data.boundary_facets[name].size
        ]
        pec_facets = (
            np.unique(np.concatenate(pec_parts)).astype(np.int64, copy=False)
            if pec_parts
            else np.empty(0, dtype=np.int64)
        )
        impedance_boundaries: list[tuple[NDArray[np.int64], complex]] = []
        for boundary_region in self.geometry.boundaries:
            if boundary_region.kind != "impedance":
                continue
            if boundary_region.impedance is None:  # defensive geometry invariant
                raise ConfigurationError(
                    f"Impedance boundary {boundary_region.name!r} has no Zs value."
                )
            facets = mesh_data.boundary_facets.get(boundary_region.name)
            if facets is None or facets.size == 0:
                raise ConfigurationError(
                    f"Impedance boundary {boundary_region.name!r} has no exposed mesh facets."
                )
            impedance_boundaries.append(
                (
                    np.asarray(facets, dtype=np.int64),
                    validate_surface_impedance(boundary_region.impedance),
                )
            )
        evaluator = (
            self.geometry.transformed_material_at
            if self.geometry.pmls
            else self.geometry.material_at
        )
        system = assemble_mode_system_2d(
            mesh_data.mesh,
            frequency=self.frequency,
            k0=self.k0,
            material_at=evaluator,
            boundary=self.geometry.outer_boundary,
            quadrature_order=quadrature,
            pec_facets=pec_facets,
            impedance_boundaries=impedance_boundaries,
        )
        self._mesh_data = mesh_data
        self._system = system
        self._discretized_revision = self.geometry.revision
        self._discretization_settings = {
            "max_element_size": maximum,
            "resolution": resolution,
            "wavelength_elements": wavelength_count,
            "material_aware": bool(material_aware),
            "interface_refinement": interface_factor,
            "interface_refinement_width": interface_width,
            "boundary_refinement": boundary_factor,
            "boundary_refinement_width": boundary_width,
            "element_order": order,
            "quadrature_order": quadrature,
            "refinement_scale": refinement_scale,
        }
        return mesh_data

    def refine(self, factor: float = 2.0) -> FEMMesh2D:
        """Remesh with ``factor`` times the density and rebuild the system.

        All previous discretization options are retained.  Global,
        wavelength, interface, boundary, and explicit local size targets are
        scaled consistently; any previous modal solution is invalidated.
        """

        density_factor = _positive_real(factor, "refinement factor")
        if not np.isfinite(density_factor) or density_factor <= 1.0:
            raise ConfigurationError(
                "refinement factor must be finite and greater than one."
            )
        self._require_current_system()
        if self._discretization_settings is None:  # defensive invariant
            raise NotDiscretizedError("Call discretize() before refine().")
        settings = dict(self._discretization_settings)
        maximum = settings["max_element_size"]
        resolution = settings["resolution"]
        if maximum is not None:
            maximum = float(maximum) / density_factor
        if resolution is not None:
            resolution = tuple(
                int(np.ceil(int(value) * density_factor))
                for value in resolution  # type: ignore[union-attr]
            )
        wavelength_count = int(
            np.ceil(int(settings["wavelength_elements"]) * density_factor)
        )
        return self._discretize(
            max_element_size=maximum,
            resolution=resolution,  # type: ignore[arg-type]
            wavelength_elements=wavelength_count,
            material_aware=bool(settings["material_aware"]),
            interface_refinement=settings["interface_refinement"],  # type: ignore[arg-type]
            interface_refinement_width=settings["interface_refinement_width"],  # type: ignore[arg-type]
            boundary_refinement=settings["boundary_refinement"],  # type: ignore[arg-type]
            boundary_refinement_width=settings["boundary_refinement_width"],  # type: ignore[arg-type]
            element_order=int(settings["element_order"]),
            quadrature_order=int(settings["quadrature_order"]),
            refinement_scale=float(settings["refinement_scale"]) / density_factor,
        )

    def _require_current_system(self) -> ModeFEMSystem2D:
        if self._system is None or self._mesh_data is None:
            raise NotDiscretizedError(
                "The continuous scene has not been discretized; call discretize() before solve()."
            )
        if self._discretized_revision != self.geometry.revision:
            raise StaleDiscretizationError(
                "The geometry changed after discretization; call discretize() again."
            )
        return self._system

    # ------------------------------------------------------------------
    # Field reconstruction and eigenmode filtering
    # ------------------------------------------------------------------
    def _default_guess(self, system: ModeFEMSystem2D) -> complex:
        coordinates = system.basis.global_coordinates() / self.k0
        epsilon, mu = evaluate_material(
            self.geometry.material_at, coordinates[0], coordinates[1]
        )
        index_squared = epsilon * mu
        candidate = index_squared.reshape(3, -1)
        selected = candidate.ravel()[int(np.argmax(np.abs(candidate)))]
        return _forward_root(complex(selected))

    def _physical_integration_mask(self, coordinates: FloatArray) -> NDArray[np.bool_]:
        mask = np.ones(coordinates.shape[1:], dtype=bool)
        x, y = coordinates
        xmin, xmax = self.x_span
        ymin, ymax = self.y_span
        for pml in self.geometry.pmls:
            if pml.direction in ("x-", "x", "all"):
                mask &= x >= xmin + pml.thickness
            if pml.direction in ("x+", "x", "all"):
                mask &= x <= xmax - pml.thickness
            if pml.direction in ("y-", "y", "all"):
                mask &= y >= ymin + pml.thickness
            if pml.direction in ("y+", "y", "all"):
                mask &= y <= ymax - pml.thickness
        return mask

    def _field_data(
        self,
        system: ModeFEMSystem2D,
        full_vector: ComplexArray,
        neff: complex,
    ) -> dict[str, Any]:
        transverse, longitudinal = system.basis.interpolate(full_vector)
        et = np.asarray(transverse, dtype=np.complex128)
        ez = np.asarray(longitudinal, dtype=np.complex128)
        grad_ez = np.asarray(longitudinal.grad, dtype=np.complex128)
        curl_et = np.asarray(transverse.curl, dtype=np.complex128)

        coordinates = np.asarray(
            system.basis.global_coordinates() / self.k0, dtype=np.float64
        )
        epsilon, mu = evaluate_material(
            system.material_at, coordinates[0], coordinates[1]
        )
        inv_mu = 1.0 / mu
        ex, ey = et[0], et[1]
        curl_x = grad_ez[1] + 1j * neff * ey
        curl_y = -1j * neff * ex - grad_ez[0]
        curl_z = curl_et
        hx = 1j / ETA_0 * inv_mu[0] * curl_x
        hy = 1j / ETA_0 * inv_mu[1] * curl_y
        hz = 1j / ETA_0 * inv_mu[2] * curl_z

        physical_weights = np.asarray(system.basis.dx / self.k0**2, dtype=float)
        physical_weights *= self._physical_integration_mask(coordinates)
        poynting = 0.5 * np.sum(
            physical_weights * (ex * np.conj(hy) - ey * np.conj(hx))
        )
        e_energy = np.sum(
            physical_weights
            * (
                np.abs(epsilon[0]) * np.abs(ex) ** 2
                + np.abs(epsilon[1]) * np.abs(ey) ** 2
                + np.abs(epsilon[2]) * np.abs(ez) ** 2
            )
        )
        h_energy = np.sum(
            physical_weights
            * ETA_0**2
            * (
                np.abs(mu[0]) * np.abs(hx) ** 2
                + np.abs(mu[1]) * np.abs(hy) ** 2
                + np.abs(mu[2]) * np.abs(hz) ** 2
            )
        )
        energy_like = float(max((e_energy + h_energy).real, 0.0))
        fields = np.stack((ex, ey, ez, hx, hy, hz), axis=-1)
        return {
            "coordinates": coordinates,
            "fields": np.asarray(fields, dtype=np.complex128),
            "epsilon": epsilon,
            "power": complex(poynting),
            "energy_like": energy_like,
        }

    @staticmethod
    def _polarization(field_values: ComplexArray) -> str:
        electric = np.sum(np.abs(field_values[..., :3]) ** 2, axis=(0, 1))
        magnetic = np.sum(np.abs(field_values[..., 3:]) ** 2, axis=(0, 1))
        e_total = float(np.sum(electric))
        h_total = float(np.sum(magnetic))
        ez_fraction = 0.0 if e_total == 0.0 else float(electric[2] / e_total)
        hz_fraction = 0.0 if h_total == 0.0 else float(magnetic[2] / h_total)
        threshold = 1e-7
        if ez_fraction <= threshold:
            return "TE"
        if hz_fraction <= threshold:
            return "TM"
        return "hybrid"

    def _make_mode(
        self,
        system: ModeFEMSystem2D,
        reduced_vector: ComplexArray,
        neff: complex,
        residual: float,
        divergence_residual: float,
        *,
        propagation_ratio_tolerance: float,
        method: str,
        index: int,
    ) -> tuple[Mode, ComplexArray, str]:
        full = system.expand(reduced_vector)
        data = self._field_data(system, full, neff)
        complex_power = complex(data["power"])
        real_power = float(complex_power.real)
        propagating = (
            abs(complex_power) > np.finfo(float).tiny
            and abs(real_power) / abs(complex_power) >= propagation_ratio_tolerance
        )
        if propagating:
            scale = 1.0 / np.sqrt(abs(real_power))
            normalization = "unit-power"
            classification = "propagating"
            direction = "forward" if real_power > 0.0 else "backward"
        else:
            if data["energy_like"] <= np.finfo(float).tiny:
                raise SolverError("A FEM eigenvector has zero flux and zero energy-like norm.")
            scale = 1.0 / np.sqrt(float(data["energy_like"]))
            normalization = "energy-like"
            classification = "evanescent"
            tolerance = 1e-10 * max(1.0, abs(neff))
            if neff.imag < -tolerance:
                direction = "right-decaying"
            elif neff.imag > tolerance:
                direction = "left-decaying"
            else:
                direction = "indeterminate"

        full = np.asarray(full * scale, dtype=np.complex128)
        pivot = full[int(np.argmax(np.abs(full)))]
        if abs(pivot) > 0.0:
            full *= np.exp(-1j * np.angle(pivot))
        data = self._field_data(system, full, neff)
        values = np.asarray(data["fields"], dtype=np.complex128)
        sample_shape = values.shape[:-1]
        element_count = self.mesh_data.elements.shape[0]
        if len(sample_shape) != 2 or sample_shape[0] != element_count:
            raise SolverError(
                "The FEM field sampler did not preserve the mesh-element axis; "
                f"received sample shape {sample_shape!r} for {element_count} elements."
            )
        # scikit-fem orders quadrature samples as (element, quadrature point).
        # Keep that ownership after flattening so visualization can average on
        # each native triangle instead of constructing a global Delaunay mesh,
        # which would incorrectly bridge internal PEC/PMC holes.
        sample_element_indices = np.repeat(
            np.arange(element_count, dtype=np.int64), sample_shape[1]
        )
        coordinates = np.moveaxis(data["coordinates"], 0, -1).reshape(-1, 2)
        sampled = SampledFields(
            coordinates,
            values.reshape(-1, len(_COMPONENT_ORDER)),
            dimension=2,
            mesh_points=self.mesh_data.nodes,
            mesh_cells=self.mesh_data.elements,
            material=np.asarray(data["epsilon"][2]).reshape(-1),
            metadata={
                "component_order": _COMPONENT_ORDER,
                "sampling": "element-quadrature",
                "sample_element_indices": sample_element_indices,
                "element_quadrature_shape": sample_shape,
                "time_convention": "exp(+1j*omega*t - 1j*beta*z)",
            },
        )
        polarization = self._polarization(values)
        mode = Mode(
            neff=complex(neff),
            beta=complex(self.k0 * neff),
            fields=sampled,
            index=index,
            polarization=polarization,
            eigenvalue=complex(neff),
            power=complex(data["power"]),
            normalization=normalization,
            residual=residual,
            divergence_residual=divergence_residual,
            metadata={
                "classification": classification,
                "direction": direction,
                "coefficient_vector": full,
                "eigensolver": method,
            },
        )
        return mode, full, direction

    @staticmethod
    def _direction_matches(direction: str, requested: Direction) -> bool:
        if requested == "all":
            return True
        if requested == "forward":
            return direction in ("forward", "right-decaying")
        return direction in ("backward", "left-decaying")

    def solve(
        self,
        neff_guess: complex | None = None,
        num_modes: int | None = None,
        *,
        direction: Direction = "forward",
        eigensolver_tolerance: float = 1e-10,
        residual_tolerance: float = 1e-8,
        divergence_tolerance: float = 1e-7,
        propagation_ratio_tolerance: float = 1e-3,
        dense_linearization_limit: int = 700,
    ) -> ModeSet:
        """Solve the assembled QEP and return validated full-vector modes."""

        system = self._require_current_system()
        requested = self.num_modes if num_modes is None else _positive_integer(
            num_modes, "num_modes"
        )
        if direction not in ("forward", "backward", "all"):
            raise ConfigurationError("direction must be 'forward', 'backward', or 'all'.")
        tolerances = (
            (eigensolver_tolerance, "eigensolver_tolerance"),
            (residual_tolerance, "residual_tolerance"),
            (divergence_tolerance, "divergence_tolerance"),
            (propagation_ratio_tolerance, "propagation_ratio_tolerance"),
        )
        for value, name in tolerances:
            _positive_real(value, name)
        dense_limit = _positive_integer(
            dense_linearization_limit, "dense_linearization_limit", minimum=4
        )
        raw_guess = self.neff_guess if neff_guess is None else neff_guess
        target = self._default_guess(system) if raw_guess is None else _finite_complex(
            raw_guess, "neff_guess"
        )
        candidate_count = max(4 * requested, requested + 12)
        values, vectors, method = solve_qep_candidates(
            system,
            target=target,
            candidate_count=candidate_count,
            tolerance=float(eigensolver_tolerance),
            dense_linearization_limit=dense_limit,
        )

        accepted: list[Mode] = []
        coefficients: list[ComplexArray] = []
        accepted_reduced: list[ComplexArray] = []
        rejected_residual = 0
        rejected_divergence = 0
        rejected_direction = 0
        for neff, raw_vector in zip(values, vectors.T, strict=True):
            vector_norm = float(np.linalg.norm(raw_vector))
            if not np.isfinite(neff) or vector_norm <= np.finfo(float).tiny:
                continue
            vector = np.asarray(raw_vector / vector_norm, dtype=np.complex128)
            residual = system.relative_residual(vector, complex(neff))
            if not np.isfinite(residual) or residual > residual_tolerance:
                rejected_residual += 1
                continue
            full = system.expand(vector)
            divergence = system.divergence_residual(full, complex(neff))
            if not np.isfinite(divergence) or divergence > divergence_tolerance:
                rejected_divergence += 1
                continue
            duplicate = False
            for previous_neff, previous_vector in zip(
                (item.neff for item in accepted), accepted_reduced, strict=True
            ):
                if abs(previous_neff - neff) > 1e-8 * max(1.0, abs(neff)):
                    continue
                overlap = abs(np.vdot(previous_vector, vector)) / (
                    np.linalg.norm(previous_vector) * np.linalg.norm(vector)
                )
                if overlap > 1.0 - 1e-8:
                    duplicate = True
                    break
            if duplicate:
                continue
            try:
                mode, coefficient, mode_direction = self._make_mode(
                    system,
                    vector,
                    complex(neff),
                    residual,
                    divergence,
                    propagation_ratio_tolerance=float(propagation_ratio_tolerance),
                    method=method,
                    index=len(accepted) + 1,
                )
            except SolverError:
                continue
            if not self._direction_matches(mode_direction, direction):
                rejected_direction += 1
                continue
            accepted.append(mode)
            coefficients.append(coefficient)
            accepted_reduced.append(vector)
            if len(accepted) == requested:
                break

        if len(accepted) < requested:
            raise SolverError(
                f"Requested {requested} {direction} FEM mode(s) near neff={target!r}, "
                f"but only {len(accepted)} passed validation. Rejected: "
                f"eigen-residual={rejected_residual}, Gauss-law={rejected_divergence}, "
                f"direction={rejected_direction}. Try a closer neff_guess, a finer mesh, "
                "or direction='all'."
            )

        mesh_info = asdict(self.mesh_data.info)
        result = ModeSet(
            tuple(accepted),
            frequency=self.frequency,
            k0=self.k0,
            dimension=2,
            backend="fem-nedelec-qep",
            metadata={
                "method": method,
                "candidate_count": int(values.size),
                "neff_guess": target,
                "requested_direction": direction,
                "geometry_revision": self.geometry.revision,
                "mesh": mesh_info,
                "residual_tolerance": float(residual_tolerance),
                "divergence_tolerance": float(divergence_tolerance),
                "time_convention": "exp(+1j*omega*t - 1j*beta*z)",
            },
        )
        self._result = result
        self.modes = result
        self.neff = result.neff
        self.beta = result.beta
        self.propagation_constant = result.propagation_constant
        self.attenuation_constant = result.attenuation_constant
        self.alpha = np.asarray([mode.alpha for mode in result], dtype=float)
        self.eigenvalues = -self.neff**2
        self.coefficients = np.column_stack(coefficients)
        self.eigenvectors = np.column_stack(accepted_reduced)
        for component in _COMPONENT_ORDER:
            setattr(
                self,
                component,
                np.column_stack([mode.component(component) for mode in result]),
            )
        return result


__all__ = ["ModeSolver2D"]
