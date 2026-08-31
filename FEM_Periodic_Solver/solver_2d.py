"""Public fixed-frequency 2D periodic finite-element mode solver."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .assembly_2d import (
    PeriodicFEMSystem2D,
    assemble_periodic_system_2d,
    evaluate_material,
    solve_qep_candidates,
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
from .meshing_2d import FEMPeriodicMesh2D, discretize_periodic_2d
from .results import PeriodicMode, PeriodicModeSet, PeriodicSampledFields

ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]
Direction = Literal["forward", "backward", "all"]


def _positive_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ConfigurationError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be finite and positive.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"{name} must be finite and positive.")
    return result


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.") from exc
    if not np.isfinite(numeric) or numeric != np.floor(numeric) or numeric < minimum:
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    return int(numeric)


def _finite_complex(value: Any, name: str) -> complex:
    result = complex(value)
    if not np.isfinite((result.real, result.imag)).all():
        raise ConfigurationError(f"{name} must be finite.")
    return result


class PeriodicModeSolver2D:
    """Solve complex Floquet propagation constants in one ``x-z`` unit cell."""

    def __init__(
        self,
        frequency: float,
        x_range: float | Sequence[float],
        z_range: float | Sequence[float],
        num_modes: int = 4,
        neff_guess: complex | None = None,
        *,
        polarization: str = "both",
        background_epsilon: MaterialInput = 1.0,
        background_mu: MaterialInput = 1.0,
        boundary: str = "pec",
        eigensolver: str = "auto",
        arnoldi_backend: str = "auto",
    ) -> None:
        self.frequency = _positive_real(frequency, "frequency")
        self.omega = 2.0 * np.pi * self.frequency
        self.k0 = self.omega / C_0
        self.k_0 = self.k0
        self.num_modes = _positive_integer(num_modes, "num_modes")
        self.neff_guess = None if neff_guess is None else _finite_complex(neff_guess, "neff_guess")
        selected_polarization = str(polarization).strip().upper()
        if selected_polarization not in ("TE", "TM", "BOTH"):
            raise ConfigurationError("polarization must be 'TE', 'TM', or 'both'.")
        self.polarization = selected_polarization.lower()
        self.eigensolver = str(eigensolver).strip().lower()
        if self.eigensolver not in ("auto", "dense", "refined"):
            raise ConfigurationError("eigensolver must be 'auto', 'dense', or 'refined'.")
        self.arnoldi_backend = str(arnoldi_backend).strip().lower()
        if self.arnoldi_backend not in ("auto", "cython", "python", "numpy"):
            raise ConfigurationError("arnoldi_backend must be 'auto', 'cython', or 'python'.")

        self.geometry = GeometryModel2D(
            x_range,
            z_range,
            Material(background_epsilon, background_mu),
        )
        self.geometry.set_outer_boundary(boundary)
        self.x_span = self.geometry.x_span
        self.z_span = self.geometry.z_span
        self.x_range = self.x_span[1] - self.x_span[0]
        self.z_range = self.z_span[1] - self.z_span[0]
        self.period = self.z_range
        self.geometry.add_change_listener(self._geometry_changed)
        self._mesh_data: FEMPeriodicMesh2D | None = None
        self._systems: dict[str, PeriodicFEMSystem2D] = {}
        self._discretized_revision: int | None = None
        self._discretization_settings: dict[str, Any] | None = None
        self._result: PeriodicModeSet | None = None
        self._invalidate_solution()

    def _invalidate_solution(self) -> None:
        self._result = None
        self.modes: PeriodicModeSet | None = None
        self.neff: ComplexArray | None = None
        self.beta: ComplexArray | None = None
        self.gamma: ComplexArray | None = None
        self.gammas: ComplexArray | None = None
        self.eigenvalues: ComplexArray | None = None
        self.eigenvectors: ComplexArray | None = None
        self.coefficients: ComplexArray | None = None
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            setattr(self, component, None)

    def _invalidate_discretization(self) -> None:
        self._mesh_data = None
        self._systems = {}
        self._discretized_revision = None
        self._invalidate_solution()

    def _geometry_changed(self) -> None:
        self._invalidate_discretization()

    # ------------------------------------------------------------------
    # Continuous geometry
    # ------------------------------------------------------------------
    def add_rectangle(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        x_range: Sequence[float],
        z_range: Sequence[float],
        *,
        name: str | None = None,
    ) -> Region:
        return self.geometry.add_region(
            Rectangle(tuple(x_range), tuple(z_range)),
            Material(epsilon, mu),
            name=name,
        )

    def add_circle(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        center: Sequence[float],
        radius: float,
        inner_radius: float | None = None,
        *,
        name: str | None = None,
    ) -> Region:
        return self.geometry.add_region(
            Circle(tuple(center), radius, inner_radius),  # type: ignore[arg-type]
            Material(epsilon, mu),
            name=name,
        )

    def add_polygon(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        points: Sequence[Sequence[float]],
        *,
        name: str | None = None,
    ) -> Region:
        vertices = tuple((float(point[0]), float(point[1])) for point in points)
        return self.geometry.add_region(Polygon(vertices), Material(epsilon, mu), name=name)

    def add_triangle(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        p1: Sequence[float],
        p2: Sequence[float],
        p3: Sequence[float],
        *,
        name: str | None = None,
    ) -> Region:
        return self.add_polygon(epsilon, mu, (p1, p2, p3), name=name)

    def add_mesh_refinement(
        self,
        shape: Shape2D,
        max_element_size: float,
        *,
        name: str | None = None,
    ) -> MeshRefinement:
        return self.geometry.add_mesh_refinement(
            shape, _positive_real(max_element_size, "max_element_size"), name=name
        )

    def _add_wall(
        self,
        kind: str,
        x_range: Sequence[float],
        z_range: Sequence[float],
        *,
        name: str | None,
        components: Sequence[str] | str | None,
    ) -> BoundaryRegion:
        if components is not None:
            raise BackendCapabilityError(
                "Component-selective conductor masks are not defined for the scalar FEM traces."
            )
        return self.geometry.add_boundary(
            Rectangle(tuple(x_range), tuple(z_range)), kind, name=name
        )

    def add_pec(
        self,
        x_range: Sequence[float] | None = None,
        z_range: Sequence[float] | None = None,
        components: Sequence[str] | str | None = None,
        *,
        name: str | None = None,
    ) -> BoundaryRegion | None:
        if x_range is None and z_range is None:
            if components is not None or name is not None:
                raise ConfigurationError("components/name apply only to internal PEC objects.")
            self.set_outer_boundary("pec")
            return None
        if x_range is None or z_range is None:
            raise ConfigurationError("Provide both x_range and z_range for an internal PEC.")
        return self._add_wall("pec", x_range, z_range, name=name, components=components)

    def add_pmc(
        self,
        x_range: Sequence[float] | None = None,
        z_range: Sequence[float] | None = None,
        components: Sequence[str] | str | None = None,
        *,
        name: str | None = None,
    ) -> BoundaryRegion | None:
        if x_range is None and z_range is None:
            if components is not None or name is not None:
                raise ConfigurationError("components/name apply only to internal PMC objects.")
            self.set_outer_boundary("pmc")
            return None
        if x_range is None or z_range is None:
            raise ConfigurationError("Provide both x_range and z_range for an internal PMC.")
        return self._add_wall("pmc", x_range, z_range, name=name, components=components)

    def add_pml(
        self,
        pml_width: float,
        order: int = 3,
        sigma_max: float = 5.0,
        direction: str = "x",
    ) -> PMLSpec:
        normalized = str(direction).strip().lower()
        if normalized == "both":
            normalized = "x"
        return self.geometry.add_pml(
            PMLSpec(
                _positive_real(pml_width, "pml_width"),
                _positive_integer(order, "order"),
                float(sigma_max),
                normalized,
            )
        )

    def add_UPML(
        self,
        pml_width: float,
        n: int = 3,
        sigma_max: float = 5.0,
        direction: str = "x",
    ) -> PMLSpec:
        return self.add_pml(pml_width, order=n, sigma_max=sigma_max, direction=direction)

    def set_outer_boundary(self, kind: str) -> None:
        self.geometry.set_outer_boundary(kind)

    def remove(self, handle: Region | BoundaryRegion | MeshRefinement | PMLSpec) -> None:
        self.geometry.remove(handle)

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------
    @property
    def discretized(self) -> bool:
        return self._mesh_data is not None and bool(self._systems)

    @property
    def mesh_data(self) -> FEMPeriodicMesh2D:
        if self._mesh_data is None:
            raise NotDiscretizedError("Call discretize() before requesting the mesh.")
        return self._mesh_data

    @property
    def mesh(self) -> FEMPeriodicMesh2D:
        return self.mesh_data

    @property
    def native_mesh(self) -> Any:
        return self.mesh_data.mesh

    @property
    def systems(self) -> dict[str, PeriodicFEMSystem2D]:
        self._require_current_systems()
        return dict(self._systems)

    @property
    def system(self) -> PeriodicFEMSystem2D:
        systems = self._require_current_systems()
        if len(systems) != 1:
            raise ConfigurationError("Both polarizations are assembled; use systems['TE'] or systems['TM'].")
        return next(iter(systems.values()))

    @property
    def result(self) -> PeriodicModeSet:
        if self._result is None:
            raise SolverError("Call solve() before requesting result.")
        return self._result

    @property
    def solution(self) -> PeriodicModeSet | None:
        return self._result

    def _background_index(self) -> float:
        epsilon = np.asarray(self.geometry.background.eps_r)
        mu = np.asarray(self.geometry.background.mu_r)
        return float(np.sqrt(np.max(np.abs(epsilon * mu))))

    def discretize(
        self,
        *,
        max_element_size: float | None = None,
        resolution: tuple[int, int] | None = None,
        wavelength_elements: int = 10,
        element_order: int = 1,
        quadrature_order: int = 4,
    ) -> FEMPeriodicMesh2D:
        wavelength_count = _positive_integer(wavelength_elements, "wavelength_elements", 4)
        order = _positive_integer(element_order, "element_order")
        quadrature = _positive_integer(quadrature_order, "quadrature_order", 2)
        if order != 1:
            raise BackendCapabilityError("v1 supports first-order triangular elements only.")
        if max_element_size is not None and resolution is not None:
            raise ConfigurationError("Supply only one of max_element_size or resolution.")
        if resolution is not None:
            if len(resolution) != 2:
                raise ConfigurationError("resolution must be an (Nx, Nz) pair.")
            nx = _positive_integer(resolution[0], "resolution[0]", 2)
            nz = _positive_integer(resolution[1], "resolution[1]", 2)
            maximum = min(self.x_range / (nx - 1), self.period / (nz - 1))
        elif max_element_size is not None:
            maximum = _positive_real(max_element_size, "max_element_size")
        else:
            wavelength = C_0 / self.frequency / max(self._background_index(), 1.0)
            maximum = min(wavelength / wavelength_count, min(self.x_range, self.period) / 12.0)

        self._invalidate_discretization()
        mesh_data = discretize_periodic_2d(
            self.geometry,
            max_element_size=maximum,
            element_order=order,
        )
        polarizations = ("TE", "TM") if self.polarization == "both" else (self.polarization.upper(),)
        systems = {
            selected: assemble_periodic_system_2d(
                mesh_data,
                polarization=selected,  # type: ignore[arg-type]
                frequency=self.frequency,
                k0=self.k0,
                material_at=self.geometry.transformed_material_at,
                quadrature_order=quadrature,
            )
            for selected in polarizations
        }
        self._mesh_data = mesh_data
        self._systems = systems
        self._discretized_revision = self.geometry.revision
        self._discretization_settings = {
            "max_element_size": maximum,
            "element_order": order,
            "quadrature_order": quadrature,
        }
        return mesh_data

    def refine(self, factor: float = 2.0) -> FEMPeriodicMesh2D:
        if self._discretization_settings is None:
            raise NotDiscretizedError("Call discretize() before refine().")
        refinement = _positive_real(factor, "factor")
        if refinement <= 1.0:
            raise ConfigurationError("factor must be greater than one.")
        return self.discretize(
            max_element_size=float(self._discretization_settings["max_element_size"]) / refinement,
            element_order=int(self._discretization_settings["element_order"]),
            quadrature_order=int(self._discretization_settings["quadrature_order"]),
        )

    def _require_current_systems(self) -> dict[str, PeriodicFEMSystem2D]:
        if self._mesh_data is None or not self._systems:
            raise NotDiscretizedError("Call discretize() before solve().")
        if self._discretized_revision != self.geometry.revision:
            raise StaleDiscretizationError("Geometry changed; call discretize() again.")
        return self._systems

    # ------------------------------------------------------------------
    # Fields, normalization, and filtering
    # ------------------------------------------------------------------
    def _default_guess(self) -> complex:
        values: list[complex] = []
        for material in (self.geometry.background, *(region.material for region in self.geometry.regions)):
            epsilon = np.asarray(material.eps_r)
            mu = np.asarray(material.mu_r)
            values.extend(np.sqrt(epsilon * mu))
        selected = values[int(np.argmax(np.abs(values)))]
        if selected.real < 0.0 or (abs(selected.real) <= 1e-14 and selected.imag > 0.0):
            selected = -selected
        return complex(selected)

    def _field_data(
        self,
        system: PeriodicFEMSystem2D,
        full: ComplexArray,
        neff: complex,
    ) -> dict[str, Any]:
        field = system.basis.interpolate(full)
        primary = np.asarray(field, dtype=np.complex128)
        gradient = np.asarray(field.grad, dtype=np.complex128)
        coordinates = np.asarray(system.basis.global_coordinates() / self.k0, dtype=np.float64)
        transformed_epsilon, transformed_mu = evaluate_material(
            self.geometry.transformed_material_at, coordinates[0], coordinates[1]
        )
        physical_epsilon, physical_mu = evaluate_material(
            self.geometry.material_at, coordinates[0], coordinates[1]
        )
        zeros = np.zeros_like(primary)
        if system.polarization == "TE":
            ex, ey, ez = zeros, primary, zeros
            hx = -1j / ETA_0 / transformed_mu[0] * (gradient[1] - 1j * neff * primary)
            hy = zeros
            hz = 1j / ETA_0 / transformed_mu[2] * gradient[0]
        else:
            hx, hy, hz = zeros, primary, zeros
            ex = 1j * ETA_0 / transformed_epsilon[0] * (gradient[1] - 1j * neff * primary)
            ey = zeros
            ez = -1j * ETA_0 / transformed_epsilon[2] * gradient[0]
        fields = np.stack((ex, ey, ez, hx, hy, hz), axis=-1)
        weights = np.asarray(system.basis.dx / self.k0**2, dtype=float)
        pml_mask = self.geometry.pml_mask(coordinates[0])
        physical_weights = weights * ~pml_mask
        poynting = 0.5 * np.sum(physical_weights * (ex * np.conj(hy) - ey * np.conj(hx)))
        density = (
            np.abs(physical_epsilon[0]) * np.abs(ex) ** 2
            + np.abs(physical_epsilon[1]) * np.abs(ey) ** 2
            + np.abs(physical_epsilon[2]) * np.abs(ez) ** 2
            + ETA_0**2
            * (
                np.abs(physical_mu[0]) * np.abs(hx) ** 2
                + np.abs(physical_mu[1]) * np.abs(hy) ** 2
                + np.abs(physical_mu[2]) * np.abs(hz) ** 2
            )
        )
        total_energy = float(np.sum(weights * density).real)
        physical_energy = float(np.sum(physical_weights * density).real)
        pml_fraction = 0.0 if total_energy <= 0.0 else max(0.0, min(1.0, 1.0 - physical_energy / total_energy))
        return {
            "coordinates": coordinates,
            "fields": fields,
            "epsilon": physical_epsilon,
            "power": complex(poynting),
            "energy": max(physical_energy, 0.0),
            "pml_fraction": pml_fraction,
        }

    def _barycentric_fields(
        self,
        system: PeriodicFEMSystem2D,
        full: ComplexArray,
        neff: complex,
    ) -> tuple[FloatArray, ComplexArray, ComplexArray, ComplexArray, FloatArray]:
        """Reconstruct one complex E/H sample at every triangle barycentre."""

        cells = np.asarray(system.mesh_data.mesh.t.T, dtype=np.int64)
        computational_points = np.asarray(system.basis.mesh.p.T, dtype=np.float64)
        cell_points = computational_points[cells]
        differences = cell_points[:, 1:, :] - cell_points[:, :1, :]
        nodal_values = np.asarray(full[cells], dtype=np.complex128)
        gradient = np.linalg.solve(
            differences,
            (nodal_values[:, 1:] - nodal_values[:, :1])[..., np.newaxis],
        )[..., 0]
        primary = np.mean(nodal_values, axis=1)
        coordinates = np.asarray(system.mesh_data.nodes[cells].mean(axis=1), dtype=np.float64)
        transformed_epsilon, transformed_mu = evaluate_material(
            self.geometry.transformed_material_at,
            coordinates[:, 0],
            coordinates[:, 1],
        )
        physical_epsilon, physical_mu = evaluate_material(
            self.geometry.material_at,
            coordinates[:, 0],
            coordinates[:, 1],
        )
        zeros = np.zeros_like(primary)
        if system.polarization == "TE":
            ex, ey, ez = zeros, primary, zeros
            hx = -1j / ETA_0 / transformed_mu[0] * (
                gradient[:, 1] - 1j * neff * primary
            )
            hy = zeros
            hz = 1j / ETA_0 / transformed_mu[2] * gradient[:, 0]
        else:
            hx, hy, hz = zeros, primary, zeros
            ex = 1j * ETA_0 / transformed_epsilon[0] * (
                gradient[:, 1] - 1j * neff * primary
            )
            ey = zeros
            ez = -1j * ETA_0 / transformed_epsilon[2] * gradient[:, 0]
        fields = np.column_stack((ex, ey, ez, hx, hy, hz))
        pml_fraction = self.geometry.pml_mask(coordinates[:, 0]).astype(np.float64)
        return (
            coordinates,
            np.asarray(fields, dtype=np.complex128),
            np.moveaxis(physical_epsilon, 0, -1),
            np.moveaxis(physical_mu, 0, -1),
            pml_fraction,
        )

    def _make_mode(
        self,
        system: PeriodicFEMSystem2D,
        reduced: ComplexArray,
        neff: complex,
        residual: float,
        method: str,
        index: int,
        propagation_ratio_tolerance: float,
    ) -> tuple[PeriodicMode, ComplexArray]:
        full = system.expand(reduced)
        data = self._field_data(system, full, neff)
        power = complex(data["power"])
        propagating = abs(power) > np.finfo(float).tiny and abs(power.real) / abs(power) >= propagation_ratio_tolerance
        if propagating:
            scale = 1.0 / np.sqrt(abs(power.real))
            normalization = "unit-longitudinal-power"
            direction = "forward" if power.real > 0.0 else "backward"
        else:
            if data["energy"] <= np.finfo(float).tiny:
                raise SolverError("An eigenvector has zero physical power and energy.")
            scale = 1.0 / np.sqrt(float(data["energy"]))
            normalization = "energy-like"
            tolerance = 1e-10 * max(1.0, abs(neff))
            direction = "right-decaying" if neff.imag < -tolerance else (
                "left-decaying" if neff.imag > tolerance else "indeterminate"
            )
        full = np.asarray(full * scale, dtype=np.complex128)
        pivot = full[int(np.argmax(np.abs(full)))]
        if abs(pivot) > 0.0:
            full *= np.exp(-1j * np.angle(pivot))
        data = self._field_data(system, full, neff)
        (
            coordinates,
            values,
            cell_epsilon,
            cell_mu,
            cell_pml_fraction,
        ) = self._barycentric_fields(system, full, neff)
        if values.shape != (self.mesh_data.info.elements, 6):
            raise SolverError("The field sampler did not preserve triangle ownership.")
        sample_owners = np.arange(self.mesh_data.info.elements, dtype=np.int64)
        sampled = PeriodicSampledFields(
            coordinates,
            values,
            dimension=2,
            mesh_points=self.mesh_data.nodes,
            mesh_cells=self.mesh_data.elements,
            sample_element_indices=sample_owners,
            material=cell_epsilon,
            metadata={
                "component_order": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                "sampling": "element-barycentre",
                "coordinate_order": ("x", "z"),
                "time_convention": "exp(+1j*omega*t - 1j*k0*neff*z)",
                "cell_epsilon_r": cell_epsilon,
                "cell_mu_r": cell_mu,
                "cell_pml_fraction": cell_pml_fraction,
                "mesh_element_tags": self.mesh_data.element_tags,
                "periodic_node_pairs": np.column_stack(
                    (self.mesh_data.slave_nodes, self.mesh_data.master_nodes)
                ),
            },
        )
        mode = PeriodicMode(
            neff=neff,
            k0=self.k0,
            period=self.period,
            fields=sampled,
            coefficients=full,
            index=index,
            polarization=system.polarization,
            power=complex(data["power"]),
            direction=direction,
            normalization=normalization,
            residual=residual,
            pml_fraction=float(data["pml_fraction"]),
            metadata={"eigensolver": method},
        )
        return mode, full

    @staticmethod
    def _direction_matches(direction: str, requested: Direction) -> bool:
        if requested == "all":
            return True
        if requested == "forward":
            return direction in ("forward", "right-decaying", "indeterminate")
        return direction in ("backward", "left-decaying")

    def solve(
        self,
        neff_guess: complex | None = None,
        num_modes: int | None = None,
        *,
        direction: Direction = "forward",
        eigensolver: str | None = None,
        arnoldi_backend: str | None = None,
        eigensolver_tolerance: float = 1e-10,
        residual_tolerance: float = 1e-8,
        propagation_ratio_tolerance: float = 1e-3,
        max_pml_fraction: float | None = 0.5,
        dense_linearization_limit: int = 700,
    ) -> PeriodicModeSet:
        """Solve for validated modes nearest ``neff_guess``.

        ``max_pml_fraction`` rejects modes whose energy-like density is too
        concentrated in the transverse PML.  Pass ``None`` to disable this
        filter explicitly.
        """

        systems = self._require_current_systems()
        requested = self.num_modes if num_modes is None else _positive_integer(num_modes, "num_modes")
        if direction not in ("forward", "backward", "all"):
            raise ConfigurationError("direction must be 'forward', 'backward', or 'all'.")
        for value, name in (
            (eigensolver_tolerance, "eigensolver_tolerance"),
            (residual_tolerance, "residual_tolerance"),
            (propagation_ratio_tolerance, "propagation_ratio_tolerance"),
        ):
            _positive_real(value, name)
        if max_pml_fraction is None:
            pml_limit = None
        else:
            if isinstance(max_pml_fraction, (bool, np.bool_)):
                raise ConfigurationError("max_pml_fraction must lie in [0, 1] or be None.")
            pml_limit = float(max_pml_fraction)
            if not np.isfinite(pml_limit) or not 0.0 <= pml_limit <= 1.0:
                raise ConfigurationError("max_pml_fraction must lie in [0, 1] or be None.")
        target = self._default_guess() if neff_guess is None and self.neff_guess is None else _finite_complex(
            self.neff_guess if neff_guess is None else neff_guess, "neff_guess"
        )
        method = self.eigensolver if eigensolver is None else str(eigensolver).strip().lower()
        backend = self.arnoldi_backend if arnoldi_backend is None else str(arnoldi_backend).strip().lower()
        candidate_count = max(4 * requested, requested + 12)
        candidates: list[tuple[float, complex, ComplexArray, float, str, PeriodicFEMSystem2D]] = []
        for system in systems.values():
            values, vectors, _, used_method = solve_qep_candidates(
                system,
                target=target,
                candidate_count=min(candidate_count, max(1, 2 * system.ndofs - 2)),
                tolerance=float(eigensolver_tolerance),
                eigensolver=method,
                arnoldi_backend=backend,
                dense_linearization_limit=_positive_integer(dense_linearization_limit, "dense_linearization_limit", 4),
            )
            for value, vector in zip(values, vectors.T, strict=True):
                candidates.append((abs(value - target), complex(value), np.asarray(vector), np.nan, used_method, system))
        candidates.sort(key=lambda item: item[0])

        accepted: list[PeriodicMode] = []
        coefficients: list[ComplexArray] = []
        accepted_reduced: list[tuple[str, complex, ComplexArray]] = []
        rejected_residual = rejected_duplicate = rejected_energy = 0
        rejected_pml = rejected_direction = 0
        for _, neff, raw_vector, _, used_method, system in candidates:
            vector_norm = float(np.linalg.norm(raw_vector))
            if not np.isfinite(neff) or vector_norm <= np.finfo(float).tiny:
                continue
            vector = np.asarray(raw_vector / vector_norm, dtype=np.complex128)
            residual = system.relative_residual(vector, neff)
            if not np.isfinite(residual) or residual > residual_tolerance:
                rejected_residual += 1
                continue
            duplicate = False
            for previous_polarization, previous_value, previous_vector in accepted_reduced:
                if previous_polarization != system.polarization or abs(previous_value - neff) > 1e-8 * max(1.0, abs(neff)):
                    continue
                overlap = abs(np.vdot(previous_vector, vector)) / (
                    np.linalg.norm(previous_vector) * np.linalg.norm(vector)
                )
                if overlap > 1.0 - 1e-8:
                    duplicate = True
                    break
            if duplicate:
                rejected_duplicate += 1
                continue
            try:
                mode, full = self._make_mode(
                    system,
                    vector,
                    neff,
                    residual,
                    used_method,
                    len(accepted) + 1,
                    float(propagation_ratio_tolerance),
                )
            except SolverError:
                rejected_energy += 1
                continue
            if pml_limit is not None and mode.pml_fraction > pml_limit:
                rejected_pml += 1
                continue
            if not self._direction_matches(mode.direction, direction):
                rejected_direction += 1
                continue
            accepted.append(mode)
            coefficients.append(full)
            accepted_reduced.append((system.polarization, neff, vector))
            if len(accepted) == requested:
                break
        if len(accepted) < requested:
            raise SolverError(
                f"Only {len(accepted)} validated mode(s) were found; requested {requested}. "
                f"Rejected: QEP={rejected_residual}, duplicate={rejected_duplicate}, "
                f"energy={rejected_energy}, PML={rejected_pml}, "
                f"direction={rejected_direction}. Try direction='all', a closer "
                "neff_guess, a larger max_pml_fraction, or a finer mesh."
            )

        result = PeriodicModeSet(
            accepted,
            frequency=self.frequency,
            period=self.period,
            dimension=2,
            metadata={
                "eigensolver": method,
                "arnoldi_backend": backend,
                "geometry_revision": self.geometry.revision,
                "mesh_element_tags": self.mesh_data.element_tags,
                "periodic_node_pairs": np.column_stack(
                    (self.mesh_data.slave_nodes, self.mesh_data.master_nodes)
                ),
                "physical_names": self.mesh_data.physical_names,
                "boundary_facets": {
                    name: np.asarray(self.mesh_data.mesh.facets[:, facets].T, dtype=np.int64)
                    for name, facets in self.mesh_data.boundary_facets.items()
                    if len(facets)
                },
            },
        )
        self._result = result
        self.modes = result
        self.neff = result.neff
        self.beta = result.beta
        self.gamma = result.gamma
        self.gammas = result.gamma
        self.eigenvalues = result.neff
        self.coefficients = np.stack(coefficients)
        self.eigenvectors = self.coefficients.T
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            setattr(self, component, np.stack([mode.component(component) for mode in result]))
        return result

    def visualize(self, mode: int | PeriodicMode = 1, **kwargs: Any) -> Any:
        from .visualization import visualize

        selected = self.result.mode(mode) if isinstance(mode, int) else mode
        return visualize(selected, **kwargs)

    def visualize_with_gui(self) -> Any:
        """Open every solved mode in the standalone native viewer."""

        from .visualization import visualize_with_gui

        return visualize_with_gui(self.result)


__all__ = ["PeriodicModeSolver2D"]
