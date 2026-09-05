"""Geometry-first full-vector periodic FEM solver in three dimensions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .assembly_3d import (
    PeriodicFEMSystem3D,
    assemble_periodic_system_3d,
    evaluate_material_3d,
    linearized_pencil_3d,
)
from .constants import C_0, ETA_0
from .exceptions import (
    ConfigurationError,
    NotDiscretizedError,
    SolverError,
    StaleDiscretizationError,
)
from .geometry import (
    Box,
    Cylinder,
    GeometryModel3D,
    PMLSpec,
    Shape3D,
    Sphere,
)
from .materials import Material, MaterialInput
from .meshing_3d import PeriodicMesh3D, discretize_3d
from .results import PeriodicMode, PeriodicModeSet, PeriodicSampledFields


Direction = Literal["forward", "backward", "all"]


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.") from exc
    if result != value or result < minimum:
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    return result


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ConfigurationError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be finite and positive.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"{name} must be finite and positive.")
    return result


def _finite_complex(value: complex, name: str) -> complex:
    result = complex(value)
    if not np.isfinite((result.real, result.imag)).all():
        raise ConfigurationError(f"{name} must be finite.")
    return result


def _dense_candidates(
    system: PeriodicFEMSystem3D,
    target: complex,
    count: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64], str]:
    left, right = linearized_pencil_3d(system)
    try:
        homogeneous, eigenvectors = linalg.eig(
            left.toarray(),
            right.toarray(),
            right=True,
            check_finite=False,
            homogeneous_eigvals=True,
        )
    except np.linalg.LinAlgError as exc:
        raise SolverError("Dense homogeneous QZ failed to converge.") from exc
    alpha, denominator = homogeneous
    scale = np.maximum(np.abs(alpha), np.abs(denominator))
    finite = np.abs(denominator) > 256.0 * np.finfo(float).eps * np.maximum(
        scale, 1.0
    )
    values = np.asarray(alpha[finite] / denominator[finite], dtype=np.complex128)
    vectors = np.asarray(
        eigenvectors[: system.ndofs, finite], dtype=np.complex128
    )
    order = np.argsort(np.abs(values - target))[: min(values.size, count)]
    values = values[order]
    vectors = vectors[:, order]
    residuals = np.asarray(
        [system.relative_residual(vectors[:, index], values[index]) for index in range(values.size)],
        dtype=np.float64,
    )
    return values, vectors, residuals, "dense-qz"


def _refined_candidates(
    system: PeriodicFEMSystem3D,
    target: complex,
    count: int,
    tolerance: float,
    backend: str,
    ncv: int | None,
    max_restarts: int,
    random_seed: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64], str]:
    try:
        import periodic_eigensolver
    except ImportError as exc:
        raise SolverError(
            "The refined solve requires the periodic-eigensolver distribution."
        ) from exc
    left, right = linearized_pencil_3d(system)
    public_backend = "python" if backend == "numpy" else backend
    if hasattr(periodic_eigensolver, "solve_generalized"):
        result = periodic_eigensolver.solve_generalized(
            left,
            right,
            sigma=target,
            num_modes=count,
            tol=tolerance,
            ncv=ncv,
            max_restarts=max_restarts,
            random_seed=random_seed,
            backend=public_backend,
        )
        if not result.converged:
            raise SolverError(
                "Refined Arnoldi exhausted its restart budget; best original-pencil "
                f"residual is {float(np.max(result.physical_residuals)):.3e}."
            )
        values = np.asarray(result.eigenvalues, dtype=np.complex128)
        full_vectors = np.asarray(result.eigenvectors, dtype=np.complex128)
        residuals = np.asarray(result.physical_residuals, dtype=np.float64)
        backend_name = str(result.backend)
    else:  # pragma: no cover - compatibility with an early core checkout
        values, full_vectors, residuals, _ = periodic_eigensolver.refined_shift_invert_arnoldi(
            left,
            right,
            target,
            count,
            tolerance,
            ncv=ncv,
            max_restarts=max_restarts,
            random_seed=random_seed,
            kernel_backend=backend,
        )
        values = np.asarray(values, dtype=np.complex128)
        full_vectors = np.asarray(full_vectors, dtype=np.complex128)
        residuals = np.asarray(residuals, dtype=np.float64)
        backend_name = backend
    return values, full_vectors[: system.ndofs], residuals, f"refined-{backend_name}"


class PeriodicModeSolver3D:
    """Solve complex fixed-frequency Bloch propagation in a tetrahedral cell."""

    def __init__(
        self,
        frequency: float,
        x_range: float | tuple[float, float],
        y_range: float | tuple[float, float],
        z_range: float | tuple[float, float],
        num_modes: int = 4,
        neff_guess: complex | None = None,
        *,
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
        self.neff_guess = (
            None if neff_guess is None else _finite_complex(neff_guess, "neff_guess")
        )
        self.eigensolver = str(eigensolver).strip().lower()
        if self.eigensolver not in ("auto", "dense", "refined"):
            raise ConfigurationError("eigensolver must be 'auto', 'dense', or 'refined'.")
        self.arnoldi_backend = str(arnoldi_backend).strip().lower()
        if self.arnoldi_backend not in ("auto", "cython", "python", "numpy"):
            raise ConfigurationError("arnoldi_backend must be auto, cython, or python.")
        background = Material(background_epsilon, background_mu)
        self.geometry = GeometryModel3D(
            x_range, y_range, z_range, background
        )
        self.geometry.set_outer_boundary(boundary)
        self.geometry.add_change_listener(self._geometry_changed)
        self.x_span = self.geometry.x_span
        self.y_span = self.geometry.y_span
        self.z_span = self.geometry.z_span
        self.period = self.z_span[1] - self.z_span[0]
        self.x_range = self.x_span[1] - self.x_span[0]
        self.y_range = self.y_span[1] - self.y_span[0]
        self.z_range = self.period
        self.mesh_data: PeriodicMesh3D | None = None
        self.mesh: PeriodicMesh3D | None = None
        self.native_mesh = None
        self.system: PeriodicFEMSystem3D | None = None
        self.result: PeriodicModeSet | None = None
        self.modes: PeriodicModeSet | None = None
        self._discretized_revision: int | None = None
        self._discretize_options: dict[str, object] | None = None
        self._refinement_scale = 1.0
        self._invalidate_solution()

    def _invalidate_solution(self) -> None:
        self.result = None
        self.modes = None
        self.neff = None
        self.beta = None
        self.gamma = None

    def _geometry_changed(self) -> None:
        self.mesh_data = None
        self.mesh = None
        self.native_mesh = None
        self.system = None
        self._invalidate_solution()
        self._discretized_revision = None

    def add_box(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
        *,
        name: str | None = None,
    ) -> object:
        return self.geometry.add_region(
            Box(x_range, y_range, z_range), Material(epsilon, mu), name=name
        )

    def add_cylinder(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        center: tuple[float, float],
        radius: float,
        z_range: tuple[float, float],
        *,
        name: str | None = None,
    ) -> object:
        return self.geometry.add_region(
            Cylinder(center, radius, z_range), Material(epsilon, mu), name=name
        )

    def add_sphere(
        self,
        epsilon: MaterialInput,
        mu: MaterialInput,
        center: tuple[float, float, float],
        radius: float,
        *,
        name: str | None = None,
    ) -> object:
        return self.geometry.add_region(
            Sphere(center, radius), Material(epsilon, mu), name=name
        )

    def add_pec(self, shape: Shape3D, *, name: str | None = None) -> object:
        return self.geometry.add_boundary(shape, "pec", name=name)

    def add_pmc(self, shape: Shape3D, *, name: str | None = None) -> object:
        return self.geometry.add_boundary(shape, "pmc", name=name)

    def add_pml(
        self,
        thickness: float,
        *,
        order: int = 3,
        sigma_max: float = 5.0,
        direction: str = "all",
    ) -> object:
        return self.geometry.add_pml(
            PMLSpec(float(thickness), int(order), float(sigma_max), str(direction))
        )

    def add_mesh_refinement(
        self, shape: Shape3D, max_element_size: float, *, name: str | None = None
    ) -> object:
        return self.geometry.add_mesh_refinement(
            shape, max_element_size, name=name
        )

    def remove(self, handle: object) -> None:
        self.geometry.remove(handle)  # type: ignore[arg-type]

    def set_outer_boundary(self, kind: str) -> None:
        self.geometry.set_outer_boundary(kind)

    def discretize(
        self,
        *,
        max_element_size: float | None = None,
        wavelength_elements: int = 4,
        material_aware: bool = True,
        quadrature_order: int = 3,
    ) -> PeriodicMesh3D:
        quadrature_order = _positive_integer(quadrature_order, "quadrature_order", 2)
        options = {
            "max_element_size": max_element_size,
            "wavelength_elements": wavelength_elements,
            "material_aware": material_aware,
            "quadrature_order": quadrature_order,
        }
        mesh_data = discretize_3d(
            self.geometry,
            max_element_size=max_element_size,
            wavelength_elements=wavelength_elements,
            material_aware=material_aware,
            k0=self.k0,
            _refinement_scale=self._refinement_scale,
        )
        system = assemble_periodic_system_3d(
            mesh_data,
            frequency=self.frequency,
            k0=self.k0,
            material_at=self.geometry.transformed_material_at,
            quadrature_order=quadrature_order,
        )
        self.mesh_data = mesh_data
        self.mesh = mesh_data
        self.native_mesh = mesh_data.mesh
        self.system = system
        self._invalidate_solution()
        self._discretized_revision = self.geometry.revision
        self._discretize_options = options
        return mesh_data

    def refine(self, factor: float = 2.0) -> PeriodicMesh3D:
        selected = _positive_real(factor, "factor")
        if selected <= 1.0:
            raise ConfigurationError("factor must be greater than one.")
        if self._discretize_options is None:
            raise NotDiscretizedError("discretize() must be called before refine().")
        previous_scale = self._refinement_scale
        self._refinement_scale /= selected
        try:
            return self.discretize(**self._discretize_options)
        except Exception:
            self._refinement_scale = previous_scale
            raise

    def _require_system(self) -> PeriodicFEMSystem3D:
        if self.system is None or self.mesh_data is None:
            raise NotDiscretizedError("Call discretize() before solve().")
        if self._discretized_revision != self.geometry.revision:
            raise StaleDiscretizationError(
                "Geometry changed after discretization; discretize again."
            )
        return self.system

    def _default_guess(self) -> complex:
        epsilon = np.asarray(self.geometry.background.eps_r)
        mu = np.asarray(self.geometry.background.mu_r)
        return complex(np.sqrt(np.max(np.abs(epsilon)) * np.max(np.abs(mu))))

    @staticmethod
    def _direction_matches(mode_direction: str, requested: Direction) -> bool:
        if requested == "all":
            return True
        if requested == "forward":
            return mode_direction in ("forward", "right-decaying")
        return mode_direction in ("backward", "left-decaying")

    def _field_data(
        self,
        system: PeriodicFEMSystem3D,
        full_vector: NDArray[np.complex128],
        neff: complex,
    ) -> dict[str, object]:
        field = system.basis.interpolate(full_vector)
        electric = np.asarray(field, dtype=np.complex128)
        curl = np.asarray(field.curl, dtype=np.complex128)
        cross = np.stack((-electric[1], electric[0], np.zeros_like(electric[0])))
        shifted_curl = curl - 1j * neff * cross
        coordinates = np.asarray(
            system.basis.global_coordinates() / self.k0, dtype=np.float64
        )
        epsilon, mu = evaluate_material_3d(
            system.material_at, coordinates[0], coordinates[1], coordinates[2]
        )
        physical_epsilon, physical_mu = evaluate_material_3d(
            self.geometry.material_at,
            coordinates[0],
            coordinates[1],
            coordinates[2],
        )
        magnetic = 1j / ETA_0 * (1.0 / mu) * shifted_curl
        weights = np.asarray(system.basis.dx / self.k0**3, dtype=np.float64)
        pml = self.geometry.pml_mask(coordinates[0], coordinates[1])
        physical_weights = weights * (~pml)
        poynting = 0.5 * np.sum(
            physical_weights
            * (electric[0] * np.conj(magnetic[1]) - electric[1] * np.conj(magnetic[0]))
        )
        density = np.sum(np.abs(physical_epsilon) * np.abs(electric) ** 2, axis=0) + ETA_0**2 * np.sum(
            np.abs(physical_mu) * np.abs(magnetic) ** 2, axis=0
        )
        total_energy = float(np.sum(weights * density).real)
        physical_energy = float(np.sum(physical_weights * density).real)
        pml_energy = float(np.sum(weights * pml * density).real)
        return {
            "electric": electric,
            "magnetic": magnetic,
            "coordinates": coordinates,
            "epsilon": physical_epsilon,
            "mu": physical_mu,
            "weights": weights,
            "power": complex(poynting),
            "energy_like": max(physical_energy, 0.0),
            "pml_fraction": 0.0 if total_energy <= 0.0 else max(0.0, min(1.0, pml_energy / total_energy)),
        }

    @staticmethod
    def _polarization(electric: NDArray[np.complex128], magnetic: NDArray[np.complex128]) -> str:
        electric_energy = np.sum(np.abs(electric) ** 2, axis=(1, 2))
        magnetic_energy = np.sum(np.abs(magnetic) ** 2, axis=(1, 2))
        e_total = float(np.sum(electric_energy))
        h_total = float(np.sum(magnetic_energy))
        # First-order tetrahedra on an unstructured mesh introduce a small
        # longitudinal leakage even for analytically pure TE/TM modes.
        if e_total and electric_energy[2] / e_total <= 1e-2:
            return "TE"
        if h_total and magnetic_energy[2] / h_total <= 1e-2:
            return "TM"
        return "hybrid"

    def _make_mode(
        self,
        system: PeriodicFEMSystem3D,
        reduced: NDArray[np.complex128],
        neff: complex,
        residual: float,
        gauss: float,
        *,
        method: str,
        index: int,
        propagation_ratio_tolerance: float,
    ) -> PeriodicMode:
        full = system.expand(reduced)
        data = self._field_data(system, full, neff)
        power = complex(data["power"])
        real_power = power.real
        propagating = (
            abs(power) > np.finfo(float).tiny
            and abs(real_power) / abs(power) >= propagation_ratio_tolerance
        )
        if propagating:
            scale = 1.0 / np.sqrt(abs(real_power))
            normalization = "unit-power"
            direction = "forward" if real_power > 0.0 else "backward"
            classification = "propagating"
        else:
            energy = float(data["energy_like"])
            if energy <= np.finfo(float).tiny:
                raise SolverError("A 3D eigenvector has zero physical energy.")
            scale = 1.0 / np.sqrt(energy)
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
        if abs(pivot):
            full *= np.exp(-1j * np.angle(pivot))
        data = self._field_data(system, full, neff)
        electric = np.asarray(data["electric"], dtype=np.complex128)
        magnetic = np.asarray(data["magnetic"], dtype=np.complex128)
        coordinates = np.asarray(data["coordinates"], dtype=np.float64)
        weights = np.asarray(data["weights"], dtype=np.float64)
        weight_sum = np.sum(weights, axis=1)
        sample_points = np.sum(coordinates * weights[np.newaxis, ...], axis=2) / weight_sum
        electric_samples = np.sum(electric * weights[np.newaxis, ...], axis=2) / weight_sum
        magnetic_samples = np.sum(magnetic * weights[np.newaxis, ...], axis=2) / weight_sum
        epsilon = np.asarray(data["epsilon"], dtype=np.complex128)
        epsilon_samples = np.sum(epsilon * weights[np.newaxis, ...], axis=2) / weight_sum
        mu = np.asarray(data["mu"], dtype=np.complex128)
        mu_samples = np.sum(mu * weights[np.newaxis, ...], axis=2) / weight_sum
        cell_pml = self.geometry.pml_mask(sample_points[0], sample_points[1]).astype(np.float64)
        fields = PeriodicSampledFields(
            sample_points.T,
            {
                "Ex": electric_samples[0],
                "Ey": electric_samples[1],
                "Ez": electric_samples[2],
                "Hx": magnetic_samples[0],
                "Hy": magnetic_samples[1],
                "Hz": magnetic_samples[2],
            },
            dimension=3,
            mesh_points=system.mesh_data.nodes,
            mesh_cells=system.mesh_data.elements,
            sample_element_indices=np.arange(system.mesh_data.elements.shape[0]),
            material=epsilon_samples.T,
            metadata={
                "sampling": "element-barycentre",
                "time_convention": "exp(+1j*omega*t - 1j*k0*neff*z)",
                "cell_epsilon_r": epsilon_samples.T,
                "cell_mu_r": mu_samples.T,
                "cell_pml_fraction": cell_pml,
            },
        )
        return PeriodicMode(
            neff=neff,
            k0=self.k0,
            period=self.period,
            fields=fields,
            coefficients=full,
            index=index,
            polarization=self._polarization(electric, magnetic),
            power=complex(data["power"]),
            direction=direction,
            normalization=normalization,
            residual=residual,
            gauss_residual=gauss,
            pml_fraction=float(data["pml_fraction"]),
            metadata={
                "classification": classification,
                "eigensolver": method,
            },
        )

    def solve(self, *args, max_refinements: int = 2,
              adaptive_tolerance: float = 0.05, **options) -> PeriodicModeSet:
        """Solve and remesh until the interface residual meets the threshold
        or the refinement budget is exhausted. Gmsh regenerates periodic node
        and edge constraints on every mesh. Zero refinements means one solve.
        """
        from .adaptive import solve_periodic
        return solve_periodic(self, 3, args, options, max_refinements, adaptive_tolerance)

    def _solve_once(
        self,
        neff_guess: complex | None = None,
        num_modes: int | None = None,
        *,
        direction: Direction = "forward",
        eigensolver: str | None = None,
        arnoldi_backend: str | None = None,
        eigensolver_tolerance: float = 1e-10,
        residual_tolerance: float = 1e-8,
        divergence_tolerance: float = 1e-6,
        propagation_ratio_tolerance: float = 1e-3,
        max_pml_fraction: float | None = 0.5,
        dense_linearization_limit: int = 700,
        ncv: int | None = None,
        max_restarts: int = 12,
        random_seed: int = 0,
    ) -> PeriodicModeSet:
        """Solve for validated modes nearest ``neff_guess``.

        ``divergence_tolerance`` applies to the dimensionless weak Gauss
        *energy* residual returned by :meth:`PeriodicFEMSystem3D.divergence_residual`.
        ``max_pml_fraction`` rejects modes dominated by transverse-PML energy;
        pass ``None`` to disable that filter explicitly.
        """

        system = self._require_system()
        requested = self.num_modes if num_modes is None else _positive_integer(num_modes, "num_modes")
        if direction not in ("forward", "backward", "all"):
            raise ConfigurationError("direction must be 'forward', 'backward', or 'all'.")
        method = self.eigensolver if eigensolver is None else str(eigensolver).strip().lower()
        if method not in ("auto", "dense", "refined"):
            raise ConfigurationError("eigensolver must be 'auto', 'dense', or 'refined'.")
        backend = self.arnoldi_backend if arnoldi_backend is None else str(arnoldi_backend).strip().lower()
        if backend not in ("auto", "cython", "python", "numpy"):
            raise ConfigurationError("arnoldi_backend must be auto, cython, or python.")
        for value, name in (
            (eigensolver_tolerance, "eigensolver_tolerance"),
            (residual_tolerance, "residual_tolerance"),
            (divergence_tolerance, "divergence_tolerance"),
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
        dense_linearization_limit = _positive_integer(
            dense_linearization_limit, "dense_linearization_limit", 4
        )
        max_restarts = _positive_integer(max_restarts, "max_restarts", 0)
        random_seed = _positive_integer(random_seed, "random_seed", 0)
        if ncv is not None:
            ncv = _positive_integer(ncv, "ncv", 2)
        candidate_count = min(
            max(4 * requested, requested + 12), max(1, 2 * system.ndofs - 2)
        )
        resolved_method = method
        if method == "auto":
            resolved_method = "dense" if 2 * system.ndofs <= int(dense_linearization_limit) else "refined"
        if resolved_method == "dense":
            values, vectors, _, method_name = _dense_candidates(system, target, candidate_count)
        else:
            values, vectors, _, method_name = _refined_candidates(
                system,
                target,
                candidate_count,
                float(eigensolver_tolerance),
                backend,
                ncv,
                int(max_restarts),
                int(random_seed),
            )

        accepted: list[PeriodicMode] = []
        accepted_reduced: list[NDArray[np.complex128]] = []
        rejected_residual = rejected_gauss = rejected_pml = rejected_direction = 0
        for value, raw in zip(values, vectors.T, strict=True):
            norm = float(np.linalg.norm(raw))
            if not np.isfinite(value) or norm <= np.finfo(float).tiny:
                continue
            vector = np.asarray(raw / norm, dtype=np.complex128)
            residual = system.relative_residual(vector, complex(value))
            if not np.isfinite(residual) or residual > residual_tolerance:
                rejected_residual += 1
                continue
            gauss = system.divergence_residual(vector, complex(value))
            if not np.isfinite(gauss) or gauss > divergence_tolerance:
                rejected_gauss += 1
                continue
            duplicate = False
            for previous_value, previous in zip(
                (mode.neff for mode in accepted), accepted_reduced, strict=True
            ):
                if abs(previous_value - value) > 1e-8 * max(1.0, abs(value)):
                    continue
                overlap = abs(np.vdot(previous, vector)) / (
                    np.linalg.norm(previous) * np.linalg.norm(vector)
                )
                if overlap > 1.0 - 1e-8:
                    duplicate = True
                    break
            if duplicate:
                continue
            try:
                mode = self._make_mode(
                    system,
                    vector,
                    complex(value),
                    residual,
                    gauss,
                    method=method_name,
                    index=len(accepted) + 1,
                    propagation_ratio_tolerance=propagation_ratio_tolerance,
                )
            except SolverError:
                continue
            if pml_limit is not None and mode.pml_fraction > pml_limit:
                rejected_pml += 1
                continue
            if not self._direction_matches(mode.direction, direction):
                rejected_direction += 1
                continue
            accepted.append(mode)
            accepted_reduced.append(vector)
            if len(accepted) == requested:
                break
        if len(accepted) < requested:
            raise SolverError(
                f"Requested {requested} {direction} mode(s) near neff={target!r}, "
                f"but {len(accepted)} passed validation. Rejected: QEP={rejected_residual}, "
                f"Gauss={rejected_gauss}, PML={rejected_pml}, "
                f"direction={rejected_direction}."
            )
        result = PeriodicModeSet(
            accepted,
            frequency=self.frequency,
            period=self.period,
            dimension=3,
            metadata={
                "backend": method_name,
                "neff_guess": target,
                "requested_direction": direction,
                "mesh": asdict(self.mesh_data.info),
                "mesh_element_tags": self.mesh_data.element_tags,
                "periodic_node_pairs": self.mesh_data.periodic_node_pairs,
                "periodic_edge_pairs": self.mesh_data.periodic_edge_pairs,
                "edge_nodes": self.mesh_data.edge_nodes,
                "cell_edges": self.mesh_data.cell_edges,
                "cell_edge_signs": self.mesh_data.cell_edge_signs,
                "physical_names": dict(self.mesh_data.physical_names),
                "boundary_facets": {
                    name: np.asarray(self.mesh_data.mesh.facets[:, facets].T, dtype=np.int64)
                    for name, facets in self.mesh_data.boundary_facets.items()
                    if len(facets)
                },
                "time_convention": "exp(+1j*omega*t - 1j*k0*neff*z)",
            },
        )
        self.result = result
        self.modes = result
        self.neff = result.neff
        self.beta = result.beta
        self.gamma = result.gamma
        return result

    def visualize(self, mode: int | PeriodicMode = 1, **kwargs: Any) -> Any:
        """Create a Matplotlib figure for one solved three-dimensional mode."""

        from .visualization import visualize

        if self.result is None:
            raise SolverError("No solved modes are available; call solve() first.")
        selected = self.result.mode(mode) if isinstance(mode, int) else mode
        return visualize(selected, **kwargs)

    def visualize_with_gui(self) -> Any:
        """Open every solved mode in the standalone native viewer."""

        from .visualization import visualize_with_gui

        if self.result is None:
            raise SolverError("No solved modes are available; call solve() first.")
        return visualize_with_gui(self.result)


__all__ = ["PeriodicModeSolver3D"]
