"""Scattering results, sampled fields, S-parameters, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from os import PathLike
from operator import index as integer_index
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .modes import Mode
from .scene import Scene2D


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
PortKey = tuple[str, int, int]
BetaKey = tuple[str, int]
_SIDES = frozenset(("left", "right"))


def _finite_complex_scalar(value: object, name: str) -> complex:
    """Return one finite complex scalar without accepting strings or arrays."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite complex scalar.") from exc
    if array.shape != () or isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite complex scalar.")
    try:
        scalar = complex(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite complex scalar.") from exc
    if not np.isfinite((scalar.real, scalar.imag)).all():
        raise ValueError(f"{name} must be a finite complex scalar.")
    return scalar


def _finite_real_scalar(value: object, name: str) -> float:
    """Return one finite real scalar, rejecting silent complex truncation."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc
    if (
        array.shape != ()
        or np.iscomplexobj(array)
        or isinstance(value, (bool, str, bytes))
    ):
        raise ValueError(f"{name} must be a finite real scalar.")
    try:
        scalar = complex(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc
    if scalar.imag != 0.0 or not np.isfinite((scalar.real, scalar.imag)).all():
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(scalar.real)


def _side(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be 'left' or 'right'.")
    normalized = value.lower()
    if normalized not in _SIDES:
        raise ValueError(f"{name} must be 'left' or 'right'; received {value!r}.")
    return normalized


def _mode_index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer.")
    try:
        normalized = integer_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return normalized


def _mapping_items(value: object, name: str) -> Any:
    try:
        return value.items()  # type: ignore[union-attr]
    except AttributeError as exc:
        raise ValueError(f"{name} must be a mapping.") from exc


def _complex_field(value: ArrayLike, npoints: int, name: str) -> ComplexArray:
    try:
        array = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain complex-valued samples.") from exc
    if array.shape != (3, npoints):
        raise ValueError(f"{name} must have shape (3, {npoints}); received {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value.")
    return array


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One structured numerical diagnostic."""

    severity: Literal["info", "warning", "error"]
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Collection returned by :meth:`ScatteringResult.check`."""

    diagnostics: tuple[Diagnostic, ...]

    @property
    def ok(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "warning")


@dataclass(frozen=True, slots=True)
class ScatteringResult:
    """Self-contained sampled FEM fields and modal scattering observables."""

    coordinates: RealArray
    E_incident: ComplexArray
    E_scattered: ComplexArray
    H_incident: ComplexArray
    H_scattered: ComplexArray
    s_parameters: Mapping[PortKey, complex]
    reflected_power: float
    transmitted_power: float
    radiated_power: float
    absorbed_power: float
    incident_power: float
    ndofs: int
    solve_info: Mapping[str, Any] = field(default_factory=dict)
    mesh_info: Mapping[str, Any] = field(default_factory=dict)
    projection_condition_numbers: Mapping[str, float] = field(default_factory=dict)
    reference_planes: Mapping[str, float] = field(default_factory=dict)
    port_betas: Mapping[BetaKey, complex] = field(default_factory=dict)
    frequency_hz: float | None = None
    ky: float | None = None
    modes: tuple[Mode, ...] = field(default_factory=tuple)
    h5_path: Path | None = None
    scene: Scene2D | None = None

    def __post_init__(self) -> None:
        if self.scene is not None and not isinstance(self.scene, Scene2D):
            raise ValueError("scene must be a Scene2D instance or None.")
        raw_coordinates = np.asarray(self.coordinates)
        if np.iscomplexobj(raw_coordinates) and np.any(np.imag(raw_coordinates) != 0.0):
            raise ValueError("coordinates must contain real values.")
        try:
            coordinates = np.asarray(np.real(raw_coordinates), dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coordinates must contain real values.") from exc
        if coordinates.ndim != 2 or coordinates.shape[0] != 2:
            raise ValueError("coordinates must have shape (2, npoints).")
        if coordinates.shape[1] == 0:
            raise ValueError("coordinates must contain at least one sample point.")
        if not np.isfinite(coordinates).all():
            raise ValueError("coordinates contain a non-finite value.")
        npoints = coordinates.shape[1]
        object.__setattr__(self, "coordinates", coordinates)
        for name in ("E_incident", "E_scattered", "H_incident", "H_scattered"):
            object.__setattr__(self, name, _complex_field(getattr(self, name), npoints, name))
        power_names = (
            "reflected_power",
            "transmitted_power",
            "radiated_power",
            "absorbed_power",
            "incident_power",
        )
        powers = {
            name: _finite_real_scalar(getattr(self, name), name) for name in power_names
        }
        if any(value < 0.0 for value in powers.values()):
            raise ValueError("All reported powers must be finite and nonnegative.")
        if powers["incident_power"] == 0.0:
            raise ValueError("incident_power must be positive.")
        for name, value in powers.items():
            object.__setattr__(self, name, value)

        ndofs = _mode_index(self.ndofs, "ndofs")
        if ndofs == 0:
            raise ValueError("ndofs must be a positive integer.")
        object.__setattr__(self, "ndofs", ndofs)

        s_parameters: dict[PortKey, complex] = {}
        for raw_key, raw_value in _mapping_items(self.s_parameters, "s_parameters"):
            if not isinstance(raw_key, tuple) or len(raw_key) != 3:
                raise ValueError(
                    "Each s_parameters key must be a (side, out_mode, in_mode) tuple."
                )
            key = (
                _side(raw_key[0], "S-parameter side"),
                _mode_index(raw_key[1], "S-parameter out_mode"),
                _mode_index(raw_key[2], "S-parameter in_mode"),
            )
            if key in s_parameters:
                raise ValueError(f"Duplicate normalized S-parameter key {key!r}.")
            s_parameters[key] = _finite_complex_scalar(
                raw_value, f"S-parameter {key!r}"
            )
        object.__setattr__(self, "s_parameters", MappingProxyType(s_parameters))

        try:
            solve_info = dict(self.solve_info)
            mesh_info = dict(self.mesh_info)
        except (TypeError, ValueError) as exc:
            raise ValueError("solve_info and mesh_info must be mappings.") from exc
        object.__setattr__(self, "solve_info", MappingProxyType(solve_info))
        object.__setattr__(self, "mesh_info", MappingProxyType(mesh_info))

        conditions: dict[str, float] = {}
        for monitor, raw_condition in _mapping_items(
            self.projection_condition_numbers, "projection_condition_numbers"
        ):
            if not isinstance(monitor, str) or not monitor.strip():
                raise ValueError(
                    "Projection-condition keys must be nonempty monitor names."
                )
            condition = _finite_real_scalar(
                raw_condition, f"projection condition for {monitor!r}"
            )
            if condition < 1.0:
                raise ValueError(
                    f"projection condition for {monitor!r} must be at least one."
                )
            conditions[monitor] = condition
        object.__setattr__(
            self,
            "projection_condition_numbers",
            MappingProxyType(conditions),
        )

        reference_planes: dict[str, float] = {}
        for raw_side, raw_plane in _mapping_items(
            self.reference_planes, "reference_planes"
        ):
            side = _side(raw_side, "reference-plane side")
            if side in reference_planes:
                raise ValueError(f"Duplicate normalized reference-plane side {side!r}.")
            reference_planes[side] = _finite_real_scalar(
                raw_plane, f"reference plane for {side!r}"
            )
        object.__setattr__(
            self, "reference_planes", MappingProxyType(reference_planes)
        )

        betas: dict[BetaKey, complex] = {}
        for raw_key, raw_beta in _mapping_items(self.port_betas, "port_betas"):
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError("Each port_betas key must be a (side, mode) tuple.")
            key = (
                _side(raw_key[0], "port-beta side"),
                _mode_index(raw_key[1], "port-beta mode"),
            )
            if key in betas:
                raise ValueError(f"Duplicate normalized port-beta key {key!r}.")
            beta = _finite_complex_scalar(raw_beta, f"port beta for {key!r}")
            direction_tolerance = 1e-10 * max(1.0, abs(beta))
            if beta.real < -direction_tolerance or (
                abs(beta.real) <= direction_tolerance
                and beta.imag < -direction_tolerance
            ):
                raise ValueError(
                    "port_betas must store the +z propagation root for each "
                    "modal family (positive-real propagating or positive-imaginary "
                    "right-decaying), including reflected left-port outputs."
                )
            betas[key] = beta
        object.__setattr__(self, "port_betas", MappingProxyType(betas))

        if self.frequency_hz is not None:
            frequency_hz = _finite_real_scalar(self.frequency_hz, "frequency_hz")
            if frequency_hz <= 0.0:
                raise ValueError("frequency_hz must be positive when provided.")
            object.__setattr__(self, "frequency_hz", frequency_hz)
        if self.ky is not None:
            object.__setattr__(self, "ky", _finite_real_scalar(self.ky, "ky"))

        try:
            modes = tuple(self.modes)
        except TypeError as exc:
            raise ValueError("modes must be an iterable of Mode objects.") from exc
        if any(not isinstance(mode, Mode) for mode in modes):
            raise ValueError("modes must contain only wavefem.Mode objects.")
        object.__setattr__(self, "modes", modes)

        if self.h5_path is not None:
            if not isinstance(self.h5_path, (str, PathLike)):
                raise ValueError("h5_path must be path-like or None.")
            object.__setattr__(self, "h5_path", Path(self.h5_path))

    @property
    def E_total(self) -> ComplexArray:
        return self.E_incident + self.E_scattered

    @property
    def H_total(self) -> ComplexArray:
        return self.H_incident + self.H_scattered

    def S(self, side: str, *, out_mode: int = 0, in_mode: int = 0) -> complex:
        """Return an indexed outgoing modal amplitude."""

        key = (
            _side(side, "side"),
            _mode_index(out_mode, "out_mode"),
            _mode_index(in_mode, "in_mode"),
        )
        if key not in self.s_parameters:
            raise KeyError(f"No S-parameter is available for {key!r}.")
        return complex(self.s_parameters[key])

    @property
    def S11(self) -> complex:
        return self.S("left", out_mode=0, in_mode=0)

    @property
    def S21(self) -> complex:
        return self.S("right", out_mode=0, in_mode=0)

    @property
    def reflection(self) -> float:
        return self.reflected_power / self.incident_power

    @property
    def transmission(self) -> float:
        return self.transmitted_power / self.incident_power

    @property
    def power_balance(self) -> float:
        return (
            self.reflected_power
            + self.transmitted_power
            + self.radiated_power
            + self.absorbed_power
        ) / self.incident_power

    @property
    def power_balance_error(self) -> float:
        return abs(1.0 - self.power_balance)

    def field(
        self,
        component: str = "E",
        *,
        quantity: Literal["complex", "abs", "real", "imag", "phase", "norm"] = "complex",
        part: Literal["total", "incident", "scattered"] = "total",
    ) -> ComplexArray | RealArray:
        """Return a sampled electric or magnetic component/derived quantity."""

        if part not in {"total", "incident", "scattered"}:
            raise ValueError("part must be 'total', 'incident', or 'scattered'.")
        if not isinstance(component, str) or not component:
            raise ValueError("component must be E, H, or a Cartesian field component.")
        prefix = component[0].upper()
        if prefix not in {"E", "H"}:
            raise ValueError("component must start with E or H.")
        base = (
            getattr(self, f"{prefix}_{part}")
            if part != "total"
            else getattr(self, f"{prefix}_total")
        )
        suffix = component[1:].lower()
        if suffix in {"x", "y", "z"}:
            values = base[{"x": 0, "y": 1, "z": 2}[suffix]]
        elif suffix == "":
            values = np.sqrt(np.sum(np.abs(base) ** 2, axis=0))
            if quantity == "complex":
                quantity = "norm"
        else:
            raise ValueError(f"Unknown field component {component!r}.")
        if quantity == "complex":
            return np.asarray(values, dtype=np.complex128)
        if quantity in {"abs", "norm"}:
            return np.asarray(np.abs(values), dtype=float)
        if quantity == "real":
            return np.asarray(np.real(values), dtype=float)
        if quantity == "imag":
            return np.asarray(np.imag(values), dtype=float)
        if quantity == "phase":
            return np.asarray(np.angle(values), dtype=float)
        raise ValueError(f"Unknown field quantity {quantity!r}.")

    def plot_field(
        self,
        component: str = "E",
        *,
        quantity: Literal["abs", "real", "imag", "phase", "norm"] = "abs",
        part: Literal["total", "incident", "scattered"] = "total",
        ax: Any | None = None,
        cmap: Any | None = None,
        levels: int = 50,
        colorbar: bool = True,
    ) -> Any:
        """Plot a sampled real-valued electric or magnetic field.

        Triangular filled contours are used for two-dimensional point clouds.
        A scatter plot is used for collinear monitor-like samples.  The method
        returns the Matplotlib axes and never calls ``show``.
        """

        if quantity not in {"abs", "real", "imag", "phase", "norm"}:
            raise ValueError(
                "plot_field quantity must be 'abs', 'real', 'imag', 'phase', or 'norm'."
            )
        level_count = _mode_index(levels, "levels")
        if level_count < 2:
            raise ValueError("levels must be an integer of at least two.")
        if not isinstance(colorbar, bool):
            raise ValueError("colorbar must be a boolean.")

        values = np.asarray(
            self.field(component, quantity=quantity, part=part), dtype=float
        )
        if values.shape != (self.coordinates.shape[1],):
            raise ValueError("The requested field must produce one scalar per coordinate.")

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        elif not all(hasattr(ax, attribute) for attribute in ("scatter", "tricontourf")):
            raise ValueError("ax must be a Matplotlib axes object.")

        # Exact duplicate sample locations are averaged only for visualization;
        # the stored FEM samples remain untouched.
        # Stored samples use (x, z), while all 2D visualizations place z on
        # the horizontal axis and x on the vertical axis.
        points = np.asarray(self.coordinates[[1, 0]].T, dtype=float)
        unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        values = np.bincount(inverse, weights=values) / counts

        selected_cmap = cmap
        if selected_cmap is None:
            selected_cmap = (
                "twilight" if quantity == "phase" else
                "RdBu_r" if quantity in {"real", "imag"} else
                "viridis"
            )

        is_two_dimensional = (
            unique_points.shape[0] >= 3
            and np.linalg.matrix_rank(unique_points - unique_points[0]) == 2
        )
        if is_two_dimensional:
            try:
                artist = ax.tricontourf(
                    unique_points[:, 0],
                    unique_points[:, 1],
                    values,
                    levels=level_count,
                    cmap=selected_cmap,
                )
            except (RuntimeError, ValueError):
                artist = ax.scatter(
                    unique_points[:, 0],
                    unique_points[:, 1],
                    c=values,
                    cmap=selected_cmap,
                )
        else:
            artist = ax.scatter(
                unique_points[:, 0],
                unique_points[:, 1],
                c=values,
                cmap=selected_cmap,
            )

        ax.set_xlabel("z (m)")
        ax.set_ylabel("x (m)")
        ax.set_aspect("equal")
        ax.set_title(f"{part} {component}: {quantity}")
        if colorbar:
            color_axis = ax.figure.colorbar(artist, ax=ax)
            color_axis.set_label(quantity)
        return ax

    def visualize(
        self,
        component: str = "E",
        *,
        quantity: Literal["abs", "real", "imag", "phase", "norm"] = "abs",
        part: Literal["total", "incident", "scattered"] = "total",
        ax: Any | None = None,
        cmap: Any | None = None,
        levels: int = 50,
        colorbar: bool = True,
        show: bool = True,
    ) -> Any:
        """Create and show a Matplotlib field figure.

        Pass ``show=False`` when embedding the returned axes.  Use the
        zero-argument :meth:`visualize_with_gui` method for the native viewer.
        """

        if not isinstance(show, bool):
            raise ValueError("show must be a boolean.")
        axes = self.plot_field(
            component,
            quantity=quantity,
            part=part,
            ax=ax,
            cmap=cmap,
            levels=levels,
            colorbar=colorbar,
        )
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return axes

    def visualize_with_gui(self) -> Any:
        """Open the complete result and all stored modes in the native viewer."""

        from .viewer import _persist_and_launch

        return _persist_and_launch(
            self,
            h5_path=None,
            default_filename="wavefem_result.h5",
        )

    def check(
        self,
        *,
        power_balance_tolerance: float = 1e-3,
        projection_condition_warning: float = 1e10,
        projection_residual_warning: float = 1e-3,
        incoming_projection_warning: float = 1e-3,
        port_gram_diagonal_warning: float = 1e-2,
        s_parameter_power_tolerance: float = 1e-6,
    ) -> DiagnosticReport:
        """Return structured solver-quality diagnostics without printing."""

        power_tolerance = _finite_real_scalar(
            power_balance_tolerance, "power_balance_tolerance"
        )
        condition_warning = _finite_real_scalar(
            projection_condition_warning, "projection_condition_warning"
        )
        residual_warning = _finite_real_scalar(
            projection_residual_warning, "projection_residual_warning"
        )
        incoming_warning = _finite_real_scalar(
            incoming_projection_warning, "incoming_projection_warning"
        )
        gram_diagonal_warning = _finite_real_scalar(
            port_gram_diagonal_warning, "port_gram_diagonal_warning"
        )
        s_power_tolerance = _finite_real_scalar(
            s_parameter_power_tolerance, "s_parameter_power_tolerance"
        )
        if power_tolerance < 0.0:
            raise ValueError("power_balance_tolerance must be nonnegative.")
        if condition_warning < 1.0:
            raise ValueError("projection_condition_warning must be at least one.")
        if residual_warning < 0.0:
            raise ValueError("projection_residual_warning must be nonnegative.")
        if incoming_warning < 0.0:
            raise ValueError("incoming_projection_warning must be nonnegative.")
        if gram_diagonal_warning < 0.0:
            raise ValueError("port_gram_diagonal_warning must be nonnegative.")
        if s_power_tolerance < 0.0:
            raise ValueError("s_parameter_power_tolerance must be nonnegative.")

        items: list[Diagnostic] = []
        if self.power_balance_error > power_tolerance:
            items.append(
                Diagnostic(
                    "warning",
                    "poor_power_balance",
                    f"Power-balance error {self.power_balance_error:.3e} exceeds "
                    f"{power_tolerance:.3e}.",
                )
            )
        if "independent_energy_residual" in self.solve_info:
            try:
                independent_residual = _finite_real_scalar(
                    self.solve_info["independent_energy_residual"],
                    "independent_energy_residual",
                )
            except ValueError:
                items.append(
                    Diagnostic(
                        "error",
                        "invalid_independent_energy_residual",
                        "The independent Poynting energy residual is not finite and real.",
                    )
                )
            else:
                if independent_residual < 0.0:
                    items.append(
                        Diagnostic(
                            "error",
                            "invalid_independent_energy_residual",
                            "The independent Poynting energy residual is negative.",
                        )
                    )
                elif independent_residual > power_tolerance:
                    items.append(
                        Diagnostic(
                            "warning",
                            "poor_independent_energy_balance",
                            f"Independent Poynting energy residual "
                            f"{independent_residual:.3e} exceeds "
                            f"{power_tolerance:.3e}.",
                        )
                    )

        for key in (
            "raw_absorbed_power",
            "raw_radiated_power",
            "raw_reflected_modal_power",
            "raw_transmitted_modal_power",
        ):
            if key not in self.solve_info:
                continue
            try:
                raw_power = _finite_real_scalar(self.solve_info[key], key)
            except ValueError:
                items.append(
                    Diagnostic(
                        "error",
                        "invalid_raw_power",
                        f"Raw power diagnostic {key!r} is not finite and real.",
                    )
                )
                continue
            if raw_power < -power_tolerance * self.incident_power:
                items.append(
                    Diagnostic(
                        "warning",
                        "negative_raw_power",
                        f"Raw power diagnostic {key!r} is negative "
                        f"({raw_power:.3e}); a reported nonnegative power may "
                        "have been clamped.",
                    )
                )
        for monitor, condition in self.projection_condition_numbers.items():
            if condition > condition_warning:
                items.append(
                    Diagnostic(
                        "warning",
                        "ill_conditioned_projection",
                        f"Projection at {monitor!r} has condition number {condition:.3e}.",
                    )
                )

        for key, raw_residual in self.solve_info.items():
            if not isinstance(key, str) or not key.endswith("_projection_residual"):
                continue
            monitor = key.removesuffix("_projection_residual") or key
            try:
                residual = _finite_real_scalar(raw_residual, key)
            except ValueError:
                items.append(
                    Diagnostic(
                        "error",
                        "invalid_projection_residual",
                        f"Projection residual {key!r} is not a finite real scalar.",
                    )
                )
                continue
            if residual < 0.0:
                items.append(
                    Diagnostic(
                        "error",
                        "invalid_projection_residual",
                        f"Projection residual {key!r} is negative ({residual:.3e}).",
                    )
                )
            elif residual > residual_warning:
                items.append(
                    Diagnostic(
                        "warning",
                        "poor_projection_residual",
                        f"Projection at {monitor!r} has relative residual {residual:.3e}, "
                        f"exceeding {residual_warning:.3e}.",
                    )
                )

        if "incoming_projection_relative_error" in self.solve_info:
            try:
                incoming_error = _finite_real_scalar(
                    self.solve_info["incoming_projection_relative_error"],
                    "incoming_projection_relative_error",
                )
            except ValueError:
                items.append(
                    Diagnostic(
                        "error",
                        "invalid_incoming_projection",
                        "The projected incoming-amplitude error is not finite and real.",
                    )
                )
            else:
                if incoming_error < 0.0:
                    items.append(
                        Diagnostic(
                            "error",
                            "invalid_incoming_projection",
                            "The projected incoming-amplitude error is negative.",
                        )
                    )
                elif incoming_error > incoming_warning:
                    items.append(
                        Diagnostic(
                            "warning",
                            "incoming_projection_mismatch",
                            f"Projected and prescribed incoming amplitudes differ by "
                            f"{incoming_error:.3e} relative, exceeding "
                            f"{incoming_warning:.3e}.",
                        )
                    )

        for side in ("forward", "backward"):
            key = f"{side}_port_gram_diagonal_error"
            if key not in self.solve_info:
                continue
            try:
                diagonal_error = _finite_real_scalar(self.solve_info[key], key)
            except ValueError:
                items.append(
                    Diagnostic(
                        "error",
                        "invalid_port_gram_diagonal",
                        f"Port-Gram diagonal error {key!r} is not finite and real.",
                    )
                )
                continue
            if diagonal_error < 0.0:
                items.append(
                    Diagnostic(
                        "error",
                        "invalid_port_gram_diagonal",
                        f"Port-Gram diagonal error {key!r} is negative.",
                    )
                )
            elif diagonal_error > gram_diagonal_warning:
                items.append(
                    Diagnostic(
                        "warning",
                        "port_gram_normalization_error",
                        f"{side.capitalize()} port-Gram unit-power diagonal error "
                        f"{diagonal_error:.3e} exceeds {gram_diagonal_warning:.3e}.",
                    )
                )

        input_modes = {key[2] for key in self.s_parameters}
        if len(input_modes) == 1:
            input_mode = next(iter(input_modes))
            beta_in = self.port_betas.get(("left", input_mode))
            if beta_in is not None and self._is_effectively_propagating(beta_in):
                for side, reported_ratio, label in (
                    ("left", self.reflection, "reflected"),
                    ("right", self.transmission, "transmitted"),
                ):
                    amplitudes = [
                        (out_mode, value)
                        for (out_side, out_mode, in_mode), value in self.s_parameters.items()
                        if out_side == side and in_mode == input_mode
                    ]
                    # A sum of |S|^2 is a valid power check only for one
                    # unit-power mode.  Multimode results use the full
                    # non-orthogonal projection Gram in Scattering2D.
                    if len(amplitudes) != 1:
                        continue
                    output_betas = [
                        self.port_betas.get((side, out_mode))
                        for out_mode, _ in amplitudes
                    ]
                    if any(beta is None for beta in output_betas):
                        continue
                    modal_ratio = 0.0
                    overflowed = False
                    for (_, amplitude), beta in zip(
                        amplitudes, output_betas, strict=True
                    ):
                        assert beta is not None
                        if not self._is_effectively_propagating(beta):
                            continue
                        magnitude = abs(amplitude)
                        if magnitude > np.sqrt(np.finfo(float).max):
                            overflowed = True
                            break
                        modal_ratio += magnitude * magnitude
                    if overflowed or not np.isfinite(modal_ratio):
                        items.append(
                            Diagnostic(
                                "error",
                                "s_parameter_power_overflow",
                                f"The {side!r}-port S-parameter power sum overflowed.",
                            )
                        )
                        continue
                    mismatch = abs(modal_ratio - reported_ratio)
                    scale = max(1.0, abs(modal_ratio), abs(reported_ratio))
                    if mismatch > s_power_tolerance * scale:
                        items.append(
                            Diagnostic(
                                "warning",
                                "s_parameter_power_mismatch",
                                f"Unit-power S-parameters imply {label} ratio "
                                f"{modal_ratio:.6e}, but the reported ratio is "
                                f"{reported_ratio:.6e}.",
                            )
                        )
        if not items:
            items.append(Diagnostic("info", "ok", "No numerical warning threshold was exceeded."))
        return DiagnosticReport(tuple(items))

    def save_h5(self, path: str | PathLike[str]) -> Path:
        """Persist fields, S-parameters, powers, metadata, and lead modes.

        The returned path is absolute.  This frozen result is not mutated;
        :meth:`Scattering2D.run` returns a copy whose ``h5_path`` records the
        written file.
        """

        from .hdf5 import save_result_h5

        return Path(save_result_h5(self, path, modes=self.modes))

    @staticmethod
    def _is_effectively_propagating(beta: complex) -> bool:
        """Conservatively identify a lossless +z propagating beta.

        ScatteringResult does not store modal classification or normalization,
        so S-to-power checks are skipped for appreciably complex roots.
        """

        tolerance = 1e-10 * max(1.0, abs(beta.real))
        return beta.real > 0.0 and abs(beta.imag) <= tolerance

    def deembed(self, *, left: float, right: float) -> "ScatteringResult":
        """Move left/right reference planes for a left-incident result.

        This applies the project convention ``exp(+i beta z)``.  Multimode
        beta values are the +z roots from ``port_betas[(side, mode)]``; the
        left-port outgoing wave direction is handled by the phase formula,
        not by storing a negative beta.
        """

        if "left" not in self.reference_planes or "right" not in self.reference_planes:
            raise ValueError(
                "Both original monitor reference planes are required for de-embedding."
            )
        new_left = _finite_real_scalar(left, "left reference plane")
        new_right = _finite_real_scalar(right, "right reference plane")
        dl = self.reference_planes["left"] - new_left
        dr = self.reference_planes["right"] - new_right
        updated: dict[PortKey, complex] = {}
        for (side, out_mode, in_mode), value in self.s_parameters.items():
            beta_in = self.port_betas.get(("left", in_mode))
            beta_out = self.port_betas.get((side, out_mode))
            if beta_in is None or beta_out is None:
                raise ValueError("Missing modal beta required for de-embedding.")
            if side == "left":
                factor = np.exp(1j * (beta_in + beta_out) * dl)
            elif side == "right":
                factor = np.exp(1j * beta_in * dl - 1j * beta_out * dr)
            else:
                raise ValueError(f"Unsupported output side {side!r}.")
            updated[(side, out_mode, in_mode)] = complex(value * factor)
        return replace(
            self,
            s_parameters=updated,
            reference_planes={"left": new_left, "right": new_right},
            # The associated file contains the pre-de-embedded amplitudes.
            # Clear the path until this derived result is explicitly saved.
            h5_path=None,
        )


__all__ = ["Diagnostic", "DiagnosticReport", "ScatteringResult"]
