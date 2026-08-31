"""Immutable solver results shared by the 2D and 3D periodic packages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_COMPONENTS = {name.lower(): name for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}


def _readonly(value: ArrayLike, dtype: Any, name: str) -> NDArray[Any]:
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values.")
    snapshot = np.ascontiguousarray(array).tobytes()
    return np.frombuffer(snapshot, dtype=np.dtype(dtype)).reshape(array.shape)


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            dtype = np.complex128
        elif np.issubdtype(value.dtype, np.bool_):
            dtype = np.bool_
        elif np.issubdtype(value.dtype, np.integer):
            dtype = np.int64
        elif np.issubdtype(value.dtype, np.floating):
            dtype = np.float64
        else:
            return tuple(_freeze(item) for item in value.tolist())
        return _readonly(value, dtype, "metadata array")
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _finite_complex(value: complex, name: str) -> complex:
    result = complex(value)
    if not np.isfinite((result.real, result.imag)).all():
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PeriodicSampledFields:
    """Complex Cartesian fields sampled at element-owned points."""

    coordinates: FloatArray
    values: Mapping[str, ComplexArray]
    dimension: int
    mesh_points: FloatArray
    mesh_cells: NDArray[np.int64]
    sample_element_indices: NDArray[np.int64]
    material: ComplexArray | None
    metadata: Mapping[str, Any]

    def __init__(
        self,
        coordinates: ArrayLike,
        values: Mapping[str, ArrayLike] | ArrayLike,
        *,
        dimension: int,
        mesh_points: ArrayLike,
        mesh_cells: ArrayLike,
        sample_element_indices: ArrayLike,
        material: ArrayLike | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(dimension, bool) or int(dimension) not in (2, 3):
            raise ValueError("dimension must be 2 or 3.")
        dimension = int(dimension)
        points = _readonly(coordinates, np.float64, "coordinates")
        if points.ndim != 2 or points.shape[1] != dimension or points.shape[0] == 0:
            raise ValueError(f"coordinates must have shape (N, {dimension}).")
        if isinstance(values, Mapping):
            items = values.items()
        else:
            raw = np.asarray(values)
            order = None if metadata is None else metadata.get("component_order")
            if order is None or raw.ndim != 2 or raw.shape[1] != len(order):
                raise ValueError("Array fields require metadata['component_order'] and shape (N, C).")
            items = ((name, raw[:, index]) for index, name in enumerate(order))
        components: dict[str, ComplexArray] = {}
        for raw_name, raw_values in items:
            name = _COMPONENTS.get(str(raw_name).strip().lower())
            if name is None:
                raise ValueError(f"Unknown Maxwell component {raw_name!r}.")
            component = _readonly(raw_values, np.complex128, f"field {name}")
            if component.shape != (points.shape[0],):
                raise ValueError(f"field {name} must have shape ({points.shape[0]},).")
            components[name] = component
        if not components:
            raise ValueError("At least one field component is required.")

        native_points = _readonly(mesh_points, np.float64, "mesh_points")
        native_cells = _readonly(mesh_cells, np.int64, "mesh_cells")
        if native_points.ndim != 2 or native_points.shape[1] != dimension:
            raise ValueError(f"mesh_points must have shape (P, {dimension}).")
        if native_cells.ndim != 2 or native_cells.shape[1] != dimension + 1:
            raise ValueError(f"mesh_cells must have shape (M, {dimension + 1}).")
        if native_cells.size and (native_cells.min() < 0 or native_cells.max() >= native_points.shape[0]):
            raise ValueError("mesh_cells contains a point index outside mesh_points.")
        owners = _readonly(sample_element_indices, np.int64, "sample_element_indices")
        if owners.shape != (points.shape[0],) or (owners.size and (owners.min() < 0 or owners.max() >= native_cells.shape[0])):
            raise ValueError("sample_element_indices must assign every sample to a mesh cell.")
        material_values = None
        if material is not None:
            material_values = _readonly(material, np.complex128, "material")
            if material_values.shape[0] != points.shape[0]:
                raise ValueError("material must have one leading entry per field sample.")

        object.__setattr__(self, "coordinates", points)
        object.__setattr__(self, "values", MappingProxyType(components))
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "mesh_points", native_points)
        object.__setattr__(self, "mesh_cells", native_cells)
        object.__setattr__(self, "sample_element_indices", owners)
        object.__setattr__(self, "material", material_values)
        object.__setattr__(self, "metadata", MappingProxyType({} if metadata is None else {str(k): _freeze(v) for k, v in metadata.items()}))

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.values)

    @property
    def x(self) -> FloatArray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> FloatArray | None:
        return None if self.dimension == 2 else self.coordinates[:, 1]

    @property
    def z(self) -> FloatArray:
        return self.coordinates[:, 1 if self.dimension == 2 else 2]

    def component(self, name: str) -> ComplexArray:
        canonical = _COMPONENTS.get(str(name).strip().lower())
        if canonical is None or canonical not in self.values:
            raise KeyError(f"Field component {name!r} is unavailable.")
        return self.values[canonical]

    def quantity(self, component: str, quantity: str = "real") -> FloatArray:
        family = str(component).strip().upper().replace("|", "")
        if family in ("E", "H"):
            selected = [np.asarray(self.values[name]) for name in (f"{family}x", f"{family}y", f"{family}z") if name in self.values]
            if not selected:
                raise KeyError(f"No {family} components are available.")
            result = np.sqrt(sum(np.abs(values) ** 2 for values in selected))
        else:
            values = self.component(component)
            normalized = str(quantity).strip().lower()
            if normalized in ("real", "re"):
                result = np.real(values)
            elif normalized in ("imag", "imaginary", "im"):
                result = np.imag(values)
            elif normalized in ("magnitude", "mag", "abs"):
                result = np.abs(values)
            elif normalized in ("phase", "angle"):
                result = np.angle(values)
            else:
                raise ValueError("quantity must be real, imag, magnitude, or phase.")
        output = np.asarray(result, dtype=np.float64)
        output.setflags(write=False)
        return output


@dataclass(frozen=True, slots=True, eq=False)
class PeriodicMode:
    neff: complex
    k0: float
    period: float
    fields: PeriodicSampledFields
    coefficients: ComplexArray
    index: int = 1
    polarization: str | None = None
    power: complex | None = None
    direction: str = "indeterminate"
    normalization: str = "unnormalized"
    residual: float | None = None
    gauss_residual: float | None = None
    pml_fraction: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or int(self.index) < 1:
            raise ValueError("index must be positive.")
        if not isinstance(self.fields, PeriodicSampledFields):
            raise TypeError("fields must be PeriodicSampledFields.")
        neff = _finite_complex(self.neff, "neff")
        k0 = float(self.k0)
        period = float(self.period)
        if not np.isfinite(k0) or k0 <= 0.0 or not np.isfinite(period) or period <= 0.0:
            raise ValueError("k0 and period must be finite and positive.")
        coefficients = _readonly(self.coefficients, np.complex128, "coefficients")
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("coefficients must be a nonempty vector.")
        for value, name in ((self.residual, "residual"), (self.gauss_residual, "gauss_residual"), (self.pml_fraction, "pml_fraction")):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.pml_fraction > 1.0 + 1e-12:
            raise ValueError("pml_fraction must not exceed one.")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "neff", neff)
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "metadata", MappingProxyType({str(k): _freeze(v) for k, v in self.metadata.items()}))
        if self.power is not None:
            object.__setattr__(self, "power", _finite_complex(self.power, "power"))

    @property
    def beta(self) -> complex:
        return self.k0 * self.neff

    @property
    def gamma(self) -> complex:
        return 1j * self.beta

    @property
    def bloch_multiplier(self) -> complex:
        return complex(np.exp(-self.gamma * self.period))

    @property
    def folded_beta(self) -> complex:
        reciprocal = 2.0 * np.pi / self.period
        real = (self.beta.real + 0.5 * reciprocal) % reciprocal - 0.5 * reciprocal
        return complex(real, self.beta.imag)

    @property
    def folded_neff(self) -> complex:
        return self.folded_beta / self.k0

    @property
    def attenuation_constant(self) -> float:
        return float(self.gamma.real)

    def component(self, name: str) -> ComplexArray:
        return self.fields.component(name)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PeriodicModeSet(Sequence[PeriodicMode]):
    modes: tuple[PeriodicMode, ...]
    frequency: float
    period: float
    dimension: int
    metadata: Mapping[str, Any]

    def __init__(self, modes: Sequence[PeriodicMode], *, frequency: float, period: float, dimension: int, metadata: Mapping[str, Any] | None = None) -> None:
        entries = tuple(modes)
        if not entries or not all(isinstance(mode, PeriodicMode) for mode in entries):
            raise ValueError("modes must contain at least one PeriodicMode.")
        if int(dimension) not in (2, 3) or any(mode.fields.dimension != int(dimension) for mode in entries):
            raise ValueError("dimension must match every mode field.")
        object.__setattr__(self, "modes", entries)
        object.__setattr__(self, "frequency", float(frequency))
        object.__setattr__(self, "period", float(period))
        object.__setattr__(self, "dimension", int(dimension))
        object.__setattr__(self, "metadata", MappingProxyType({} if metadata is None else {str(k): _freeze(v) for k, v in metadata.items()}))

    @overload
    def __getitem__(self, index: int) -> PeriodicMode: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[PeriodicMode, ...]: ...

    def __getitem__(self, index: int | slice) -> PeriodicMode | tuple[PeriodicMode, ...]:
        return self.modes[index]

    def __len__(self) -> int:
        return len(self.modes)

    def __iter__(self) -> Iterator[PeriodicMode]:
        return iter(self.modes)

    def mode(self, number: int) -> PeriodicMode:
        if number < 1 or number > len(self):
            raise IndexError(f"mode number must be in 1..{len(self)}.")
        return self.modes[number - 1]

    @property
    def neff(self) -> ComplexArray:
        return _readonly([mode.neff for mode in self.modes], np.complex128, "neff")

    @property
    def beta(self) -> ComplexArray:
        return _readonly([mode.beta for mode in self.modes], np.complex128, "beta")

    @property
    def gamma(self) -> ComplexArray:
        return _readonly([mode.gamma for mode in self.modes], np.complex128, "gamma")

    @property
    def folded_beta(self) -> ComplexArray:
        return _readonly([mode.folded_beta for mode in self.modes], np.complex128, "folded_beta")

    @property
    def folded_neff(self) -> ComplexArray:
        return _readonly([mode.folded_neff for mode in self.modes], np.complex128, "folded_neff")

    @property
    def bloch_multiplier(self) -> ComplexArray:
        return _readonly(
            [mode.bloch_multiplier for mode in self.modes], np.complex128, "bloch_multiplier"
        )

    @property
    def directions(self) -> tuple[str, ...]:
        return tuple(mode.direction for mode in self.modes)

    def by_polarization(self, polarization: str) -> tuple[PeriodicMode, ...]:
        selected = str(polarization).strip().upper()
        return tuple(
            mode for mode in self.modes
            if mode.polarization is not None and mode.polarization.upper() == selected
        )

    def save_h5(self, path: str | Path) -> Path:
        """Persist this result through the package's versioned HDF5 writer."""

        try:
            from .persistence import save_periodic_h5
        except ImportError as exc:  # pragma: no cover - integration guard
            raise RuntimeError("The HDF5 persistence module is not installed.") from exc
        return save_periodic_h5(self, path)


# Concise aliases are useful to applications while preserving explicit names.
Mode = PeriodicMode
ModeSet = PeriodicModeSet
SampledFields = PeriodicSampledFields

__all__ = [
    "Mode", "ModeSet", "PeriodicMode", "PeriodicModeSet",
    "PeriodicSampledFields", "SampledFields",
]
