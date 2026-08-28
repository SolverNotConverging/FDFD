"""Immutable modal results shared by the one- and two-dimensional FEM solvers.

The solver implementations deliberately hand visualization a sampled field
representation instead of their native finite-element coefficient vectors.
This keeps result consumers independent of the element family and makes the
same plotting API useful for line meshes, triangular meshes, and user-selected
sampling grids.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
NumericArray = NDArray[np.float64] | NDArray[np.complex128]

_COMPONENT_NAMES = {
    "ex": "Ex",
    "ey": "Ey",
    "ez": "Ez",
    "hx": "Hx",
    "hy": "Hy",
    "hz": "Hz",
}
_QUANTITY_NAMES = {
    "real": "real",
    "re": "real",
    "imag": "imag",
    "imaginary": "imag",
    "im": "imag",
    "magnitude": "magnitude",
    "mag": "magnitude",
    "abs": "magnitude",
    "absolute": "magnitude",
    "phase": "phase",
    "angle": "phase",
}


def _readonly_array(
    value: ArrayLike,
    *,
    dtype: np.dtype[Any] | type[Any],
    name: str,
) -> NDArray[Any]:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    # ``setflags(write=False)`` alone is reversible when an array owns mutable
    # storage.  A bytes-backed snapshot cannot be made writable again, so the
    # result remains immutable even if a caller invokes ``setflags`` directly.
    snapshot = array.tobytes(order="C")
    return np.frombuffer(snapshot, dtype=array.dtype).reshape(array.shape)


def _readonly_numeric_array(value: ArrayLike, *, name: str) -> NumericArray:
    raw = np.asarray(value)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    return _readonly_array(raw, dtype=dtype, name=name)


def _freeze(value: Any) -> Any:
    """Best-effort recursive freezing for public result metadata."""

    if isinstance(value, np.ndarray):
        return _readonly_numeric_array(value, name="metadata array")
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _frozen_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping or None.")
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _canonical_component(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Field component names must be non-empty strings.")
    stripped = name.strip()
    return _COMPONENT_NAMES.get(stripped.lower(), stripped)


def _canonical_quantity(name: str) -> str:
    try:
        return _QUANTITY_NAMES[str(name).strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(set(_QUANTITY_NAMES.values())))
        raise ValueError(f"quantity must be one of {choices}; got {name!r}.") from exc


def _coerce_coordinates(
    coordinates: ArrayLike | Sequence[ArrayLike],
    dimension: int | None,
) -> tuple[tuple[FloatArray, ...], int]:
    if dimension is not None:
        if isinstance(dimension, (bool, np.bool_)) or int(dimension) not in (1, 2):
            raise ValueError("dimension must be 1 or 2.")
        dimension = int(dimension)

    # A tuple of arrays is the unambiguous structured/curvilinear form.  A
    # tuple of scalars remains an ordinary one-dimensional coordinate vector.
    if isinstance(coordinates, tuple) and len(coordinates) in (1, 2) and any(
        np.asarray(axis).ndim > 0 for axis in coordinates
    ):
        axes = tuple(
            _readonly_array(axis, dtype=np.float64, name=f"coordinate axis {index}")
            for index, axis in enumerate(coordinates)
        )
        inferred = len(axes)
    else:
        raw = _readonly_array(coordinates, dtype=np.float64, name="coordinates")
        if raw.ndim == 1:
            axes = (raw,)
            inferred = 1
        elif raw.ndim == 2 and raw.shape[1] in (1, 2):
            inferred = int(raw.shape[1])
            axes = tuple(
                _readonly_array(raw[:, index], dtype=np.float64, name=f"coordinate axis {index}")
                for index in range(inferred)
            )
        else:
            raise ValueError(
                "coordinates must be a 1D array, an (N, 1)/(N, 2) point array, "
                "or a tuple containing one or two coordinate arrays."
            )

    if dimension is not None and inferred != dimension:
        raise ValueError(
            f"coordinates describe a {inferred}D sample but dimension={dimension}."
        )
    for axis in axes:
        if axis.size == 0:
            raise ValueError("coordinate arrays must not be empty.")
    return axes, inferred


def _coerce_values(
    values: Mapping[str, ArrayLike] | ArrayLike,
    metadata: Mapping[str, Any] | None,
) -> dict[str, NumericArray]:
    if isinstance(values, Mapping):
        items = values.items()
    else:
        raw = np.asarray(values)
        component_order = None if metadata is None else metadata.get("component_order")
        if component_order is None:
            raise ValueError(
                "Array-valued fields require metadata['component_order']; alternatively "
                "pass a mapping such as {'Ex': ex, 'Ey': ey}."
            )
        names = tuple(component_order)
        if raw.ndim == 0 or raw.shape[-1] != len(names):
            raise ValueError(
                "The final values axis must match metadata['component_order']; "
                f"received values shape {raw.shape} and {len(names)} names."
            )
        items = ((name, raw[..., index]) for index, name in enumerate(names))

    result: dict[str, NumericArray] = {}
    for raw_name, raw_value in items:
        name = _canonical_component(str(raw_name))
        if name in result:
            raise ValueError(f"Duplicate sampled field component {name!r}.")
        result[name] = _readonly_numeric_array(raw_value, name=f"field component {name}")
    if not result:
        raise ValueError("At least one sampled field component is required.")
    return result


def _sample_layout(
    coordinates: tuple[FloatArray, ...],
    values: Mapping[str, NumericArray],
) -> tuple[str, tuple[int, ...]]:
    shapes = {array.shape for array in values.values()}
    if len(shapes) != 1:
        raise ValueError(
            "All sampled components must share one sample shape; received "
            f"{sorted(shapes)!r}."
        )
    value_shape = next(iter(shapes))
    if len(coordinates) == 1:
        if coordinates[0].ndim != 1 or value_shape != coordinates[0].shape:
            raise ValueError(
                "One-dimensional component arrays must match the x-coordinate shape."
            )
        return "line", value_shape

    x, y = coordinates
    if x.ndim == y.ndim == 1 and value_shape == (y.size, x.size):
        return "structured", value_shape
    if x.shape == y.shape == value_shape:
        return ("points" if x.ndim == 1 else "curvilinear"), value_shape
    raise ValueError(
        "Two-dimensional samples must use either 1D axes with component shape "
        "(len(y), len(x)), or x/y arrays matching every component shape."
    )


def _coerce_mesh(
    points: ArrayLike | None,
    cells: ArrayLike | None,
    dimension: int,
) -> tuple[FloatArray | None, NDArray[np.int64] | None]:
    if points is None and cells is None:
        return None, None
    if points is None or cells is None:
        raise ValueError("mesh_points and mesh_cells must be supplied together.")

    mesh_points = _readonly_array(points, dtype=np.float64, name="mesh_points")
    if dimension == 1:
        if mesh_points.ndim == 1:
            mesh_points = _readonly_array(
                mesh_points[:, np.newaxis], dtype=np.float64, name="mesh_points"
            )
        if mesh_points.ndim != 2 or mesh_points.shape[1] != 1:
            raise ValueError("A 1D mesh must have mesh_points shape (N, 1).")
    elif mesh_points.ndim != 2 or mesh_points.shape[1] != 2:
        raise ValueError("A 2D mesh must have mesh_points shape (N, 2).")

    mesh_cells = _readonly_array(
        cells,
        dtype=np.int64,
        name="mesh_cells",
    )
    expected_vertices = 2 if dimension == 1 else 3
    if mesh_cells.ndim != 2 or mesh_cells.shape[1] < expected_vertices:
        raise ValueError(
            f"mesh_cells must have shape (M, {expected_vertices}) or wider."
        )
    if mesh_cells.size and (
        np.min(mesh_cells) < 0 or np.max(mesh_cells) >= mesh_points.shape[0]
    ):
        raise ValueError("mesh_cells contains a point index outside mesh_points.")
    return mesh_points, mesh_cells


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SampledFields:
    """Immutable samples of one modal field on common 1D or 2D coordinates.

    ``coordinates`` may be an ``(N, 1)``/``(N, 2)`` point array or a tuple of
    coordinate axes.  ``values`` may be a component mapping, or one array whose
    final axis follows ``metadata['component_order']``.  All public arrays are
    defensive copies with writes disabled.
    """

    coordinates: tuple[FloatArray, ...]
    values: Mapping[str, NumericArray]
    dimension: int
    layout: str
    sample_shape: tuple[int, ...]
    mesh_points: FloatArray | None
    mesh_cells: NDArray[np.int64] | None
    material: NumericArray | None
    metadata: Mapping[str, Any]

    def __init__(
        self,
        coordinates: ArrayLike | Sequence[ArrayLike],
        values: Mapping[str, ArrayLike] | ArrayLike,
        *,
        dimension: int | None = None,
        mesh_points: ArrayLike | None = None,
        mesh_cells: ArrayLike | None = None,
        material: ArrayLike | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        axes, inferred_dimension = _coerce_coordinates(coordinates, dimension)
        components = _coerce_values(values, metadata)
        layout, sample_shape = _sample_layout(axes, components)
        points, cells = _coerce_mesh(mesh_points, mesh_cells, inferred_dimension)

        material_array = None
        if material is not None:
            material_array = _readonly_numeric_array(material, name="material")
            if material_array.shape != sample_shape:
                raise ValueError(
                    "material samples must match the field sample shape; received "
                    f"{material_array.shape}, expected {sample_shape}."
                )

        object.__setattr__(self, "coordinates", axes)
        object.__setattr__(self, "values", MappingProxyType(components))
        object.__setattr__(self, "dimension", inferred_dimension)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "sample_shape", sample_shape)
        object.__setattr__(self, "mesh_points", points)
        object.__setattr__(self, "mesh_cells", cells)
        object.__setattr__(self, "material", material_array)
        object.__setattr__(self, "metadata", _frozen_metadata(metadata))

    @property
    def x(self) -> FloatArray:
        return self.coordinates[0]

    @property
    def y(self) -> FloatArray | None:
        return None if self.dimension == 1 else self.coordinates[1]

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.values)

    def component(self, name: str) -> NumericArray:
        """Return a sampled component using case-insensitive Maxwell names."""

        canonical = _canonical_component(name)
        try:
            return self.values[canonical]
        except KeyError as exc:
            available = ", ".join(self.components)
            raise KeyError(
                f"Component {canonical!r} is unavailable; sampled components: {available}."
            ) from exc

    def vector_magnitude(self, field: str) -> FloatArray:
        """Return ``|E|`` or ``|H|`` from all available Cartesian components."""

        family = str(field).strip().upper()
        if family not in ("E", "H"):
            raise ValueError("field must be 'E' or 'H'.")
        arrays = [
            np.asarray(self.values[name])
            for name in (f"{family}x", f"{family}y", f"{family}z")
            if name in self.values
        ]
        if not arrays:
            raise KeyError(f"No {family}-field components are available.")
        magnitude = np.sqrt(sum(np.abs(array) ** 2 for array in arrays))
        magnitude = np.asarray(magnitude, dtype=np.float64)
        magnitude.setflags(write=False)
        return magnitude

    def quantity(self, component: str, quantity: str = "real") -> NumericArray:
        """Return a real, imaginary, magnitude, or phase view for plotting."""

        canonical = str(component).strip()
        family = canonical.upper()
        selected_quantity = _canonical_quantity(quantity)
        if family in ("E", "H", "|E|", "|H|"):
            field = family.replace("|", "")
            return self.vector_magnitude(field)

        values = np.asarray(self.component(canonical))
        if selected_quantity == "real":
            result = np.real(values)
        elif selected_quantity == "imag":
            result = np.imag(values)
        elif selected_quantity == "magnitude":
            result = np.abs(values)
        else:
            result = np.angle(values)
        result = np.asarray(result, dtype=np.float64)
        result.setflags(write=False)
        return result


def _finite_complex(value: complex, name: str) -> complex:
    result = complex(value)
    if not np.isfinite((result.real, result.imag)).all():
        raise ValueError(f"{name} must be finite.")
    return result


def _optional_nonnegative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class Mode:
    """One immutable mode and its sampled electric/magnetic fields."""

    neff: complex
    beta: complex
    fields: SampledFields
    index: int = 1
    polarization: str | None = None
    eigenvalue: complex | None = None
    power: complex | None = None
    normalization: str = "unnormalized"
    residual: float | None = None
    divergence_residual: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.index, (bool, np.bool_)) or int(self.index) < 1:
            raise ValueError("index must be a positive, user-facing mode number.")
        if not isinstance(self.fields, SampledFields):
            raise TypeError("fields must be a SampledFields instance.")
        if self.polarization is not None and not str(self.polarization).strip():
            raise ValueError("polarization must be a non-empty string or None.")
        if not str(self.normalization).strip():
            raise ValueError("normalization must be a non-empty string.")

        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "neff", _finite_complex(self.neff, "neff"))
        object.__setattr__(self, "beta", _finite_complex(self.beta, "beta"))
        if self.eigenvalue is not None:
            object.__setattr__(
                self, "eigenvalue", _finite_complex(self.eigenvalue, "eigenvalue")
            )
        if self.power is not None:
            object.__setattr__(self, "power", _finite_complex(self.power, "power"))
        if self.polarization is not None:
            object.__setattr__(self, "polarization", str(self.polarization))
        object.__setattr__(self, "normalization", str(self.normalization))
        object.__setattr__(
            self, "residual", _optional_nonnegative(self.residual, "residual")
        )
        object.__setattr__(
            self,
            "divergence_residual",
            _optional_nonnegative(self.divergence_residual, "divergence_residual"),
        )
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    @property
    def propagation_constant(self) -> float:
        """Legacy normalized phase constant, ``Re(neff)``."""

        return float(self.neff.real)

    @property
    def attenuation_constant(self) -> float:
        """Legacy normalized attenuation for ``exp(+j wt - j beta z)``."""

        return float(-self.neff.imag)

    @property
    def alpha(self) -> float:
        """Dimensional forward attenuation ``-Im(beta)`` in inverse metres."""

        return float(-self.beta.imag)

    def component(self, name: str) -> NumericArray:
        return self.fields.component(name)

    def quantity(self, component: str, quantity: str = "real") -> NumericArray:
        return self.fields.quantity(component, quantity)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ModeSet(Sequence[Mode]):
    """Immutable, sequence-like modes returned by either FEM solver."""

    modes: tuple[Mode, ...]
    frequency: float
    k0: float
    dimension: int
    backend: str
    metadata: Mapping[str, Any]

    def __init__(
        self,
        modes: Sequence[Mode],
        *,
        frequency: float,
        k0: float | None = None,
        dimension: int | None = None,
        backend: str = "fem",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        frozen_modes = tuple(modes)
        if any(not isinstance(mode, Mode) for mode in frozen_modes):
            raise TypeError("modes must contain only Mode instances.")
        frequency_value = float(frequency)
        if not np.isfinite(frequency_value) or frequency_value <= 0.0:
            raise ValueError("frequency must be finite and positive.")
        k0_value = (
            2.0 * np.pi * frequency_value / 299_792_458.0
            if k0 is None
            else float(k0)
        )
        if not np.isfinite(k0_value) or k0_value <= 0.0:
            raise ValueError("k0 must be finite and positive.")

        mode_dimensions = {mode.fields.dimension for mode in frozen_modes}
        if len(mode_dimensions) > 1:
            raise ValueError("All modes in a ModeSet must have the same dimension.")
        if dimension is None:
            if not mode_dimensions:
                raise ValueError("dimension is required when modes is empty.")
            dimension_value = mode_dimensions.pop()
        else:
            dimension_value = int(dimension)
            if dimension_value not in (1, 2):
                raise ValueError("dimension must be 1 or 2.")
            if mode_dimensions and mode_dimensions != {dimension_value}:
                raise ValueError("dimension does not match the sampled mode fields.")
        if not str(backend).strip():
            raise ValueError("backend must be a non-empty string.")

        object.__setattr__(self, "modes", frozen_modes)
        object.__setattr__(self, "frequency", frequency_value)
        object.__setattr__(self, "k0", k0_value)
        object.__setattr__(self, "dimension", dimension_value)
        object.__setattr__(self, "backend", str(backend))
        object.__setattr__(self, "metadata", _frozen_metadata(metadata))

    @overload
    def __getitem__(self, index: int) -> Mode: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Mode, ...]: ...

    def __getitem__(self, index: int | slice) -> Mode | tuple[Mode, ...]:
        return self.modes[index]

    def __len__(self) -> int:
        return len(self.modes)

    def __iter__(self) -> Iterator[Mode]:
        return iter(self.modes)

    def mode(self, number: int) -> Mode:
        """Return a mode by user-facing one-based number."""

        if isinstance(number, (bool, np.bool_)) or not 1 <= int(number) <= len(self):
            raise IndexError(f"mode must be between 1 and {len(self)}; got {number!r}.")
        return self.modes[int(number) - 1]

    @staticmethod
    def _readonly_vector(values: Sequence[complex]) -> ComplexArray:
        return _readonly_array(
            values,
            dtype=np.complex128,
            name="mode result vector",
        )

    @property
    def neff(self) -> ComplexArray:
        return self._readonly_vector([mode.neff for mode in self])

    @property
    def beta(self) -> ComplexArray:
        return self._readonly_vector([mode.beta for mode in self])

    @property
    def propagation_constant(self) -> FloatArray:
        return _readonly_array(
            [mode.propagation_constant for mode in self],
            dtype=np.float64,
            name="propagation constants",
        )

    @property
    def attenuation_constant(self) -> FloatArray:
        return _readonly_array(
            [mode.attenuation_constant for mode in self],
            dtype=np.float64,
            name="attenuation constants",
        )

    @property
    def components(self) -> tuple[str, ...]:
        ordered: dict[str, None] = {}
        for mode in self:
            ordered.update(dict.fromkeys(mode.fields.components))
        return tuple(ordered)

    def by_polarization(self, polarization: str) -> "ModeSet":
        requested = str(polarization).strip().casefold()
        selected = tuple(
            mode
            for mode in self
            if mode.polarization is not None
            and mode.polarization.casefold() == requested
        )
        return ModeSet(
            selected,
            frequency=self.frequency,
            k0=self.k0,
            dimension=self.dimension,
            backend=self.backend,
            metadata=self.metadata,
        )


__all__ = ["Mode", "ModeSet", "SampledFields"]
