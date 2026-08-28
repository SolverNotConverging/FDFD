"""Immutable public results for FEM transmission-line calculations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import C_0
from ..results import Mode, ModeSet, SampledFields
from .electrostatics import QuasiTEMSolution

if TYPE_CHECKING:  # pragma: no cover - imports used only by static checkers
    from .specs import TransmissionLineSpec
    from .templates import BuiltTransmissionLine


def _readonly_array(
    value: ArrayLike,
    *,
    dtype: np.dtype[Any] | type[Any],
    name: str,
    allow_nan: bool = False,
) -> NDArray[Any]:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    valid = ~np.isinf(array) if allow_nan else np.isfinite(array)
    if not np.all(valid):
        qualifier = "finite values or NaN" if allow_nan else "only finite values"
        raise ValueError(f"{name} must contain {qualifier}.")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        dtype = np.complex128 if np.iscomplexobj(value) else np.float64
        return _readonly_array(value, dtype=dtype, name="metadata array", allow_nan=True)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _finite_complex(value: complex, name: str) -> complex:
    result = complex(value)
    if not np.isfinite((result.real, result.imag)).all():
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_real(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class TransmissionLineResult:
    """One unit-voltage forward TEM or quasi-TEM line mode and its metrics."""

    spec: "TransmissionLineSpec"
    label: str
    frequency: float
    mode: Mode
    modes: ModeSet
    neff: complex
    characteristic_impedance: complex
    wave_impedance: complex
    capacitance_per_length: complex
    vacuum_capacitance_per_length: float
    inductance_per_length: float
    voltage: complex
    current: complex
    power: complex
    electric_potential: NDArray[np.complex128]
    vacuum_potential: NDArray[np.complex128]
    local_wave_impedance: NDArray[np.complex128]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        if not label:
            raise ValueError("label must be non-empty.")
        frequency = _positive_real(self.frequency, "frequency")
        if not isinstance(self.mode, Mode):
            raise TypeError("mode must be a FEM_Mode_Solver.results.Mode instance.")
        if not isinstance(self.modes, ModeSet):
            raise TypeError("modes must be a FEM_Mode_Solver.results.ModeSet instance.")
        if len(self.modes) != 1 or self.modes[0] is not self.mode:
            raise ValueError("modes must contain the same single quasi-TEM mode as mode.")
        neff = _finite_complex(self.neff, "neff")
        characteristic = _finite_complex(
            self.characteristic_impedance, "characteristic_impedance"
        )
        wave = _finite_complex(self.wave_impedance, "wave_impedance")
        capacitance = _finite_complex(
            self.capacitance_per_length, "capacitance_per_length"
        )
        vacuum_capacitance = _positive_real(
            self.vacuum_capacitance_per_length,
            "vacuum_capacitance_per_length",
        )
        inductance = _positive_real(
            self.inductance_per_length, "inductance_per_length"
        )
        voltage = _finite_complex(self.voltage, "voltage")
        current = _finite_complex(self.current, "current")
        power = _finite_complex(self.power, "power")
        electric_potential = _readonly_array(
            self.electric_potential,
            dtype=np.complex128,
            name="electric_potential",
        )
        vacuum_potential = _readonly_array(
            self.vacuum_potential,
            dtype=np.complex128,
            name="vacuum_potential",
        )
        local_impedance = _readonly_array(
            self.local_wave_impedance,
            dtype=np.complex128,
            name="local_wave_impedance",
            allow_nan=True,
        )
        if electric_potential.ndim != 1 or vacuum_potential.shape != electric_potential.shape:
            raise ValueError(
                "electric_potential and vacuum_potential must be equal-length vectors."
            )
        if local_impedance.shape != self.mode.fields.sample_shape:
            raise ValueError(
                "local_wave_impedance must match the modal field sample shape."
            )
        if neff.real <= 0.0 or characteristic.real <= 0.0 or capacitance.real <= 0.0:
            raise ValueError(
                "neff, characteristic impedance, and capacitance must use positive-real "
                "physical branches."
            )

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "neff", neff)
        object.__setattr__(self, "characteristic_impedance", characteristic)
        object.__setattr__(self, "wave_impedance", wave)
        object.__setattr__(self, "capacitance_per_length", capacitance)
        object.__setattr__(self, "vacuum_capacitance_per_length", vacuum_capacitance)
        object.__setattr__(self, "inductance_per_length", inductance)
        object.__setattr__(self, "voltage", voltage)
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "electric_potential", electric_potential)
        object.__setattr__(self, "vacuum_potential", vacuum_potential)
        object.__setattr__(self, "local_wave_impedance", local_impedance)
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @classmethod
    def from_solution(
        cls,
        spec: "TransmissionLineSpec",
        built: "BuiltTransmissionLine",
        solution: QuasiTEMSolution,
        *,
        frequency: float,
    ) -> "TransmissionLineResult":
        """Wrap quadrature fields in the mode solver's common result model."""

        if not isinstance(solution, QuasiTEMSolution):
            raise TypeError("solution must be a QuasiTEMSolution instance.")
        frequency_value = _positive_real(frequency, "frequency")
        try:
            mesh_data = built.solver.mesh_data
            label = str(built.label)
            signal_boundaries = tuple(built.signal_boundaries)
            reference_boundaries = tuple(built.reference_boundaries)
        except AttributeError as exc:
            raise TypeError(
                "built must provide solver, label, signal_boundaries, and "
                "reference_boundaries."
            ) from exc

        sample_shape = solution.weights.shape
        element_count = mesh_data.elements.shape[0]
        if len(sample_shape) != 2 or sample_shape[0] != element_count:
            raise ValueError(
                "The quasi-TEM sampler must preserve the native mesh-element axis."
            )
        sample_count = int(np.prod(sample_shape))
        owners = np.repeat(
            np.arange(element_count, dtype=np.int64), sample_shape[1]
        )
        coordinates = np.moveaxis(solution.coordinates, 0, -1).reshape(-1, 2)
        zero = np.zeros(sample_count, dtype=np.complex128)
        values = {
            "Ex": solution.electric_field[0].reshape(-1),
            "Ey": solution.electric_field[1].reshape(-1),
            "Ez": zero,
            "Hx": solution.magnetic_field[0].reshape(-1),
            "Hy": solution.magnetic_field[1].reshape(-1),
            "Hz": zero,
        }
        sampled = SampledFields(
            coordinates,
            values,
            dimension=2,
            mesh_points=mesh_data.nodes,
            mesh_cells=mesh_data.elements,
            material=solution.relative_epsilon.reshape(-1),
            metadata={
                "component_order": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                "sampling": "element-quadrature",
                "sample_element_indices": owners,
                "element_quadrature_shape": sample_shape,
                "polarization": "quasi-TEM",
                "normalization": "unit-voltage",
                "time_convention": "exp(+1j*omega*t - 1j*beta*z)",
            },
        )
        k0 = 2.0 * np.pi * frequency_value / C_0
        mode = Mode(
            neff=solution.neff,
            beta=k0 * solution.neff,
            fields=sampled,
            index=1,
            polarization="quasi-TEM",
            power=solution.power,
            normalization="unit-voltage",
            metadata={
                "classification": "propagating",
                "direction": "forward",
                "line_label": label,
                "characteristic_impedance": solution.characteristic_impedance,
                "wave_impedance": solution.wave_impedance,
                "voltage": solution.voltage,
                "current": solution.current,
            },
        )
        modes = ModeSet(
            (mode,),
            frequency=frequency_value,
            k0=k0,
            dimension=2,
            backend="fem-quasi-tem",
            metadata={
                "formulation": "scalar-quasi-TEM-P1",
                "line_label": label,
            },
        )
        return cls(
            spec=spec,
            label=label,
            frequency=frequency_value,
            mode=mode,
            modes=modes,
            neff=solution.neff,
            characteristic_impedance=solution.characteristic_impedance,
            wave_impedance=solution.wave_impedance,
            capacitance_per_length=solution.capacitance_per_length,
            vacuum_capacitance_per_length=solution.vacuum_capacitance_per_length,
            inductance_per_length=solution.inductance_per_length,
            voltage=solution.voltage,
            current=solution.current,
            power=solution.power,
            electric_potential=solution.electric_potential,
            vacuum_potential=solution.vacuum_potential,
            local_wave_impedance=solution.local_wave_impedance.reshape(-1),
            metadata={
                **dict(solution.metadata),
                "line_type": type(spec).__name__,
                "signal_boundaries": signal_boundaries,
                "reference_boundaries": reference_boundaries,
            },
        )

    @property
    def fields(self) -> SampledFields:
        return self.mode.fields

    @property
    def capacitance(self) -> complex:
        return self.capacitance_per_length

    @property
    def vacuum_capacitance(self) -> float:
        return self.vacuum_capacitance_per_length

    @property
    def inductance(self) -> float:
        return self.inductance_per_length

    @property
    def Zc(self) -> complex:  # noqa: N802 - conventional line-parameter name
        return self.characteristic_impedance

    @property
    def Zw(self) -> complex:  # noqa: N802 - conventional line-parameter name
        return self.wave_impedance

    def visualize(self, **kwargs: Any) -> Any:
        """Plot the transverse electric and magnetic fields."""

        from .visualization import visualize

        return visualize(self, **kwargs)

    def visualize_with_gui(self, **kwargs: Any) -> Any:
        """Open the interactive transmission-line vector-field viewer."""

        from .visualization import visualize_with_gui

        return visualize_with_gui(self, **kwargs)


__all__ = ["TransmissionLineResult"]
