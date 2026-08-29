"""Scalar P1 FEM extraction for TEM and quasi-TEM transmission lines.

The electric potential is solved on the physical cross-section using

``div(epsilon grad(phi)) = 0``

with one volt on all signal conductors and zero volts on the named reference
conductors and the exterior PEC wall.  A second solve on the identical mesh
with vacuum permittivity supplies the magnetic-field dual and the vacuum
capacitance used by the standard quasi-TEM capacitance method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, FacetBasis, MeshTri, asm
from skfem.element import ElementTriP1

from ..assembly import evaluate_material
from ..constants import C_0 as SPEED_OF_LIGHT
from ..constants import EPSILON_0, MU_0
from ..exceptions import ConfigurationError, SolverError

if TYPE_CHECKING:  # pragma: no cover - imports used only by static checkers
    from .templates import BuiltTransmissionLine


ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


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
        raise SolverError(f"{name} must contain {qualifier}.")
    # A bytes-backed snapshot cannot subsequently be made writable by a caller.
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
        raise SolverError(f"{name} must be finite.")
    return result


def _positive_real(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise SolverError(f"{name} must be finite and positive.")
    return result


def _nonnegative_real(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise SolverError(f"{name} must be finite and nonnegative.")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class QuasiTEMSolution:
    """Immutable quadrature-level solution of the quasi-TEM field problem."""

    electric_potential: ComplexArray
    vacuum_potential: ComplexArray
    coordinates: FloatArray
    weights: FloatArray
    relative_epsilon: ComplexArray
    electric_field: ComplexArray
    magnetic_field: ComplexArray
    capacitance_per_length: complex
    vacuum_capacitance_per_length: float
    inductance_per_length: float
    neff: complex
    characteristic_impedance: complex
    wave_impedance: complex
    local_wave_impedance: ComplexArray
    voltage: complex
    current: complex
    power: complex
    metadata: Mapping[str, Any] = field(default_factory=dict)
    resistance_per_length: float = 0.0
    conductance_per_length: float = 0.0
    surface_resistance: float = 0.0
    external_inductance_per_length: float | None = None

    def __post_init__(self) -> None:
        potential = _readonly_array(
            self.electric_potential,
            dtype=np.complex128,
            name="electric_potential",
        )
        vacuum = _readonly_array(
            self.vacuum_potential,
            dtype=np.complex128,
            name="vacuum_potential",
        )
        coordinates = _readonly_array(
            self.coordinates, dtype=np.float64, name="coordinates"
        )
        weights = _readonly_array(self.weights, dtype=np.float64, name="weights")
        epsilon = _readonly_array(
            self.relative_epsilon,
            dtype=np.complex128,
            name="relative_epsilon",
        )
        electric = _readonly_array(
            self.electric_field, dtype=np.complex128, name="electric_field"
        )
        magnetic = _readonly_array(
            self.magnetic_field, dtype=np.complex128, name="magnetic_field"
        )
        local_impedance = _readonly_array(
            self.local_wave_impedance,
            dtype=np.complex128,
            name="local_wave_impedance",
            allow_nan=True,
        )

        if potential.ndim != 1 or vacuum.shape != potential.shape:
            raise SolverError(
                "electric and vacuum nodal potentials must be equal-length vectors."
            )
        if coordinates.ndim != 3 or coordinates.shape[0] != 2:
            raise SolverError("coordinates must have shape (2, elements, quadrature).")
        sample_shape = coordinates.shape[1:]
        if weights.shape != sample_shape or epsilon.shape != sample_shape:
            raise SolverError(
                "weights and relative_epsilon must match the element-quadrature shape."
            )
        if electric.shape != (2, *sample_shape) or magnetic.shape != electric.shape:
            raise SolverError(
                "electric_field and magnetic_field must have shape "
                "(2, elements, quadrature)."
            )
        if local_impedance.shape != sample_shape:
            raise SolverError(
                "local_wave_impedance must match the element-quadrature shape."
            )
        if np.any(weights <= 0.0):
            raise SolverError("quadrature weights must be strictly positive.")

        capacitance = _finite_complex(
            self.capacitance_per_length, "capacitance_per_length"
        )
        if capacitance.real <= 0.0:
            raise SolverError("capacitance_per_length must have a positive real part.")
        vacuum_capacitance = _positive_real(
            self.vacuum_capacitance_per_length,
            "vacuum_capacitance_per_length",
        )
        inductance = _positive_real(
            self.inductance_per_length, "inductance_per_length"
        )
        neff = _finite_complex(self.neff, "neff")
        characteristic = _finite_complex(
            self.characteristic_impedance, "characteristic_impedance"
        )
        wave = _finite_complex(self.wave_impedance, "wave_impedance")
        voltage = _finite_complex(self.voltage, "voltage")
        current = _finite_complex(self.current, "current")
        power = _finite_complex(self.power, "power")
        resistance = _nonnegative_real(
            self.resistance_per_length, "resistance_per_length"
        )
        conductance = _nonnegative_real(
            self.conductance_per_length, "conductance_per_length"
        )
        surface_resistance = _nonnegative_real(
            self.surface_resistance, "surface_resistance"
        )
        external_inductance = (
            inductance
            if self.external_inductance_per_length is None
            else _positive_real(
                self.external_inductance_per_length,
                "external_inductance_per_length",
            )
        )
        inductance_tolerance = 64.0 * np.finfo(float).eps * max(
            inductance, external_inductance
        )
        if inductance + inductance_tolerance < external_inductance:
            raise SolverError(
                "inductance_per_length cannot be smaller than its external part."
            )
        if neff.real <= 0.0:
            raise SolverError("neff must select the forward branch with Re(neff) > 0.")
        if characteristic.real <= 0.0:
            raise SolverError(
                "characteristic_impedance must select the branch with positive real part."
            )
        if abs(voltage) == 0.0 or abs(current) == 0.0:
            raise SolverError("voltage and current must be nonzero.")

        object.__setattr__(self, "electric_potential", potential)
        object.__setattr__(self, "vacuum_potential", vacuum)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "relative_epsilon", epsilon)
        object.__setattr__(self, "electric_field", electric)
        object.__setattr__(self, "magnetic_field", magnetic)
        object.__setattr__(self, "local_wave_impedance", local_impedance)
        object.__setattr__(self, "capacitance_per_length", capacitance)
        object.__setattr__(self, "vacuum_capacitance_per_length", vacuum_capacitance)
        object.__setattr__(self, "inductance_per_length", inductance)
        object.__setattr__(self, "neff", neff)
        object.__setattr__(self, "characteristic_impedance", characteristic)
        object.__setattr__(self, "wave_impedance", wave)
        object.__setattr__(self, "voltage", voltage)
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "resistance_per_length", resistance)
        object.__setattr__(self, "conductance_per_length", conductance)
        object.__setattr__(self, "surface_resistance", surface_resistance)
        object.__setattr__(
            self, "external_inductance_per_length", external_inductance
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    # Short electromagnetic aliases are convenient in numerical workflows.
    @property
    def potential(self) -> ComplexArray:
        return self.electric_potential

    @property
    def et(self) -> ComplexArray:
        return self.electric_field

    @property
    def ht(self) -> ComplexArray:
        return self.magnetic_field

    @property
    def element_coordinates(self) -> FloatArray:
        return self.coordinates

    @property
    def quadrature_weights(self) -> FloatArray:
        return self.weights

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
    def C(self) -> complex:  # noqa: N802 - conventional line-parameter name
        return self.capacitance_per_length

    @property
    def C0(self) -> float:  # noqa: N802 - conventional line-parameter name
        return self.vacuum_capacitance_per_length

    @property
    def L(self) -> float:  # noqa: N802 - conventional line-parameter name
        return self.inductance_per_length

    @property
    def R(self) -> float:  # noqa: N802 - conventional line-parameter name
        return self.resistance_per_length

    @property
    def G(self) -> float:  # noqa: N802 - conventional line-parameter name
        return self.conductance_per_length

    @property
    def series_impedance_per_length(self) -> complex:
        """Return ``R' + j omega L'`` in ohms per metre."""

        try:
            frequency = _positive_real(self.metadata["frequency"], "frequency")
        except KeyError as exc:  # defensive invariant for manually built solutions
            raise SolverError(
                "solution metadata must contain frequency to form the series impedance."
            ) from exc
        omega = 2.0 * np.pi * frequency
        return complex(
            self.resistance_per_length,
            omega * self.inductance_per_length,
        )

    @property
    def shunt_admittance_per_length(self) -> complex:
        """Return ``G' + j omega C'`` in siemens per metre."""

        try:
            frequency = _positive_real(self.metadata["frequency"], "frequency")
        except KeyError as exc:  # defensive invariant for manually built solutions
            raise SolverError(
                "solution metadata must contain frequency to form the shunt admittance."
            ) from exc
        omega = 2.0 * np.pi * frequency
        return complex(
            self.conductance_per_length,
            omega * self.capacitance_per_length.real,
        )

    @property
    def Zc(self) -> complex:  # noqa: N802 - conventional line-parameter name
        return self.characteristic_impedance

    @property
    def Zw(self) -> complex:  # noqa: N802 - conventional line-parameter name
        return self.wave_impedance


@BilinearForm(dtype=np.complex128)
def _electrostatic_form(u: object, v: object, w: object) -> object:
    """Complex sesquilinear transverse dielectric stiffness."""

    epsilon = w.epsilon
    return epsilon[0] * u.grad[0] * np.conj(v.grad[0]) + epsilon[1] * u.grad[
        1
    ] * np.conj(v.grad[1])


@BilinearForm(dtype=np.complex128)
def _boundary_mass_form(u: object, v: object, w: object) -> object:
    """Hermitian L2 mass form used to recover conductor surface charge."""

    return u * np.conj(v)


def _boundary_names(value: object, role: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise ConfigurationError(f"{role} must be a boundary name or sequence of names.")
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ConfigurationError(f"{role} contains an empty or non-string name.")
        name = item.strip()
        if name not in names:
            names.append(name)
    if not names:
        raise ConfigurationError(f"{role} must contain at least one boundary name.")
    return tuple(names)


def _validated_facets(
    mesh: MeshTri,
    boundary_facets: Mapping[str, ArrayLike],
    names: tuple[str, ...],
    role: str,
    *,
    allow_empty: bool = False,
) -> IntArray:
    available = ", ".join(sorted(boundary_facets)) or "none"
    pieces: list[IntArray] = []
    mesh_boundary = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    for name in names:
        if name not in boundary_facets:
            raise ConfigurationError(
                f"Unknown {role} boundary {name!r}; available mesh boundaries: {available}."
            )
        raw = np.asarray(boundary_facets[name])
        if raw.ndim != 1 or raw.dtype.kind == "b":
            raise ConfigurationError(
                f"Mesh facets for {role} boundary {name!r} must be a 1D integer array."
            )
        try:
            numeric = np.asarray(raw, dtype=np.complex128)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                f"Mesh facets for {role} boundary {name!r} must be integers."
            ) from exc
        if (
            not np.isfinite(numeric).all()
            or np.any(numeric.imag != 0.0)
            or np.any(numeric.real != np.floor(numeric.real))
        ):
            raise ConfigurationError(
                f"Mesh facets for {role} boundary {name!r} must be finite integers."
            )
        facets = np.unique(numeric.real.astype(np.int64))
        if facets.size == 0:
            if allow_empty:
                continue
            raise ConfigurationError(
                f"{role.capitalize()} boundary {name!r} has no exposed mesh facets."
            )
        if facets[0] < 0 or facets[-1] >= mesh.nfacets:
            raise ConfigurationError(
                f"Mesh facets for {role} boundary {name!r} are outside the mesh."
            )
        if not np.all(np.isin(facets, mesh_boundary)):
            raise ConfigurationError(
                f"Mesh facets for {role} boundary {name!r} must be boundary facets."
            )
        pieces.append(facets)
    if not pieces:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(pieces)).astype(np.int64, copy=False)


def _projected_conductor_geometry_factors(
    mesh: MeshTri,
    boundary_facets: Mapping[str, ArrayLike],
    boundary_names: tuple[str, ...],
    vacuum_reaction: ComplexArray,
    vacuum_capacitance: float,
    *,
    quadrature_order: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Recover ``integral |H_t/I|^2 dl`` on each physical conductor.

    The nodal vacuum reaction is the weak surface-charge functional.  A
    boundary L2 mass solve recovers a continuous P1 charge density before its
    squared norm is integrated.  This is substantially more accurate on
    curved conductors than sampling the discontinuous adjacent-element
    gradient directly.  Since ``H_t/I = q_s / C0`` for the vacuum dual, the
    resulting geometry factor has units of inverse metres.
    """

    factors: dict[str, float] = {}
    projection_residuals: dict[str, float] = {}
    element = ElementTriP1()
    tiny = np.finfo(float).tiny
    for name in boundary_names:
        facets = _validated_facets(
            mesh,
            boundary_facets,
            (name,),
            "conductor",
        )
        facet_basis = FacetBasis(
            mesh,
            element,
            facets=facets,
            intorder=quadrature_order,
        )
        boundary_mass = asm(_boundary_mass_form, facet_basis).astype(
            np.complex128, copy=False
        )
        dofs = np.asarray(
            facet_basis.get_dofs(facets=facets).all(), dtype=np.int64
        )
        if dofs.size == 0:  # defensive; _validated_facets already rejects this
            raise SolverError(
                f"Conductor boundary {name!r} has no scalar trace degrees of freedom."
            )
        restricted_mass = boundary_mass[dofs][:, dofs].tocsc()
        right_hand_side = np.asarray(vacuum_reaction[dofs], dtype=np.complex128)
        try:
            charge_density = splu(restricted_mass).solve(right_hand_side)
        except (RuntimeError, ValueError) as exc:
            raise SolverError(
                f"The boundary mass projection is singular on conductor {name!r}."
            ) from exc
        projected_reaction = np.asarray(
            restricted_mass @ charge_density, dtype=np.complex128
        )
        residual = float(np.linalg.norm(projected_reaction - right_hand_side)) / max(
            float(np.linalg.norm(right_hand_side)), tiny
        )
        if not np.isfinite(residual) or residual > 1e-10:
            raise SolverError(
                "The conductor boundary projection did not satisfy its residual "
                f"on {name!r} (relative residual {residual:.3e})."
            )
        squared_charge_norm = complex(
            np.vdot(charge_density, restricted_mass @ charge_density)
        )
        norm_tolerance = 256.0 * np.finfo(float).eps * max(
            abs(squared_charge_norm), tiny
        )
        if (
            not np.isfinite(
                (squared_charge_norm.real, squared_charge_norm.imag)
            ).all()
            or squared_charge_norm.real < -norm_tolerance
            or abs(squared_charge_norm.imag) > norm_tolerance
        ):
            raise SolverError(
                f"The projected surface-current norm is invalid on {name!r}."
            )
        factor = max(0.0, squared_charge_norm.real) / vacuum_capacitance**2
        if not np.isfinite(factor) or factor <= 0.0:
            raise SolverError(
                f"The conductor geometry factor is not positive on {name!r}."
            )
        factors[name] = float(factor)
        projection_residuals[name] = residual
    return factors, projection_residuals


def _solve_dirichlet(
    stiffness: Any,
    signal_dofs: IntArray,
    zero_dofs: IntArray,
) -> tuple[ComplexArray, ComplexArray, float]:
    count = int(stiffness.shape[0])
    overlap = np.intersect1d(signal_dofs, zero_dofs, assume_unique=True)
    if overlap.size:
        raise ConfigurationError(
            "Signal and zero-volt reference boundaries share FEM degrees of freedom."
        )
    fixed = np.union1d(signal_dofs, zero_dofs).astype(np.int64, copy=False)
    if signal_dofs.size == 0 or zero_dofs.size == 0:
        raise ConfigurationError(
            "The electrostatic solve requires nonempty signal and reference conductors."
        )
    free = np.setdiff1d(np.arange(count, dtype=np.int64), fixed, assume_unique=True)
    if free.size == 0:
        raise ConfigurationError(
            "The conductor constraints leave no free electrostatic FEM degrees of freedom."
        )

    potential = np.zeros(count, dtype=np.complex128)
    potential[signal_dofs] = 1.0
    reduced = stiffness[free][:, free].tocsc()
    right_hand_side = -stiffness[free][:, fixed] @ potential[fixed]
    try:
        potential[free] = splu(reduced).solve(
            np.asarray(right_hand_side, dtype=np.complex128)
        )
    except (RuntimeError, ValueError) as exc:
        raise SolverError(
            "The scalar electrostatic stiffness is singular; verify that every "
            "connected dielectric region has a voltage reference."
        ) from exc
    if not np.isfinite(potential).all():
        raise SolverError("The scalar electrostatic solve produced a non-finite potential.")

    reaction = np.asarray(stiffness @ potential, dtype=np.complex128)
    free_residual = float(np.linalg.norm(reaction[free]))
    reaction_scale = max(float(np.linalg.norm(reaction[fixed])), np.finfo(float).tiny)
    relative_residual = free_residual / reaction_scale
    if not np.isfinite(relative_residual) or relative_residual > 1e-8:
        raise SolverError(
            "The scalar electrostatic solve did not satisfy its free-DOF residual "
            f"(relative residual {relative_residual:.3e})."
        )
    return potential, reaction, relative_residual


def _positive_real_root(value: complex, name: str) -> complex:
    root = complex(np.sqrt(complex(value)))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(root))
    if root.real < -tolerance or (
        abs(root.real) <= tolerance and root.imag > tolerance
    ):
        root = -root
    if not np.isfinite((root.real, root.imag)).all() or root.real <= 0.0:
        raise SolverError(f"{name} has no physical branch with positive real part.")
    return root


def solve_quasi_tem(
    built: "BuiltTransmissionLine",
    *,
    frequency: float,
    quadrature_order: int = 4,
) -> QuasiTEMSolution:
    """Solve one forward, unit-voltage quasi-TEM transmission-line mode.

    ``built`` must refer to an already-discretized line template.  The solve
    uses the native physical-coordinate mesh; it deliberately does not use the
    full-vector mode solver's dimensionless ``k0``-scaled basis.
    """

    if isinstance(frequency, (bool, np.bool_, str, bytes)):
        raise ConfigurationError("frequency must be finite and positive.")
    frequency_value = float(frequency)
    if not np.isfinite(frequency_value) or frequency_value <= 0.0:
        raise ConfigurationError("frequency must be finite and positive.")
    if (
        isinstance(quadrature_order, (bool, np.bool_))
        or int(quadrature_order) != quadrature_order
        or int(quadrature_order) < 2
    ):
        raise ConfigurationError("quadrature_order must be an integer of at least two.")
    order = int(quadrature_order)

    try:
        solver = built.solver
        mesh_data = solver.mesh_data
        geometry = solver.geometry
        signal_names = _boundary_names(built.signal_boundaries, "signal_boundaries")
        reference_names = _boundary_names(
            built.reference_boundaries, "reference_boundaries"
        )
    except AttributeError as exc:
        raise TypeError(
            "built must provide solver, signal_boundaries, and reference_boundaries."
        ) from exc
    metal_conductivity = getattr(built, "metal_conductivity", None)
    if metal_conductivity is not None:
        if isinstance(metal_conductivity, (bool, np.bool_, str, bytes)):
            raise ConfigurationError(
                "metal_conductivity must be finite and positive in siemens per metre."
            )
        try:
            metal_conductivity = float(metal_conductivity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                "metal_conductivity must be finite and positive in siemens per metre."
            ) from exc
        if not np.isfinite(metal_conductivity) or metal_conductivity <= 0.0:
            raise ConfigurationError(
                "metal_conductivity must be finite and positive in siemens per metre."
            )
    mesh = mesh_data.mesh
    if not isinstance(mesh, MeshTri):
        raise TypeError("built.solver.mesh_data.mesh must be a scikit-fem MeshTri.")
    facet_map = mesh_data.boundary_facets
    signal_facets = _validated_facets(
        mesh, facet_map, signal_names, "signal"
    )
    reference_facets = _validated_facets(
        mesh, facet_map, reference_names, "reference"
    )
    outer_facets = _validated_facets(
        mesh,
        facet_map,
        ("outer_pec",),
        "outer PEC",
        allow_empty=True,
    )
    ground_facets = np.union1d(reference_facets, outer_facets).astype(
        np.int64, copy=False
    )
    if np.intersect1d(signal_facets, ground_facets, assume_unique=True).size:
        raise ConfigurationError("Signal and grounded mesh facet sets must be disjoint.")

    basis = Basis(mesh, ElementTriP1(), intorder=order)
    coordinates = np.asarray(basis.global_coordinates(), dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[0] != 2:
        raise SolverError(
            "scikit-fem returned an unexpected scalar-basis quadrature layout."
        )
    epsilon_r, _ = evaluate_material(
        geometry.material_at, coordinates[0], coordinates[1]
    )
    transverse_epsilon = np.asarray(epsilon_r[:2], dtype=np.complex128)
    if np.any(np.abs(transverse_epsilon) <= np.finfo(float).tiny):
        raise ConfigurationError(
            "Transverse relative permittivity must be nonzero at every quadrature point."
        )
    if np.any(np.real(transverse_epsilon) <= 0.0):
        raise ConfigurationError(
            "The quasi-TEM capacitance formulation requires positive-real transverse "
            "permittivity."
        )
    absolute_epsilon = EPSILON_0 * transverse_epsilon
    stiffness = asm(
        _electrostatic_form,
        basis,
        epsilon=absolute_epsilon,
    ).astype(np.complex128, copy=False)
    vacuum_epsilon = np.full(
        (2, *coordinates.shape[1:]), EPSILON_0, dtype=np.complex128
    )
    vacuum_stiffness = asm(
        _electrostatic_form,
        basis,
        epsilon=vacuum_epsilon,
    ).astype(np.complex128, copy=False)

    signal_dofs = np.asarray(
        basis.get_dofs(facets=signal_facets).all(), dtype=np.int64
    )
    zero_dofs = np.asarray(
        basis.get_dofs(facets=ground_facets).all(), dtype=np.int64
    )
    potential, reaction, material_residual = _solve_dirichlet(
        stiffness, signal_dofs, zero_dofs
    )
    vacuum_potential, vacuum_reaction, vacuum_residual = _solve_dirichlet(
        vacuum_stiffness, signal_dofs, zero_dofs
    )

    capacitance = complex(np.sum(reaction[signal_dofs]))
    vacuum_capacitance_complex = complex(np.sum(vacuum_reaction[signal_dofs]))
    material_energy = complex(np.vdot(potential, stiffness @ potential))
    vacuum_energy = complex(
        np.vdot(vacuum_potential, vacuum_stiffness @ vacuum_potential)
    )
    capacitance_scale = max(abs(capacitance), np.finfo(float).tiny)
    vacuum_scale = max(abs(vacuum_capacitance_complex), np.finfo(float).tiny)
    if abs(material_energy - capacitance) / capacitance_scale > 1e-8:
        raise SolverError(
            "Signal reaction and electrostatic energy disagree for the dielectric solve."
        )
    if abs(vacuum_energy - vacuum_capacitance_complex) / vacuum_scale > 1e-8:
        raise SolverError(
            "Signal reaction and electrostatic energy disagree for the vacuum solve."
        )
    if not np.isfinite((capacitance.real, capacitance.imag)).all() or capacitance.real <= 0.0:
        raise SolverError("The extracted capacitance is not finite and positive-real.")
    vacuum_imag_tolerance = 1e-10 * max(1.0, abs(vacuum_capacitance_complex.real))
    if (
        not np.isfinite(
            (vacuum_capacitance_complex.real, vacuum_capacitance_complex.imag)
        ).all()
        or vacuum_capacitance_complex.real <= 0.0
        or abs(vacuum_capacitance_complex.imag) > vacuum_imag_tolerance
    ):
        raise SolverError("The extracted vacuum capacitance is not finite and positive-real.")
    vacuum_capacitance = float(vacuum_capacitance_complex.real)

    omega = 2.0 * np.pi * frequency_value
    external_inductance = 1.0 / (SPEED_OF_LIGHT**2 * vacuum_capacitance)
    raw_conductance = -omega * capacitance.imag
    conductance_tolerance = (
        256.0
        * np.finfo(float).eps
        * omega
        * max(abs(capacitance), np.finfo(float).tiny)
    )
    if raw_conductance < -conductance_tolerance:
        raise SolverError(
            "The extracted dielectric shunt conductance is active; passive material "
            "loss requires Im(capacitance_per_length) <= 0."
        )
    conductance = float(max(0.0, raw_conductance))

    conductor_names = (*signal_names, *reference_names)
    if metal_conductivity is None:
        conductor_geometry_factors = {name: 0.0 for name in conductor_names}
        projection_residuals = {name: 0.0 for name in conductor_names}
        total_geometry_factor = 0.0
        surface_resistance = 0.0
        skin_depth: float | None = None
        resistance = 0.0
        inductance = external_inductance
        # Preserve the historical PEC/dielectric-only arithmetic exactly.
        neff = _positive_real_root(capacitance / vacuum_capacitance, "neff")
        characteristic_impedance = _positive_real_root(
            inductance / capacitance, "characteristic impedance"
        )
        beta = omega * neff / SPEED_OF_LIGHT
    else:
        conductor_geometry_factors, projection_residuals = (
            _projected_conductor_geometry_factors(
                mesh,
                facet_map,
                conductor_names,
                vacuum_reaction,
                vacuum_capacitance,
                quadrature_order=order,
            )
        )
        total_geometry_factor = float(sum(conductor_geometry_factors.values()))
        if not np.isfinite(total_geometry_factor) or total_geometry_factor <= 0.0:
            raise SolverError(
                "The total conductor geometry factor must be finite and positive."
            )
        surface_resistance = float(
            np.sqrt(np.pi * frequency_value * MU_0 / metal_conductivity)
        )
        skin_depth = float(
            np.sqrt(1.0 / (np.pi * frequency_value * MU_0 * metal_conductivity))
        )
        resistance = float(surface_resistance * total_geometry_factor)
        inductance = float(external_inductance + resistance / omega)
        series_impedance = complex(resistance, omega * inductance)
        shunt_admittance = complex(conductance, omega * capacitance.real)
        beta = _positive_real_root(
            -series_impedance * shunt_admittance,
            "complex phase constant",
        )
        neff = beta * SPEED_OF_LIGHT / omega
        characteristic_impedance = _positive_real_root(
            series_impedance / shunt_admittance,
            "characteristic impedance",
        )

    series_impedance = complex(resistance, omega * inductance)
    shunt_admittance = complex(conductance, omega * capacitance.real)
    passive_beta_tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(beta))
    if beta.imag > passive_beta_tolerance or neff.imag > (
        passive_beta_tolerance * SPEED_OF_LIGHT / omega
    ):
        raise SolverError(
            "The RLGC extraction selected an active forward propagation branch."
        )

    dielectric_field = np.asarray(-basis.interpolate(potential).grad, dtype=np.complex128)
    vacuum_field = np.asarray(
        -basis.interpolate(vacuum_potential).grad, dtype=np.complex128
    )
    if dielectric_field.shape != coordinates.shape or vacuum_field.shape != coordinates.shape:
        raise SolverError("scikit-fem returned an unexpected P1 gradient layout.")

    voltage = 1.0 + 0.0j
    current = voltage / characteristic_impedance
    # z-hat cross E_air = (-E_air_y, E_air_x).  Dividing D_air by C0
    # normalizes the circulation to one ampere; multiply by the actual current.
    unit_current_h = np.stack(
        (
            -EPSILON_0 * vacuum_field[1] / vacuum_capacitance,
            EPSILON_0 * vacuum_field[0] / vacuum_capacitance,
        ),
        axis=0,
    )
    magnetic_field = np.asarray(unit_current_h * current, dtype=np.complex128)
    weights = np.asarray(basis.dx, dtype=np.float64)
    poynting_numerator = dielectric_field[0] * np.conj(magnetic_field[1]) - dielectric_field[
        1
    ] * np.conj(magnetic_field[0])
    power = complex(0.5 * np.sum(weights * poynting_numerator))
    circuit_forward_power = float(
        0.5 * np.real(voltage * np.conj(current))
    )
    if not np.isfinite(circuit_forward_power) or circuit_forward_power <= 0.0:
        raise SolverError(
            "The RLGC characteristic impedance produced non-positive forward power."
        )
    conductor_loss = float(0.5 * resistance * abs(current) ** 2)
    dielectric_loss = float(0.5 * conductance * abs(voltage) ** 2)
    dissipated_power = conductor_loss + dielectric_loss
    attenuation = max(0.0, float(-beta.imag))
    attenuation_power_decay = float(2.0 * attenuation * circuit_forward_power)
    balance_scale = max(
        dissipated_power,
        attenuation_power_decay,
        np.finfo(float).tiny,
    )
    power_balance_residual = abs(
        dissipated_power - attenuation_power_decay
    ) / balance_scale
    if not np.isfinite(power_balance_residual) or power_balance_residual > 1e-9:
        raise SolverError(
            "The extracted RLGC parameters violate passive modal power balance "
            f"(relative residual {power_balance_residual:.3e})."
        )
    magnetic_norm = np.abs(magnetic_field[0]) ** 2 + np.abs(magnetic_field[1]) ** 2
    integrated_magnetic_norm = float(np.sum(weights * magnetic_norm))
    if not np.isfinite(integrated_magnetic_norm) or integrated_magnetic_norm <= 0.0:
        raise SolverError("The reconstructed magnetic field has zero or non-finite norm.")
    wave_impedance = complex(
        np.sum(weights * poynting_numerator) / integrated_magnetic_norm
    )
    if not np.isfinite((wave_impedance.real, wave_impedance.imag)).all():
        raise SolverError("The integrated wave impedance is non-finite.")

    local_wave_impedance = np.full(weights.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    local_threshold = (
        256.0
        * np.finfo(float).eps
        * max(float(np.max(magnetic_norm)), np.finfo(float).tiny)
    )
    valid = magnetic_norm > local_threshold
    local_wave_impedance[valid] = poynting_numerator[valid] / magnetic_norm[valid]

    # Templates are currently isotropic.  Retaining epsilon_x as the scalar
    # material sample keeps SampledFields.material shape-compatible while the
    # stiffness above remains fully transverse-diagonal for advanced callers.
    relative_epsilon = np.asarray(epsilon_r[0], dtype=np.complex128)
    return QuasiTEMSolution(
        electric_potential=potential,
        vacuum_potential=vacuum_potential,
        coordinates=coordinates,
        weights=weights,
        relative_epsilon=relative_epsilon,
        electric_field=dielectric_field,
        magnetic_field=magnetic_field,
        capacitance_per_length=capacitance,
        vacuum_capacitance_per_length=vacuum_capacitance,
        inductance_per_length=inductance,
        neff=neff,
        characteristic_impedance=characteristic_impedance,
        wave_impedance=wave_impedance,
        local_wave_impedance=local_wave_impedance,
        voltage=voltage,
        current=current,
        power=power,
        resistance_per_length=resistance,
        conductance_per_length=conductance,
        surface_resistance=surface_resistance,
        external_inductance_per_length=external_inductance,
        metadata={
            "formulation": "scalar-quasi-TEM-P1",
            "quadrature_order": order,
            "signal_boundaries": signal_names,
            "reference_boundaries": reference_names,
            "material_free_residual": material_residual,
            "vacuum_free_residual": vacuum_residual,
            "capacitance_from_energy": material_energy,
            "vacuum_capacitance_from_energy": vacuum_energy,
            "frequency": frequency_value,
            "metal_conductivity": metal_conductivity,
            "surface_resistance": surface_resistance,
            "surface_impedance": complex(surface_resistance, surface_resistance),
            "skin_depth": skin_depth,
            "conductor_geometry_factors": conductor_geometry_factors,
            "conductor_geometry_factor_per_length": total_geometry_factor,
            "conductor_projection_residuals": projection_residuals,
            "resistance_per_length": resistance,
            "conductance_per_length": conductance,
            "external_inductance_per_length": external_inductance,
            "internal_inductance_per_length": inductance - external_inductance,
            "series_impedance_per_length": series_impedance,
            "shunt_admittance_per_length": shunt_admittance,
            "conductor_loss_per_length": conductor_loss,
            "dielectric_loss_per_length": dielectric_loss,
            "dissipated_power_per_length": dissipated_power,
            "power_balance": {
                "circuit_forward_power": circuit_forward_power,
                "field_forward_power": power.real,
                "attenuation_power_decay": attenuation_power_decay,
                "relative_residual": power_balance_residual,
            },
            "time_convention": "exp(+1j*omega*t - 1j*beta*z)",
        },
    )


__all__ = ["QuasiTEMSolution", "solve_quasi_tem"]
