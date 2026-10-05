"""Immutable bulk materials and electromagnetic boundary assignments.

Constitutive values are relative and use exp(+i*omega*t). Metal presets use
reference resistivity in ohm metre, without temperature or roughness corrections.
"""
from dataclasses import dataclass
from math import pi, sqrt
import numpy as np

from .errors import ConfigurationError, BackendCapabilityError


def _value(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind in 'bOUS':
        raise ConfigurationError(f'{name} must contain finite numeric values.')
    a = np.asarray(value, dtype=complex)
    if a.shape not in ((), (1,), (2,), (3,), (1, 1), (2, 2), (3, 3)) or not np.isfinite(a).all():
        raise ConfigurationError(f'{name} must be a finite scalar, diagonal, or square tensor (dimension 1–3).')
    if a.ndim == 0:
        return complex(a)
    if a.ndim == 1:
        return tuple(complex(v) for v in a)
    return tuple(tuple(complex(v) for v in row) for row in a)


@dataclass(frozen=True, kw_only=True)
class Material:
    """Bulk material: relative epsilon/mu; scalar, diagonal, or tensor.

    Tensor components refer to physical Cartesian axes. A solver validates the
    supported dimension and tensor form without dropping unsupported components.
    """
    name: str = 'material'
    epsilon: object = 1.0
    mu: object = 1.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError('Material name must be a nonempty string.')
        object.__setattr__(self, 'epsilon', _value(self.epsilon, 'epsilon'))
        object.__setattr__(self, 'mu', _value(self.mu, 'mu'))

    @property
    def is_passive(self):
        for component in ('epsilon', 'mu'):
            raw = np.asarray(getattr(self, component))
            dimension = raw.shape[0] if raw.ndim else 3
            a = self.tensor(component, dimension=dimension)
            if np.linalg.eigvalsh(-(a-a.conj().T)/(2j)).min() < -1e-14:
                return False
        return True

    def tensor(self, component, *, dimension=3):
        if component not in ('epsilon', 'mu'):
            raise ConfigurationError("component must be 'epsilon' or 'mu'.")
        a = np.asarray(getattr(self, component), dtype=complex)
        if a.ndim == 0:
            return np.eye(dimension, dtype=complex)*a
        if a.shape == (dimension,):
            return np.diag(a)
        if a.shape == (dimension, dimension):
            return a.copy()
        raise BackendCapabilityError(f'{component} has shape {a.shape}; this solver requires dimension {dimension}.')


@dataclass(frozen=True, kw_only=True)
class IdealBoundary:
    """Ideal PEC/PMC assignment. Use the shared PEC and PMC presets."""
    name: str
    kind: str

    def __post_init__(self):
        if self.kind not in ('pec', 'pmc'):
            raise ConfigurationError('Ideal boundary kind must be pec or pmc.')


@dataclass(frozen=True, kw_only=True)
class SurfaceImpedance:
    """Constant scalar passive surface impedance in ohms."""
    name: str = 'surface impedance'
    impedance: complex

    def __post_init__(self):
        z = complex(self.impedance)
        if not np.isfinite(z) or z == 0 or z.real < 0:
            raise ConfigurationError('Surface impedance must be finite, nonzero, and have nonnegative real part.')
        object.__setattr__(self, 'impedance', z)

    def at_frequency(self, *, frequency):
        _positive(frequency, 'frequency')
        return self.impedance


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ConfigurationError(f'{name} must be finite and positive.')
    return float(value)


@dataclass(frozen=True, kw_only=True)
class GoodConductor:
    """Good-conductor SIBC with conductivity (S/m) and relative permeability.

    The conductor interior is excluded. This model assumes a thick, smooth good
    conductor; it is not a bulk or thin-film electromagnetic material model.
    """
    name: str
    conductivity: float
    mu: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'conductivity', _positive(self.conductivity, 'conductivity'))
        object.__setattr__(self, 'mu', _positive(self.mu, 'mu'))

    def at_frequency(self, *, frequency):
        f = _positive(frequency, 'frequency')
        return (1+1j)*sqrt(pi*f*(4e-7*pi)*self.mu/self.conductivity)


MaterialAssignment = Material | IdealBoundary | SurfaceImpedance | GoodConductor


@dataclass(frozen=True, kw_only=True)
class SpatialMaterial:
    """Named spatial permittivity callback for supported scattering backends.

    epsilon(x, z) returns relative scalar values; mu is currently fixed at one.
    Callbacks are evaluated during solving and are not serialized for restart.
    """
    name: str
    epsilon: object
    def __post_init__(self):
        if not callable(self.epsilon):
            raise ConfigurationError('SpatialMaterial.epsilon must be callable.')
PEC = IdealBoundary(name='PEC', kind='pec')
PMC = IdealBoundary(name='PMC', kind='pmc')
vacuum = Material(name='vacuum')
air = Material(name='air (vacuum approximation)')

# Preserve the existing project reference values when consolidating presets.
_RESISTIVITIES = {'aluminium': 2.650e-8, 'copper': 1.676e-8, 'gold': 2.192e-8,
                 'molybdenum': 5.340e-8, 'palladium': 1.054e-7, 'silver': 1.586e-8,
                 'tungsten': 5.280e-8, 'zinc': 5.964e-8}
for _name, _rho in _RESISTIVITIES.items():
    globals()[_name] = GoodConductor(name=_name, conductivity=1/_rho)


def bulk_values(material, *, form='diagonal', dimension=3):
    """Validate a backend's material capability at its implementation boundary."""
    if not isinstance(material, Material):
        raise ConfigurationError('A bulk Material object is required; define it before assigning geometry.')
    values = []
    for component in ('epsilon', 'mu'):
        a = material.tensor(component, dimension=dimension)
        if form == 'static':
            if np.any(a.imag) or not np.allclose(a.real, a.real.T) or np.linalg.eigvalsh(a.real).min() <= 0:
                raise BackendCapabilityError('Electrostatics requires real symmetric positive-definite material tensors.')
            values.append(a.real)
            continue
        diagonal = np.diag(a)
        if not np.array_equal(a, np.diag(diagonal)):
            raise BackendCapabilityError('This solver supports scalar/diagonal materials, not off-diagonal tensors.')
        if form == 'scalar':
            if not np.all(diagonal == diagonal[0]):
                raise BackendCapabilityError('This solver supports scalar isotropic materials only.')
            values.append(complex(diagonal[0]))
        else:
            values.append(tuple(diagonal))
    return tuple(values)


__all__ = ['Material', 'GoodConductor', 'SurfaceImpedance', 'SpatialMaterial', 'PEC', 'PMC', 'vacuum', 'air', *_RESISTIVITIES]
