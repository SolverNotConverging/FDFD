"""Immutable materials and boundary assignments.

Constitutive values are relative to vacuum and follow the ``exp(+i*omega*t)``
convention: passive loss has a negative imaginary part. Metal presets are
:class:`GoodConductor` surface-impedance models built from reference
resistivities in ohm metre, without temperature or roughness corrections.
"""

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from .constants import MU_0
from .errors import BackendCapabilityError, ConfigurationError


def _value(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind in "bOUS":
        raise ConfigurationError(f"{name} must contain finite numeric values.")
    a = np.asarray(value, dtype=complex)
    if a.shape not in ((), (1,), (2,), (3,), (1, 1), (2, 2), (3, 3)) or not np.isfinite(a).all():
        raise ConfigurationError(
            f"{name} must be a finite scalar, diagonal, or square tensor (dimension 1-3)."
        )
    if a.ndim == 0:
        return complex(a)
    if a.ndim == 1:
        return tuple(complex(v) for v in a)
    return tuple(tuple(complex(v) for v in row) for row in a)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be finite and positive.")
    return float(value)


@dataclass(frozen=True, kw_only=True)
class Material:
    """A bulk medium with relative permittivity and permeability.

    Parameters
    ----------
    name : str
        Label used in plots, saved results and error messages.
    epsilon, mu : complex or sequence
        Relative permittivity and permeability: a scalar, a diagonal
        ``(xx, yy, zz)`` tuple, or a square tensor. Components refer to the
        physical Cartesian axes. Each solver checks which forms it supports.
    """

    name: str = "material"
    epsilon: object = 1.0
    mu: object = 1.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("Material name must be a nonempty string.")
        object.__setattr__(self, "epsilon", _value(self.epsilon, "epsilon"))
        object.__setattr__(self, "mu", _value(self.mu, "mu"))

    @property
    def is_passive(self):
        """True when neither tensor has gain under ``exp(+i*omega*t)``."""
        for component in ("epsilon", "mu"):
            raw = np.asarray(getattr(self, component))
            dimension = raw.shape[0] if raw.ndim else 3
            a = self.tensor(component, dimension=dimension)
            if np.linalg.eigvalsh(-(a - a.conj().T) / (2j)).min() < -1e-14:
                return False
        return True

    def tensor(self, component, *, dimension=3):
        """Return ``epsilon`` or ``mu`` as a ``dimension`` x ``dimension`` array."""
        if component not in ("epsilon", "mu"):
            raise ConfigurationError("component must be 'epsilon' or 'mu'.")
        a = np.asarray(getattr(self, component), dtype=complex)
        if a.ndim == 0:
            return np.eye(dimension, dtype=complex) * a
        if a.shape == (dimension,):
            return np.diag(a)
        if a.shape == (dimension, dimension):
            return a.copy()
        raise BackendCapabilityError(
            f"{component} has shape {a.shape}; this solver requires dimension {dimension}."
        )


@dataclass(frozen=True, kw_only=True)
class IdealBoundary:
    """Perfect electric or magnetic conductor. Use the :data:`PEC` and :data:`PMC` presets."""

    name: str
    kind: str

    def __post_init__(self):
        if self.kind not in ("pec", "pmc"):
            raise ConfigurationError("Ideal boundary kind must be pec or pmc.")


@dataclass(frozen=True, kw_only=True)
class SurfaceImpedance:
    """Opaque region whose surface obeys a constant scalar impedance (ohms)."""

    name: str = "surface impedance"
    impedance: complex

    def __post_init__(self):
        z = complex(self.impedance)
        if not np.isfinite(z) or z == 0 or z.real < 0:
            raise ConfigurationError(
                "Surface impedance must be finite, nonzero, and have nonnegative real part."
            )
        object.__setattr__(self, "impedance", z)

    def at_frequency(self, *, frequency):
        """Surface impedance in ohms at ``frequency`` (Hz); constant here."""
        _positive(frequency, "frequency")
        return self.impedance


@dataclass(frozen=True, kw_only=True)
class GoodConductor:
    """Thick, smooth good conductor modelled by its surface impedance.

    ``Zs = (1 + i) * sqrt(pi * f * mu0 * mu / conductivity)``. The conductor
    interior is excluded from the field solution; this is not a bulk or
    thin-film model.

    Parameters
    ----------
    conductivity : float
        Conductivity in S/m.
    mu : float
        Relative permeability of the metal.
    """

    name: str
    conductivity: float
    mu: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "conductivity", _positive(self.conductivity, "conductivity"))
        object.__setattr__(self, "mu", _positive(self.mu, "mu"))

    def at_frequency(self, *, frequency):
        """Surface impedance in ohms at ``frequency`` (Hz)."""
        f = _positive(frequency, "frequency")
        return (1 + 1j) * sqrt(pi * f * MU_0 * self.mu / self.conductivity)


MaterialAssignment = Material | IdealBoundary | SurfaceImpedance | GoodConductor

PEC = IdealBoundary(name="PEC", kind="pec")
PMC = IdealBoundary(name="PMC", kind="pmc")
vacuum = Material(name="vacuum")
air = Material(name="air (vacuum approximation)")

RESISTIVITIES = {
    "aluminium": 2.650e-8,
    "copper": 1.676e-8,
    "gold": 2.192e-8,
    "molybdenum": 5.340e-8,
    "palladium": 1.054e-7,
    "silver": 1.586e-8,
    "tungsten": 5.280e-8,
    "zinc": 5.964e-8,
}
"""Reference resistivities (ohm metre) behind the metal presets."""

aluminium = GoodConductor(name="aluminium", conductivity=1 / RESISTIVITIES["aluminium"])
copper = GoodConductor(name="copper", conductivity=1 / RESISTIVITIES["copper"])
gold = GoodConductor(name="gold", conductivity=1 / RESISTIVITIES["gold"])
molybdenum = GoodConductor(name="molybdenum", conductivity=1 / RESISTIVITIES["molybdenum"])
palladium = GoodConductor(name="palladium", conductivity=1 / RESISTIVITIES["palladium"])
silver = GoodConductor(name="silver", conductivity=1 / RESISTIVITIES["silver"])
tungsten = GoodConductor(name="tungsten", conductivity=1 / RESISTIVITIES["tungsten"])
zinc = GoodConductor(name="zinc", conductivity=1 / RESISTIVITIES["zinc"])


def bulk_values(material, *, form="diagonal", dimension=3):
    """Return ``(epsilon, mu)`` in the form a kernel supports.

    ``form="diagonal"`` gives two length-3 tuples; ``form="scalar"`` gives two
    complex numbers. Unsupported tensors raise :class:`BackendCapabilityError`.
    """
    if not isinstance(material, Material):
        raise ConfigurationError("A bulk Material object is required; define it before assigning geometry.")
    values = []
    for component in ("epsilon", "mu"):
        a = material.tensor(component, dimension=dimension)
        diagonal = np.diag(a)
        if not np.array_equal(a, np.diag(diagonal)):
            raise BackendCapabilityError(
                "This solver supports scalar/diagonal materials, not off-diagonal tensors."
            )
        if form == "scalar":
            if not np.all(diagonal == diagonal[0]):
                raise BackendCapabilityError("This solver supports scalar isotropic materials only.")
            values.append(complex(diagonal[0]))
        else:
            values.append(tuple(diagonal))
    return tuple(values)


def max_index(media):
    """Largest ``|sqrt(epsilon_i * mu_j)|`` over the principal axes of ``media``.

    Crossed principal pairs are included, which bounds the effective index of
    any guided mode in diagonal anisotropic media.
    """
    bound = 0.0
    for material in media:
        epsilon, mu = bulk_values(material)
        products = np.asarray(epsilon).reshape(-1, 1) * np.asarray(mu).reshape(1, -1)
        bound = max(bound, float(np.sqrt(np.abs(products)).max()))
    return bound


__all__ = [
    "PEC",
    "PMC",
    "RESISTIVITIES",
    "GoodConductor",
    "Material",
    "SurfaceImpedance",
    "air",
    "aluminium",
    "copper",
    "gold",
    "molybdenum",
    "palladium",
    "silver",
    "tungsten",
    "vacuum",
    "zinc",
]
