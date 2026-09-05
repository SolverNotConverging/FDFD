r"""Shared metal presets for the good-conductor surface-impedance model.

The impedance sign follows the :math:`\exp(+j\omega t)` phasor convention
used by the FDFD solvers. Resistivities are reference-condition values in
ohm metre; callers are responsible for any temperature correction.
"""

from math import isfinite, pi, sqrt
from numbers import Real
from types import MappingProxyType
from typing import Mapping


MU_0_H_PER_M = 4e-7 * pi

METAL_RESISTIVITIES_OHM_M: Mapping[str, float] = MappingProxyType(
    {
        "aluminium": 2.650e-8,
        "copper": 1.676e-8,
        "gold": 2.192e-8,
        "molybdenum": 5.340e-8,
        "palladium": 1.054e-7,
        "silver": 1.586e-8,
        "tungsten": 5.280e-8,
        "zinc": 5.964e-8,
    }
)

_METAL_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "ag": "silver",
        "al": "aluminium",
        "aluminium": "aluminium",
        "aluminum": "aluminium",
        "au": "gold",
        "copper": "copper",
        "cu": "copper",
        "gold": "gold",
        "mo": "molybdenum",
        "molybdenum": "molybdenum",
        "palladium": "palladium",
        "pd": "palladium",
        "silver": "silver",
        "tungsten": "tungsten",
        "w": "tungsten",
        "zinc": "zinc",
        "zn": "zinc",
    }
)


def canonical_metal_name(metal: str) -> str:
    """Return the canonical lower-case full name for a metal preset."""
    if not isinstance(metal, str):
        raise TypeError("metal must be a name or chemical-symbol string.")

    key = metal.strip().casefold()
    try:
        return _METAL_ALIASES[key]
    except KeyError:
        supported = ", ".join(METAL_RESISTIVITIES_OHM_M)
        raise ValueError(
            f"Unknown metal {metal!r}; supported metals: {supported}."
        ) from None


def metal_resistivity(metal: str) -> float:
    """Return the preset resistivity in ohm metre."""
    return METAL_RESISTIVITIES_OHM_M[canonical_metal_name(metal)]


def metal_conductivity(metal: str) -> float:
    """Return conductivity in siemens per metre as the resistivity reciprocal."""
    return 1.0 / metal_resistivity(metal)


def _positive_finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def good_conductor_surface_impedance(
        metal: str,
        frequency_hz: Real,
        *,
        relative_permeability: Real = 1.0,
) -> complex:
    r"""Return the good-conductor surface impedance at one frequency.

    For the :math:`\exp(+j\omega t)` convention this uses

    .. math::

       Z_s = (1 + j)\sqrt{\pi f \mu_0 \mu_r \rho}.

    Parameters
    ----------
    metal:
        Full preset name or chemical symbol, matched case-insensitively.
    frequency_hz:
        Frequency in hertz. Must be finite and positive.
    relative_permeability:
        Relative permeability used in the good-conductor approximation.
        Must be finite and positive.
    """
    frequency_hz = _positive_finite_real("frequency_hz", frequency_hz)
    relative_permeability = _positive_finite_real(
        "relative_permeability", relative_permeability
    )
    surface_resistance = sqrt(
        pi
        * frequency_hz
        * MU_0_H_PER_M
        * relative_permeability
        * metal_resistivity(metal)
    )
    return complex(surface_resistance, surface_resistance)


__all__ = [
    "METAL_RESISTIVITIES_OHM_M",
    "MU_0_H_PER_M",
    "canonical_metal_name",
    "good_conductor_surface_impedance",
    "metal_conductivity",
    "metal_resistivity",
]
