"""Surface-impedance utilities for FEM boundary facets."""

from __future__ import annotations

from math import pi, sqrt
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .constants import MU_0
from .exceptions import ConfigurationError


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

_ALIASES = {
    "ag": "silver",
    "al": "aluminium",
    "aluminum": "aluminium",
    "aluminium": "aluminium",
    "au": "gold",
    "cu": "copper",
    "mo": "molybdenum",
    "pd": "palladium",
    "w": "tungsten",
    "zn": "zinc",
    **{name: name for name in METAL_RESISTIVITIES_OHM_M},
}


def canonical_metal_name(value: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError("metal preset must be a string.")
    try:
        return _ALIASES[value.strip().casefold()]
    except KeyError:
        supported = ", ".join(METAL_RESISTIVITIES_OHM_M)
        raise ConfigurationError(f"Unknown metal {value!r}; supported metals: {supported}.") from None


def validate_surface_impedance(value: complex | float) -> complex:
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ConfigurationError("surface impedance must be a scalar in ohms.")
    try:
        impedance = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("surface impedance must be a scalar in ohms.") from exc
    if not np.isfinite(impedance) or impedance == 0.0:
        raise ConfigurationError("surface impedance must be finite and nonzero.")
    if impedance.real < 0.0:
        raise ConfigurationError("passive surface impedance must have a nonnegative real part.")
    return impedance


def good_conductor_surface_impedance(
    metal: str,
    frequency: float,
    *,
    relative_permeability: float = 1.0,
) -> complex:
    """Return ``(1+i)*sqrt(pi*f*mu0*mu_r*rho)`` for ``exp(+iwt)``."""

    if isinstance(frequency, (bool, np.bool_)):
        raise ConfigurationError("frequency must be finite and positive.")
    if isinstance(relative_permeability, (bool, np.bool_)):
        raise ConfigurationError("relative_permeability must be finite and positive.")
    frequency = float(frequency)
    relative_permeability = float(relative_permeability)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ConfigurationError("frequency must be finite and positive.")
    if not np.isfinite(relative_permeability) or relative_permeability <= 0.0:
        raise ConfigurationError("relative_permeability must be finite and positive.")
    name = canonical_metal_name(metal)
    resistance = sqrt(pi * frequency * MU_0 * relative_permeability * METAL_RESISTIVITIES_OHM_M[name])
    return complex(resistance, resistance)


__all__ = [
    "METAL_RESISTIVITIES_OHM_M",
    "canonical_metal_name",
    "good_conductor_surface_impedance",
    "validate_surface_impedance",
]
