"""Validated, frequency-independent transmission-line specifications.

All dimensions are expressed in metres and optional bulk metal conductivity
is expressed in S/m.  The relative permittivity follows the mode solver's
``exp(+j*omega*t)`` convention, so a passive dielectric is represented by
``epsilon_r * (1 - 1j * loss_tangent)``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any, ClassVar, TypeAlias

import numpy as np

from ..exceptions import ConfigurationError


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ConfigurationError(f"{name} must be a finite positive SI value.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(
            f"{name} must be a finite positive SI value."
        ) from exc
    if not isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"{name} must be a finite positive SI value.")
    return result


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ConfigurationError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be finite and nonnegative.") from exc
    if not isfinite(result) or result < 0.0:
        raise ConfigurationError(f"{name} must be finite and nonnegative.")
    return result


def _set_positive(instance: object, *names: str) -> None:
    for name in names:
        object.__setattr__(instance, name, _positive(getattr(instance, name), name))


def _set_dielectric(instance: object) -> None:
    object.__setattr__(
        instance,
        "epsilon_r",
        _positive(getattr(instance, "epsilon_r"), "epsilon_r"),
    )
    object.__setattr__(
        instance,
        "loss_tangent",
        _nonnegative(getattr(instance, "loss_tangent"), "loss_tangent"),
    )


def _set_metal_conductivity(instance: object) -> None:
    conductivity = getattr(instance, "metal_conductivity")
    if conductivity is not None:
        conductivity = _positive(conductivity, "metal_conductivity")
    object.__setattr__(instance, "metal_conductivity", conductivity)


def _complex_epsilon(epsilon_r: float, loss_tangent: float) -> complex:
    return complex(epsilon_r * (1.0 - 1j * loss_tangent))


@dataclass(frozen=True, slots=True)
class Coaxial:
    """Circular coaxial line.

    ``outer_radius`` is the inner radius of the outer conductor.  The finite
    ``outer_conductor_thickness`` is used to construct an exact annular PEC
    hole, but does not otherwise enter the ideal coax dimensions.
    """

    kind: ClassVar[str] = "coaxial"

    inner_radius: float = 0.50e-3
    outer_radius: float = 1.67e-3
    outer_conductor_thickness: float = 0.15e-3
    epsilon_r: float = 2.10
    loss_tangent: float = 2.0e-4
    metal_conductivity: float | None = None

    def __post_init__(self) -> None:
        _set_positive(
            self,
            "inner_radius",
            "outer_radius",
            "outer_conductor_thickness",
        )
        _set_dielectric(self)
        _set_metal_conductivity(self)
        if self.outer_radius <= self.inner_radius:
            raise ConfigurationError(
                "outer_radius must be greater than inner_radius."
            )

    @property
    def complex_epsilon(self) -> complex:
        return _complex_epsilon(self.epsilon_r, self.loss_tangent)

    @property
    def epsilon(self) -> complex:
        """Alias for :attr:`complex_epsilon`."""

        return self.complex_epsilon


@dataclass(frozen=True, slots=True)
class Microstrip:
    """Microstrip over a finite-height dielectric substrate.

    ``domain_padding_factor`` scales the default clearance to the remote PEC
    truncation wall.  Increase it above one to verify open-domain convergence.
    """

    kind: ClassVar[str] = "microstrip"

    trace_width: float = 3.00e-3
    substrate_height: float = 1.524e-3
    conductor_thickness: float = 35.0e-6
    epsilon_r: float = 3.55
    loss_tangent: float = 2.7e-3
    domain_padding_factor: float = 1.0
    metal_conductivity: float | None = None

    def __post_init__(self) -> None:
        _set_positive(
            self,
            "trace_width",
            "substrate_height",
            "conductor_thickness",
            "domain_padding_factor",
        )
        _set_dielectric(self)
        _set_metal_conductivity(self)

    @property
    def complex_epsilon(self) -> complex:
        return _complex_epsilon(self.epsilon_r, self.loss_tangent)

    @property
    def epsilon(self) -> complex:
        """Alias for :attr:`complex_epsilon`."""

        return self.complex_epsilon


@dataclass(frozen=True, slots=True)
class Stripline:
    """Symmetric stripline between tied upper and lower ground planes.

    ``ground_spacing`` is the distance between the two dielectric-facing
    ground-plane surfaces.  The signal conductor is centred between them.
    """

    kind: ClassVar[str] = "stripline"

    trace_width: float = 0.80e-3
    ground_spacing: float = 1.524e-3
    conductor_thickness: float = 35.0e-6
    epsilon_r: float = 3.55
    loss_tangent: float = 2.7e-3
    domain_padding_factor: float = 1.0
    metal_conductivity: float | None = None

    def __post_init__(self) -> None:
        _set_positive(
            self,
            "trace_width",
            "ground_spacing",
            "conductor_thickness",
            "domain_padding_factor",
        )
        _set_dielectric(self)
        _set_metal_conductivity(self)
        if self.conductor_thickness >= self.ground_spacing:
            raise ConfigurationError(
                "conductor_thickness must be smaller than ground_spacing."
            )

    @property
    def complex_epsilon(self) -> complex:
        return _complex_epsilon(self.epsilon_r, self.loss_tangent)

    @property
    def epsilon(self) -> complex:
        """Alias for :attr:`complex_epsilon`."""

        return self.complex_epsilon


@dataclass(frozen=True, slots=True)
class CoplanarWaveguide:
    """Conventional CPW: centre signal against tied left/right grounds.

    ``domain_padding_factor`` scales the default clearance to the remote PEC
    truncation wall.  Increase it above one to verify open-domain convergence.
    """

    kind: ClassVar[str] = "coplanar_waveguide"

    center_width: float = 0.60e-3
    gap: float = 0.25e-3
    ground_width: float = 1.50e-3
    substrate_height: float = 0.80e-3
    conductor_thickness: float = 35.0e-6
    epsilon_r: float = 3.55
    loss_tangent: float = 2.7e-3
    domain_padding_factor: float = 1.0
    metal_conductivity: float | None = None

    def __post_init__(self) -> None:
        _set_positive(
            self,
            "center_width",
            "gap",
            "ground_width",
            "substrate_height",
            "conductor_thickness",
            "domain_padding_factor",
        )
        _set_dielectric(self)
        _set_metal_conductivity(self)

    @property
    def complex_epsilon(self) -> complex:
        return _complex_epsilon(self.epsilon_r, self.loss_tangent)

    @property
    def epsilon(self) -> complex:
        """Alias for :attr:`complex_epsilon`."""

        return self.complex_epsilon


TransmissionLineSpec: TypeAlias = Coaxial | Microstrip | Stripline | CoplanarWaveguide


def _identifier(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


_SPEC_CLASSES: dict[str, type[TransmissionLineSpec]] = {
    "coaxial": Coaxial,
    "microstrip": Microstrip,
    "stripline": Stripline,
    "coplanarwaveguide": CoplanarWaveguide,
}

_KIND_ALIASES = {
    "coax": "coaxial",
    "coaxialline": "coaxial",
    "microstripline": "microstrip",
    "stripline": "stripline",
    "striptransmissionline": "stripline",
    "cpw": "coplanarwaveguide",
    "cpwodd": "coplanarwaveguide",
    "oddcpw": "coplanarwaveguide",
    "cpwoddsignaltotiedgrounds": "coplanarwaveguide",
    "coplanar": "coplanarwaveguide",
    "coplanarline": "coplanarwaveguide",
    "coplanarwaveguideodd": "coplanarwaveguide",
    "coplanarwaveguideoddsignaltotiedgrounds": "coplanarwaveguide",
}

_COMMON_PARAMETER_ALIASES = {
    "er": "epsilon_r",
    "epsr": "epsilon_r",
    "epsilon": "epsilon_r",
    "permittivity": "epsilon_r",
    "relativepermittivity": "epsilon_r",
    "substrateepsilon": "epsilon_r",
    "substratepermittivity": "epsilon_r",
    "dielectricconstant": "epsilon_r",
    "tand": "loss_tangent",
    "tandelta": "loss_tangent",
    "losstan": "loss_tangent",
    "dielectriclosstangent": "loss_tangent",
    "conductivity": "metal_conductivity",
    "conductorconductivity": "metal_conductivity",
    "bulkconductivity": "metal_conductivity",
    "sigma": "metal_conductivity",
    "metalsigma": "metal_conductivity",
    "conductorsigma": "metal_conductivity",
    "metalconductance": "metal_conductivity",
}

_PARAMETER_ALIASES: dict[type[TransmissionLineSpec], dict[str, str]] = {
    Coaxial: {
        "innerradius": "inner_radius",
        "innerconductorradius": "inner_radius",
        "signalradius": "inner_radius",
        "outerradius": "outer_radius",
        "outerinnerradius": "outer_radius",
        "outerconductorradius": "outer_radius",
        "shieldinnerradius": "outer_radius",
        "outerconductorthickness": "outer_conductor_thickness",
        "shieldthickness": "outer_conductor_thickness",
        "conductorthickness": "outer_conductor_thickness",
        "thickness": "outer_conductor_thickness",
    },
    Microstrip: {
        "width": "trace_width",
        "tracewidth": "trace_width",
        "stripwidth": "trace_width",
        "signalwidth": "trace_width",
        "height": "substrate_height",
        "substrateheight": "substrate_height",
        "substratethickness": "substrate_height",
        "dielectricheight": "substrate_height",
        "thickness": "conductor_thickness",
        "conductorthickness": "conductor_thickness",
        "metalthickness": "conductor_thickness",
        "tracethickness": "conductor_thickness",
        "padding": "domain_padding_factor",
        "paddingfactor": "domain_padding_factor",
        "domainpadding": "domain_padding_factor",
        "domainpaddingfactor": "domain_padding_factor",
        "enclosurescale": "domain_padding_factor",
    },
    Stripline: {
        "width": "trace_width",
        "tracewidth": "trace_width",
        "stripwidth": "trace_width",
        "signalwidth": "trace_width",
        "height": "ground_spacing",
        "groundspacing": "ground_spacing",
        "platespacing": "ground_spacing",
        "dielectricheight": "ground_spacing",
        "substrateheight": "ground_spacing",
        "thickness": "conductor_thickness",
        "conductorthickness": "conductor_thickness",
        "metalthickness": "conductor_thickness",
        "tracethickness": "conductor_thickness",
        "padding": "domain_padding_factor",
        "paddingfactor": "domain_padding_factor",
        "domainpadding": "domain_padding_factor",
        "domainpaddingfactor": "domain_padding_factor",
        "enclosurescale": "domain_padding_factor",
    },
    CoplanarWaveguide: {
        "width": "center_width",
        "centerwidth": "center_width",
        "centrewidth": "center_width",
        "tracewidth": "center_width",
        "signalwidth": "center_width",
        "centerconductorwidth": "center_width",
        "centreconductorwidth": "center_width",
        "gap": "gap",
        "slot": "gap",
        "slotwidth": "gap",
        "groundwidth": "ground_width",
        "groundstripwidth": "ground_width",
        "height": "substrate_height",
        "substrateheight": "substrate_height",
        "substratethickness": "substrate_height",
        "dielectricheight": "substrate_height",
        "thickness": "conductor_thickness",
        "conductorthickness": "conductor_thickness",
        "metalthickness": "conductor_thickness",
        "padding": "domain_padding_factor",
        "paddingfactor": "domain_padding_factor",
        "domainpadding": "domain_padding_factor",
        "domainpaddingfactor": "domain_padding_factor",
        "enclosurescale": "domain_padding_factor",
    },
}


def spec_from_type(kind: str, **params: Any) -> TransmissionLineSpec:
    """Construct a specification from a GUI-friendly line name and keywords.

    Names are case-insensitive and ignore spaces, underscores, and hyphens.
    Common microwave spellings such as ``cpw``, ``width``, ``eps_r``,
    ``substrate_epsilon``, ``tan_delta``, and ``metal_sigma`` are accepted.
    """

    if not isinstance(kind, str) or not kind.strip():
        raise ConfigurationError("transmission-line kind must be a nonempty string.")
    requested = _identifier(kind)
    canonical_kind = _KIND_ALIASES.get(requested, requested)
    try:
        spec_class = _SPEC_CLASSES[canonical_kind]
    except KeyError:
        choices = ", ".join(item.kind for item in _SPEC_CLASSES.values())
        raise ConfigurationError(
            f"Unknown transmission-line kind {kind!r}; choose one of {choices}."
        ) from None

    canonical_fields = {
        _identifier(field.name): field.name for field in fields(spec_class)
    }
    aliases = {**_COMMON_PARAMETER_ALIASES, **_PARAMETER_ALIASES[spec_class]}
    normalized: dict[str, Any] = {}
    for supplied_name, value in params.items():
        compact = _identifier(supplied_name)
        canonical_name = aliases.get(compact, canonical_fields.get(compact))
        if canonical_name is None:
            allowed = ", ".join(field.name for field in fields(spec_class))
            raise ConfigurationError(
                f"Unknown {spec_class.kind} parameter {supplied_name!r}; "
                f"expected one of {allowed}."
            )
        if canonical_name in normalized:
            raise ConfigurationError(
                f"Parameter {canonical_name!r} was supplied more than once."
            )
        normalized[canonical_name] = value
    return spec_class(**normalized)


__all__ = [
    "Coaxial",
    "CoplanarWaveguide",
    "Microstrip",
    "Stripline",
    "TransmissionLineSpec",
    "spec_from_type",
]
