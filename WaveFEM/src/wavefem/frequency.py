"""Validated single-frequency configuration in SI units."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tau
from numbers import Real
from typing import TypeAlias

from .constants import C0
from .exceptions import ConfigurationError

RealInput: TypeAlias = Real


def _positive_finite_real(name: str, value: RealInput) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a real scalar in SI units.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"{name} must be finite and strictly positive.")
    return result


@dataclass(frozen=True, slots=True)
class Frequency:
    """One canonical angular frequency and its derived SI quantities.

    Constructing this class directly interprets ``omega`` in radians per
    second.  The named constructors and :func:`resolve_frequency` make the
    input quantity explicit at public API boundaries.
    """

    omega: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "omega",
            _positive_finite_real("omega", self.omega),
        )

    @classmethod
    def from_wavelength(cls, wavelength: RealInput) -> "Frequency":
        """Create a spectral point from vacuum wavelength in metres."""
        wavelength_m = _positive_finite_real("wavelength", wavelength)
        return cls(omega=tau * C0 / wavelength_m)

    @classmethod
    def from_frequency(cls, frequency: RealInput) -> "Frequency":
        """Create a spectral point from ordinary frequency in hertz."""
        frequency_hz = _positive_finite_real("frequency", frequency)
        return cls(omega=tau * frequency_hz)

    @classmethod
    def from_omega(cls, omega: RealInput) -> "Frequency":
        """Create a spectral point from angular frequency in radians/second."""
        return cls(omega=omega)

    @property
    def angular_frequency(self) -> float:
        """Angular frequency in radians per second."""
        return self.omega

    @property
    def frequency(self) -> float:
        """Ordinary frequency in hertz."""
        return self.omega / tau

    @property
    def wavelength(self) -> float:
        """Vacuum wavelength in metres."""
        return tau * C0 / self.omega

    @property
    def k0(self) -> float:
        """Vacuum angular wavenumber in radians per metre."""
        return self.omega / C0


def resolve_frequency(
    *,
    frequency: RealInput | None = None,
    omega: RealInput | None = None,
    wavelength: RealInput | None = None,
) -> Frequency:
    """Resolve exactly one independent public frequency specification.

    Parameters use SI units: metres, hertz, and radians per second.  Supplying
    none or more than one is rejected even when the values are numerically
    consistent, preventing ambiguous precedence rules in simulation APIs.
    """

    supplied = [
        ("frequency", frequency),
        ("omega", omega),
        ("wavelength", wavelength),
    ]
    present = [(name, value) for name, value in supplied if value is not None]
    if len(present) != 1:
        names = ", ".join(name for name, _ in present) or "none"
        raise ConfigurationError(
            "Specify exactly one of frequency, omega, or wavelength; "
            f"received {names}."
        )

    name, value = present[0]
    if name == "wavelength":
        return Frequency.from_wavelength(value)
    if name == "frequency":
        return Frequency.from_frequency(value)
    return Frequency.from_omega(value)


__all__ = ["Frequency", "resolve_frequency"]
