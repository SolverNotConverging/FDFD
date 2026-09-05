"""Guided incident fields in a straight waveguide lead.

The phasor convention is ``exp(-i*beta*z-i*ky*y+i*omega*t)``.  A launch from
the left therefore uses a +z mode, while a launch from the right uses its
z-mirrored -z counterpart.  ``IncidentMode`` can be passed directly wherever
an electric incident-field callback ``incident(x, z)`` is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import ConfigurationError
from .modes import ComplexArray, Mode


IncidentSide: TypeAlias = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class IncidentMode:
    """A normalized lead mode launched from one reference plane.

    Parameters
    ----------
    mode:
        A mode of the unperturbed straight lead.  If its direction disagrees
        with ``side``, its exact z-mirrored counterpart is selected.
    side:
        ``"left"`` launches toward +z and ``"right"`` launches toward -z.
    reference_plane:
        Physical z coordinate in metres at which ``amplitude`` is defined.
    amplitude:
        Complex field amplitude.  For a unit-power propagating mode, the
        incident power magnitude is ``abs(amplitude)**2`` W per invariant-y
        length.
    """

    mode: Mode
    side: IncidentSide = "left"
    reference_plane: float = 0.0
    amplitude: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        if not isinstance(self.mode, Mode):
            raise ConfigurationError("mode must be a fem_waveguide_scattering.modes.Mode instance.")
        if self.side not in ("left", "right"):
            raise ConfigurationError("side must be 'left' or 'right'.")

        try:
            reference = float(self.reference_plane)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("reference_plane must be a finite real number.") from exc
        if not np.isfinite(reference):
            raise ConfigurationError("reference_plane must be a finite real number.")

        amplitude_array = np.asarray(self.amplitude)
        if amplitude_array.shape != ():
            raise ConfigurationError("amplitude must be a finite complex scalar.")
        try:
            amplitude = complex(amplitude_array.item())
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("amplitude must be a finite complex scalar.") from exc
        if not np.isfinite((amplitude.real, amplitude.imag)).all():
            raise ConfigurationError("amplitude must be a finite complex scalar.")

        desired = (
            ("forward", "right-decaying")
            if self.side == "left"
            else ("backward", "left-decaying")
        )
        if self.mode.direction == "indeterminate":
            raise ConfigurationError(
                "Cannot launch a mode with indeterminate propagation direction."
            )
        launched = self.mode
        if launched.direction not in desired:
            launched = launched.counterpropagating()
        if launched.direction not in desired:
            raise ConfigurationError(
                f"Could not orient mode direction {self.mode.direction!r} for a "
                f"{self.side!r}-side launch."
            )

        object.__setattr__(self, "mode", launched)
        object.__setattr__(self, "reference_plane", reference)
        object.__setattr__(self, "amplitude", amplitude)

    @property
    def direction(self) -> str:
        """Direction classification of the actually launched mode."""

        return self.mode.direction

    @property
    def beta(self) -> complex:
        """Propagation constant, including the launch-direction sign."""

        return self.mode.beta

    @property
    def signed_power(self) -> float:
        """Signed longitudinal real power after amplitude scaling."""

        return float(abs(self.amplitude) ** 2 * self.mode.power)

    def fields(
        self, x: ArrayLike, z: ArrayLike
    ) -> tuple[ComplexArray, ComplexArray]:
        """Evaluate amplitude-scaled ``(E, H)`` anywhere in the straight lead."""

        electric, magnetic = self.mode.fields(
            x, z, reference_plane=self.reference_plane
        )
        electric = np.asarray(self.amplitude * electric, dtype=np.complex128)
        magnetic = np.asarray(self.amplitude * magnetic, dtype=np.complex128)
        if not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
            raise ValueError("incident fields overflowed at the requested coordinates.")
        return electric, magnetic

    def E(self, x: ArrayLike, z: ArrayLike) -> ComplexArray:
        """Evaluate the incident electric field in ``(x, y, z)`` order."""

        return self.fields(x, z)[0]

    def H(self, x: ArrayLike, z: ArrayLike) -> ComplexArray:
        """Evaluate the incident magnetic field in ``(x, y, z)`` order."""

        return self.fields(x, z)[1]

    def __call__(self, x: ArrayLike, z: ArrayLike) -> ComplexArray:
        """Alias for :meth:`E`, compatible with equivalent-source assembly."""

        return self.E(x, z)


__all__ = ["IncidentMode", "IncidentSide"]
