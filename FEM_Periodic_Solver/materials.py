"""Complex isotropic and diagonal material tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .exceptions import ConfigurationError

Scalar: TypeAlias = float | complex | np.number
MaterialInput: TypeAlias = Scalar | tuple[Scalar, Scalar, Scalar] | list[Scalar] | np.ndarray


def diagonal_values(value: MaterialInput, name: str) -> tuple[complex, complex, complex]:
    """Return a scalar or three entries as the physical ``xx, yy, zz`` diagonal."""

    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise ConfigurationError(f"{name} must be a scalar or three diagonal entries.")
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), 3)
    elif array.ndim != 1 or array.size != 3:
        raise ConfigurationError(f"{name} must be a scalar or length-three sequence.")
    if not np.isfinite(array).all():
        raise ConfigurationError(f"{name} contains a non-finite value.")
    return tuple(complex(item) for item in array)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class Material:
    """Relative diagonal permittivity and permeability.

    Complex values follow the package's ``exp(+1j*omega*t)`` convention.
    """

    eps_r: tuple[complex, complex, complex]
    mu_r: tuple[complex, complex, complex]

    def __init__(self, epsilon: MaterialInput = 1.0, mu: MaterialInput = 1.0) -> None:
        epsilon_values = diagonal_values(epsilon, "epsilon")
        permeability_values = diagonal_values(mu, "mu")
        if any(abs(value) == 0.0 for value in permeability_values):
            raise ConfigurationError("mu entries must be nonzero.")
        if any(abs(value) == 0.0 for value in epsilon_values):
            raise ConfigurationError("epsilon entries must be nonzero.")
        object.__setattr__(self, "eps_r", epsilon_values)
        object.__setattr__(self, "mu_r", permeability_values)

    @property
    def isotropic(self) -> bool:
        return (
            self.eps_r[0] == self.eps_r[1] == self.eps_r[2]
            and self.mu_r[0] == self.mu_r[1] == self.mu_r[2]
        )

    def eps_array(self) -> np.ndarray:
        return np.asarray(self.eps_r, dtype=np.complex128)

    def mu_array(self) -> np.ndarray:
        return np.asarray(self.mu_r, dtype=np.complex128)


__all__ = ["Material", "MaterialInput", "diagonal_values"]
