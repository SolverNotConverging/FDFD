"""Backend-neutral diagonal materials for FEM mode calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .exceptions import ConfigurationError


Scalar: TypeAlias = float | complex | np.number
MaterialInput: TypeAlias = Scalar | tuple[Scalar, Scalar, Scalar] | list[Scalar] | np.ndarray


def diagonal_values(value: MaterialInput, name: str) -> tuple[complex, complex, complex]:
    """Normalize a scalar or three diagonal entries in physical x/y/z order."""

    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise ConfigurationError(f"{name} must be a scalar or three diagonal entries.")
    values = np.asarray(value, dtype=np.complex128)
    if values.ndim == 0:
        result = np.repeat(values.reshape(1), 3)
    elif values.ndim == 1 and values.size == 3:
        result = values
    else:
        raise ConfigurationError(f"{name} must be a scalar or length-three sequence (xx, yy, zz).")
    if not np.isfinite(result).all():
        raise ConfigurationError(f"{name} contains a non-finite value.")
    return tuple(complex(item) for item in result)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class Material:
    """Relative diagonal permittivity and permeability.

    Complex values follow the package's ``exp(+i*omega*t)`` convention.
    """

    eps_r: tuple[complex, complex, complex]
    mu_r: tuple[complex, complex, complex]

    def __init__(self, epsilon: MaterialInput = 1.0, mu: MaterialInput = 1.0) -> None:
        eps = diagonal_values(epsilon, "epsilon")
        permeability = diagonal_values(mu, "mu")
        if any(abs(value) == 0.0 for value in permeability):
            raise ConfigurationError("mu entries must be nonzero.")
        object.__setattr__(self, "eps_r", eps)
        object.__setattr__(self, "mu_r", permeability)

    @property
    def epsilon(self) -> tuple[complex, complex, complex]:
        return self.eps_r

    @property
    def mu(self) -> tuple[complex, complex, complex]:
        return self.mu_r

    @property
    def isotropic(self) -> bool:
        return self.eps_r[0] == self.eps_r[1] == self.eps_r[2] and self.mu_r[0] == self.mu_r[1] == self.mu_r[2]

    def eps_array(self) -> np.ndarray:
        return np.asarray(self.eps_r, dtype=np.complex128)

    def mu_array(self) -> np.ndarray:
        return np.asarray(self.mu_r, dtype=np.complex128)


__all__ = ["Material", "MaterialInput", "diagonal_values"]
