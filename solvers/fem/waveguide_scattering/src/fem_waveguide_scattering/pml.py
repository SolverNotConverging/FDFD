"""Transformation-optics perfectly matched layers.

For the project convention ``exp(+i omega t)`` and outgoing
``exp(-i beta z)``, a negative imaginary coordinate stretch produces decay.
The transformed constitutive tensors follow

``det(S) S^-1 material S^-T``, with ``S = diag(sx, 1, sz)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import ConfigurationError


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class PML:
    """Polynomial complex-stretch PML specification.

    Parameters
    ----------
    thickness:
        Physical PML thickness in metres.
    order:
        Polynomial profile order.
    target_reflection:
        Nominal amplitude reflection used to choose the maximum stretch.
    """

    thickness: float
    order: int = 3
    target_reflection: float = 1e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ConfigurationError("PML thickness must be finite and positive.")
        if isinstance(self.order, bool) or int(self.order) != self.order or self.order < 1:
            raise ConfigurationError("PML order must be a positive integer.")
        if not 0.0 < self.target_reflection < 1.0:
            raise ConfigurationError("PML target_reflection must lie strictly between 0 and 1.")

    def maximum_imaginary_stretch(self, k_reference: float) -> float:
        """Return the profile peak for a nominal two-pass reflection target."""

        if not np.isfinite(k_reference) or k_reference <= 0.0:
            raise ConfigurationError("PML reference wavenumber must be finite and positive.")
        return -(
            (self.order + 1)
            * np.log(self.target_reflection)
            / (2.0 * k_reference * self.thickness)
        )

    def stretch(self, depth: ArrayLike, k_reference: float) -> ComplexArray:
        """Evaluate ``s = 1 + i alpha_max (depth/thickness)^order``."""

        d = np.clip(np.asarray(depth, dtype=float), 0.0, self.thickness)
        profile = (d / self.thickness) ** self.order
        return np.asarray(
            1.0 - 1j * self.maximum_imaginary_stretch(k_reference) * profile,
            dtype=np.complex128,
        )


@dataclass(frozen=True, slots=True)
class PMLLayout:
    """Independent x- and z-directed PMLs around a rectangular domain."""

    x: PML | None = None
    z: PML | None = None

    def validate_domain(
        self, x_span: Sequence[float], z_span: Sequence[float]
    ) -> None:
        if self.x is not None and 2.0 * self.x.thickness >= x_span[1] - x_span[0]:
            raise ConfigurationError("Two x-PMLs leave no non-PML interior.")
        if self.z is not None and 2.0 * self.z.thickness >= z_span[1] - z_span[0]:
            raise ConfigurationError("Two z-PMLs leave no non-PML interior.")

    def stretching(
        self,
        x: ArrayLike,
        z: ArrayLike,
        *,
        x_span: Sequence[float],
        z_span: Sequence[float],
        k_reference: float,
    ) -> tuple[ComplexArray, ComplexArray]:
        """Return diagonal stretch components ``(sx, sz)``."""

        self.validate_domain(x_span, z_span)
        xa, za = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
        sx = np.ones(xa.shape, dtype=np.complex128)
        sz = np.ones(za.shape, dtype=np.complex128)
        if self.x is not None:
            depth = np.maximum(x_span[0] + self.x.thickness - xa, 0.0)
            depth = np.maximum(depth, xa - (x_span[1] - self.x.thickness))
            sx = self.x.stretch(depth, k_reference)
        if self.z is not None:
            depth = np.maximum(z_span[0] + self.z.thickness - za, 0.0)
            depth = np.maximum(depth, za - (z_span[1] - self.z.thickness))
            sz = self.z.stretch(depth, k_reference)
        return sx, sz

    def transform_isotropic(
        self,
        eps_r: ArrayLike,
        mu_r: ArrayLike,
        sx: ArrayLike,
        sz: ArrayLike,
    ) -> tuple[ComplexArray, ComplexArray]:
        """Transform scalar material into diagonal ``(x, y, z)`` tensors."""

        eps, mu, sxa, sza = np.broadcast_arrays(
            np.asarray(eps_r, dtype=np.complex128),
            np.asarray(mu_r, dtype=np.complex128),
            np.asarray(sx, dtype=np.complex128),
            np.asarray(sz, dtype=np.complex128),
        )
        factors = np.stack((sza / sxa, sxa * sza, sxa / sza))
        return factors * eps, factors * mu

    def interfaces(
        self, x_span: Sequence[float], z_span: Sequence[float]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return internal x/z coordinates that should be mesh-conforming."""

        xs: tuple[float, ...] = ()
        zs: tuple[float, ...] = ()
        if self.x is not None:
            xs = (x_span[0] + self.x.thickness, x_span[1] - self.x.thickness)
        if self.z is not None:
            zs = (z_span[0] + self.z.thickness, z_span[1] - self.z.thickness)
        return xs, zs
