"""Scalar public materials and explicit internal diagonal tensors.

The public MVP intentionally accepts only isotropic scalar relative
constitutive values.  Transformation-optics PMLs are anisotropic even when
the underlying material is isotropic, so the assembly boundary uses the
separate :class:`DiagonalMaterial` representation below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .exceptions import MaterialError

ScalarMaterialInput: TypeAlias = Number
ComplexArray: TypeAlias = NDArray[np.complex128]


def _looks_tensor_like(value: object) -> bool:
    return isinstance(value, (np.ndarray, Mapping)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    )


def _finite_complex_scalar(name: str, value: ScalarMaterialInput) -> complex:
    if _looks_tensor_like(value):
        raise NotImplementedError(
            f"{name} tensor materials are not supported by the public Material "
            "API yet; supply one finite isotropic scalar."
        )
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
        raise MaterialError(f"{name} must be a finite scalar relative value.")
    result = complex(value)
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise MaterialError(f"{name} must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class Material:
    r"""Public isotropic relative material parameters.

    Complex values follow WaveFEM's :math:`e^{-i\omega t}` convention, so a
    passive lossy bulk material normally has nonnegative imaginary parts.
    Active values remain available for research use and are exposed through
    :attr:`is_passive` rather than being silently rejected.
    """

    eps_r: complex = 1.0 + 0.0j
    mu_r: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        object.__setattr__(self, "eps_r", _finite_complex_scalar("eps_r", self.eps_r))
        object.__setattr__(self, "mu_r", _finite_complex_scalar("mu_r", self.mu_r))

    @property
    def is_lossless(self) -> bool:
        """Whether both relative constitutive scalars have zero loss part."""
        return self.eps_r.imag == 0.0 and self.mu_r.imag == 0.0

    @property
    def is_passive(self) -> bool:
        """Whether both scalars obey the passive sign for ``exp(-i*omega*t)``."""
        return self.eps_r.imag >= 0.0 and self.mu_r.imag >= 0.0


@dataclass(frozen=True, slots=True)
class DiagonalTensor:
    """Internal Cartesian diagonal tensor ordered as ``(xx, yy, zz)``."""

    xx: complex
    yy: complex
    zz: complex

    def __post_init__(self) -> None:
        object.__setattr__(self, "xx", _finite_complex_scalar("xx", self.xx))
        object.__setattr__(self, "yy", _finite_complex_scalar("yy", self.yy))
        object.__setattr__(self, "zz", _finite_complex_scalar("zz", self.zz))

    @classmethod
    def isotropic(cls, value: ScalarMaterialInput) -> "DiagonalTensor":
        """Expand one scalar without changing its value or component order."""
        scalar = _finite_complex_scalar("isotropic tensor value", value)
        return cls(scalar, scalar, scalar)

    @property
    def is_isotropic(self) -> bool:
        """Whether all three Cartesian diagonal entries are equal."""
        return self.xx == self.yy == self.zz

    def as_array(self) -> ComplexArray:
        """Return a new complex array in physical ``(xx, yy, zz)`` order."""
        return np.asarray((self.xx, self.yy, self.zz), dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class DiagonalMaterial:
    """Internal diagonal relative permittivity and permeability tensors."""

    eps_r: DiagonalTensor
    mu_r: DiagonalTensor

    def __post_init__(self) -> None:
        if not isinstance(self.eps_r, DiagonalTensor):
            raise MaterialError("DiagonalMaterial.eps_r must be a DiagonalTensor.")
        if not isinstance(self.mu_r, DiagonalTensor):
            raise MaterialError("DiagonalMaterial.mu_r must be a DiagonalTensor.")

    @classmethod
    def from_isotropic(cls, material: Material) -> "DiagonalMaterial":
        """Create the internal representation of one public material."""
        if not isinstance(material, Material):
            raise MaterialError("from_isotropic expects a Material instance.")
        return cls(
            eps_r=DiagonalTensor.isotropic(material.eps_r),
            mu_r=DiagonalTensor.isotropic(material.mu_r),
        )


def as_diagonal_material(
    material: Material | DiagonalMaterial,
) -> DiagonalMaterial:
    """Return the explicit diagonal representation used by assembly and PMLs."""
    if isinstance(material, DiagonalMaterial):
        return material
    if isinstance(material, Material):
        return DiagonalMaterial.from_isotropic(material)
    raise MaterialError("Expected Material or DiagonalMaterial.")


__all__ = [
    "DiagonalMaterial",
    "DiagonalTensor",
    "Material",
    "as_diagonal_material",
]
