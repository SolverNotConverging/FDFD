"""Differential-operator helpers for the 2.5D Maxwell formulation.

The mesh coordinates are ``(x, z)`` and fields use the phasor convention
``Re(E exp(+i omega t))`` with Fourier dependence ``exp(-i k_y y)``.  Thus
``partial_y`` is replaced by ``-i k_y`` everywhere.

The in-plane field ``(E_x, E_z)`` is represented by an H(curl) finite-element
field and ``E_y`` by an H1 finite-element field.  Keeping the modified-curl
formula here prevents sign conventions from being duplicated in assembly
code.
"""

from __future__ import annotations

from typing import Protocol, SupportsIndex

import numpy as np
from numpy.typing import NDArray


class TangentialHcurlField(Protocol):
    """Structural type required by :func:`modified_curl` for ``(E_x, E_z)``."""

    curl: NDArray[np.generic]

    def __getitem__(self, key: SupportsIndex) -> NDArray[np.generic]: ...


class InvariantH1Field(Protocol):
    """Structural type required by :func:`modified_curl` for ``E_y``."""

    grad: NDArray[np.generic]


def modified_curl(
    tangential: TangentialHcurlField,
    invariant: InvariantH1Field,
    ky: complex | float,
) -> NDArray[np.generic]:
    r"""Return the 2.5D curl in physical component order ``(x, y, z)``.

    Parameters
    ----------
    tangential:
        H(curl) field whose components are ``(E_x, E_z)``.  Its ``curl``
        attribute must follow the standard 2D convention
        ``partial_x E_z - partial_z E_x``, as scikit-fem's Nedelec fields do.
    invariant:
        H1 field for ``E_y`` with ``grad[0] = partial_x E_y`` and
        ``grad[1] = partial_z E_y``.
    ky:
        Prescribed Fourier wavenumber in the invariant ``y`` direction.

    Notes
    -----
    This implements exactly

    .. math::

       \nabla_{k_y}\times E =
       (-i k_y E_z-\partial_z E_y,
        \partial_z E_x-\partial_x E_z,
        \partial_x E_y+i k_y E_x).
    """

    return np.stack(
        (
            -1j * ky * tangential[1] - invariant.grad[1],
            -tangential.curl,
            invariant.grad[0] + 1j * ky * tangential[0],
        ),
        axis=0,
    )


def electric_field_vector(
    tangential: TangentialHcurlField,
    invariant: NDArray[np.generic],
) -> NDArray[np.generic]:
    """Combine ``(E_x, E_z)`` and ``E_y`` into physical order ``(x, y, z)``."""

    return np.stack((tangential[0], invariant, tangential[1]), axis=0)
