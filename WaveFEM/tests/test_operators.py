from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.testing import assert_allclose

from wavefem.operators import electric_field_vector, modified_curl


@dataclass
class _TangentialField:
    values: np.ndarray
    curl: np.ndarray

    def __getitem__(self, key: int) -> np.ndarray:
        return self.values[key]


@dataclass
class _InvariantField:
    grad: np.ndarray


def test_modified_curl_nonzero_ky_uses_x_y_z_component_order() -> None:
    ex = np.array([1.0, -2.0])
    ez = np.array([3.0, 4.0])
    d_x_ey = np.array([5.0, 6.0])
    d_z_ey = np.array([-7.0, 8.0])
    curl_xz = np.array([9.0, -10.0])
    ky = 0.75 - 0.2j

    result = modified_curl(
        _TangentialField(np.stack((ex, ez)), curl_xz),
        _InvariantField(np.stack((d_x_ey, d_z_ey))),
        ky,
    )

    expected = np.stack(
        (
            1j * ky * ez - d_z_ey,
            -curl_xz,
            d_x_ey - 1j * ky * ex,
        )
    )
    assert_allclose(result, expected)


def test_modified_curl_at_zero_ky_reduces_to_full_2d_curl() -> None:
    ex = np.array([0.2, 0.4, 0.6])
    ez = np.array([-0.1, -0.3, -0.5])
    d_x_ey = np.array([1.0, 2.0, 3.0])
    d_z_ey = np.array([4.0, 5.0, 6.0])
    standard_curl_xz = np.array([7.0, 8.0, 9.0])

    result = modified_curl(
        _TangentialField(np.stack((ex, ez)), standard_curl_xz),
        _InvariantField(np.stack((d_x_ey, d_z_ey))),
        0.0,
    )

    assert_allclose(
        result,
        np.stack((-d_z_ey, -standard_curl_xz, d_x_ey)),
    )


def test_electric_field_vector_restores_physical_component_order() -> None:
    ex = np.array([1.0, 2.0])
    ey = np.array([3.0, 4.0])
    ez = np.array([5.0, 6.0])
    tangential = _TangentialField(np.stack((ex, ez)), np.zeros_like(ex))

    assert_allclose(
        electric_field_vector(tangential, ey),
        np.stack((ex, ey, ez)),
    )
