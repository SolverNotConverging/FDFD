from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from skfem import MeshTri

from wavefem.constants import MU_0
from wavefem.exceptions import ModeProjectionError
from wavefem.fem import create_mixed_basis
from wavefem.monitors import sample_horizontal_monitor, sample_vertical_monitor


_A = 0.4
_B = -0.6
_C = 0.37
_D = 0.2
_P = 0.7
_Q = -0.3
_KY = 0.45
_OMEGA = 2.1e9
_MONITOR_Z = 0.25


def _manufactured_coefficients(
    *, reverse_node_numbering: bool = False
) -> tuple[object, np.ndarray]:
    mesh = MeshTri.init_tensor(
        np.linspace(-1.0, 1.0, 5),
        np.array([-1.0, -0.25, _MONITOR_Z, 1.0]),
    )
    if reverse_node_numbering:
        permutation = np.arange(mesh.p.shape[1])[::-1]
        old_to_new = np.argsort(permutation)
        mesh = MeshTri(mesh.p[:, permutation], old_to_new[mesh.t])
    basis = create_mixed_basis(mesh, intorder=6)
    component_bases = basis.split_bases()
    component_indices = basis.split_indices()

    # This global polynomial is represented exactly by lowest-order Nedelec:
    # (Ex, Ez) = (A - C z, B + C x), with 2D curl equal to 2 C.
    et_coefficients = component_bases[0].project(
        lambda point: np.stack(
            (_A - _C * point[1], _B + _C * point[0])
        ),
        dtype=np.complex128,
    )
    ey_coefficients = component_bases[1].project(
        lambda point: _D + _P * point[0] + _Q * point[1],
        dtype=np.complex128,
    )
    coefficients = np.zeros(basis.N, dtype=np.complex128)
    coefficients[component_indices[0]] = et_coefficients
    coefficients[component_indices[1]] = ey_coefficients
    return basis, coefficients


def test_manufactured_monitor_trace_and_quadrature_are_exact() -> None:
    basis, coefficients = _manufactured_coefficients()

    def diagonal_mu(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.stack((2.0 + 0.0 * x, 3.0 + 0.0 * z, 4.0 + 0.0 * x))

    samples = sample_vertical_monitor(
        basis,
        coefficients,
        z=_MONITOR_Z,
        ky=_KY,
        omega=_OMEGA,
        mu_r=diagonal_mu,
        intorder=6,
    )

    assert np.all(np.diff(samples.x) > 0.0)
    assert np.all(samples.weights > 0.0)
    assert_allclose(np.sum(samples.weights), 2.0, rtol=0.0, atol=2e-14)

    ex = _A - _C * _MONITOR_Z + 0.0 * samples.x
    ey = _D + _P * samples.x + _Q * _MONITOR_Z
    ez = _B + _C * samples.x
    expected_e = np.stack((ex, ey, ez))
    expected_curl = np.stack(
        (
            1j * _KY * ez - _Q,
            -2.0 * _C + 0.0j * samples.x,
            _P - 1j * _KY * ex,
        )
    )
    expected_h = expected_curl / (
        1j * _OMEGA * MU_0 * np.array((2.0, 3.0, 4.0))[:, None]
    )

    assert samples.E.shape == (3, samples.x.size)
    assert samples.H.shape == (3, samples.x.size)
    assert_allclose(samples.E, expected_e, rtol=2e-14, atol=2e-14)
    assert_allclose(samples.H, expected_h, rtol=2e-13, atol=2e-13)


def test_length_scale_restores_physical_coordinates_weights_and_curl() -> None:
    basis, coefficients = _manufactured_coefficients()
    length_scale = 2.5e-6
    physical_ky = _KY / length_scale

    samples = sample_vertical_monitor(
        basis,
        coefficients,
        z=_MONITOR_Z * length_scale,
        ky=physical_ky,
        omega=_OMEGA,
        mu_r=2.0,
        length_scale=length_scale,
        intorder=6,
    )

    xi = samples.x / length_scale
    ex = _A - _C * _MONITOR_Z + 0.0 * xi
    ez = _B + _C * xi
    expected_curl = np.stack(
        (
            1j * _KY * ez - _Q,
            -2.0 * _C + 0.0j * xi,
            _P - 1j * _KY * ex,
        )
    ) / length_scale
    expected_h = expected_curl / (1j * _OMEGA * MU_0 * 2.0)

    assert samples.z == pytest.approx(_MONITOR_Z * length_scale)
    assert samples.x[0] > -length_scale
    assert samples.x[-1] < length_scale
    assert_allclose(np.sum(samples.weights), 2.0 * length_scale, rtol=2e-15)
    assert_allclose(samples.H, expected_h, rtol=2e-13, atol=2e-7)


def test_trace_is_independent_of_reversed_mesh_numbering_and_facet_order() -> None:
    basis, coefficients = _manufactured_coefficients()
    reversed_basis, reversed_coefficients = _manufactured_coefficients(
        reverse_node_numbering=True
    )
    arguments = dict(z=_MONITOR_Z, ky=_KY, omega=_OMEGA, mu_r=1.0, intorder=6)

    reference = sample_vertical_monitor(basis, coefficients, **arguments)
    reordered = sample_vertical_monitor(
        reversed_basis, reversed_coefficients, **arguments
    )

    assert_allclose(reordered.x, reference.x, rtol=0.0, atol=2e-15)
    assert_allclose(reordered.weights, reference.weights, rtol=0.0, atol=2e-15)
    assert_allclose(reordered.E, reference.E, rtol=2e-14, atol=2e-14)
    assert_allclose(reordered.H, reference.H, rtol=2e-13, atol=2e-13)


def test_monitor_requires_an_interior_mesh_conforming_line() -> None:
    basis, coefficients = _manufactured_coefficients()
    with pytest.raises(ModeProjectionError, match="strictly inside"):
        sample_vertical_monitor(
            basis,
            coefficients,
            z=-1.0,
            ky=0.0,
            omega=_OMEGA,
        )


def test_horizontal_side_monitor_has_exact_trace_and_physical_weights() -> None:
    basis, coefficients = _manufactured_coefficients()
    x_monitor = 0.5
    samples = sample_horizontal_monitor(
        basis,
        coefficients,
        x=x_monitor,
        ky=_KY,
        omega=_OMEGA,
        mu_r=1.0,
        intorder=6,
    )
    assert np.all(np.diff(samples.z) > 0.0)
    assert_allclose(np.sum(samples.weights), 2.0, atol=2e-14)
    ex = _A - _C * samples.z
    ey = _D + _P * x_monitor + _Q * samples.z
    ez = _B + _C * x_monitor + 0.0 * samples.z
    assert_allclose(samples.E, np.stack((ex, ey, ez)), atol=2e-14)


def test_horizontal_monitor_requires_conforming_interior_line() -> None:
    basis, coefficients = _manufactured_coefficients()
    with pytest.raises(ModeProjectionError, match="strictly inside"):
        sample_horizontal_monitor(
            basis, coefficients, x=-1.0, omega=_OMEGA
        )
    with pytest.raises(ModeProjectionError, match="No interior mesh facets"):
        sample_horizontal_monitor(
            basis, coefficients, x=0.1, omega=_OMEGA
        )
    with pytest.raises(ModeProjectionError, match="No interior mesh facets"):
        sample_vertical_monitor(
            basis,
            coefficients,
            z=0.1,
            ky=0.0,
            omega=_OMEGA,
        )
