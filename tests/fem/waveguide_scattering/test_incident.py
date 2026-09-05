from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cem_common.errors import ConfigurationError
from fem_waveguide_scattering.incident import IncidentMode
from fem_waveguide_scattering.materials import Material
from fem_waveguide_scattering.modes import CrossSection, Mode, ModeSolver


def _sample_mode() -> Mode:
    return Mode(
        beta=2.0 - 0.1j,
        neff=1.5 - 0.075j,
        E_x=np.array([2.0, 4.0], dtype=complex),
        E_y=np.array([0.0, 2.0, 6.0], dtype=complex),
        E_z=np.array([1.0, 1.0 + 2.0j, 1.0 + 6.0j]),
        H_x=np.array([5.0, 7.0], dtype=complex),
        H_y=np.array([11.0, 13.0], dtype=complex),
        H_z=np.array([17.0, 19.0], dtype=complex),
        x_nodes=np.array([0.0, 1.0, 3.0]),
        power=1.0,
        complex_power=1.0 + 0.02j,
        ky=0.3,
        omega=4.0,
        direction="forward",
        classification="propagating",
        normalization="unit-power",
        residual=1e-12,
        divergence_residual=2e-12,
    )


def test_mode_sampling_preserves_mixed_fem_representation_and_shape() -> None:
    mode = _sample_mode()
    x = np.array([[0.25, 0.75], [1.5, 2.5]])

    electric = mode.sample_E(x)
    magnetic = mode.sample_H(x)

    assert electric.shape == (3, 2, 2)
    assert magnetic.shape == (3, 2, 2)
    np.testing.assert_allclose(electric[0], [[2.0, 2.0], [4.0, 4.0]])
    np.testing.assert_allclose(electric[1], 2.0 * x)
    np.testing.assert_allclose(electric[2], 1.0 + 2.0j * x)
    np.testing.assert_allclose(magnetic[0], [[5.0, 5.0], [7.0, 7.0]])
    np.testing.assert_allclose(magnetic[1], [[11.0, 11.0], [13.0, 13.0]])
    np.testing.assert_allclose(magnetic[2], [[17.0, 17.0], [19.0, 19.0]])

    # The discontinuous P0 trace uses the right cell at an internal node.
    np.testing.assert_allclose(mode.sample_E(1.0), [4.0, 2.0, 1.0 + 2.0j])
    assert mode.sample_E(0.5).shape == (3,)

    with pytest.raises(ValueError, match="outside"):
        mode.sample_E([-1e-3, 0.5])
    with pytest.raises(ValueError, match="real"):
        mode.sample_H(0.5 + 0.1j)


def test_mode_sampling_interpolates_piecewise_linear_hx_endpoints() -> None:
    mode = replace(
        _sample_mode(),
        H_x_left=np.asarray((4.0, 6.0), dtype=complex),
        H_x_right=np.asarray((6.0, 10.0), dtype=complex),
    )
    np.testing.assert_allclose(mode.sample_H([0.25, 0.75])[0], [4.5, 5.5])
    np.testing.assert_allclose(mode.sample_H([1.5, 2.5])[0], [7.0, 9.0])
    backward = mode.counterpropagating()
    np.testing.assert_allclose(
        backward.sample_H([0.25, 1.5])[0],
        -mode.sample_H([0.25, 1.5])[0],
    )


def test_mode_fields_broadcast_and_use_the_declared_reference_plane() -> None:
    mode = _sample_mode()
    x = np.array([[0.25], [1.5]])
    z = np.array([[0.5, 0.75, 1.0]])

    electric, magnetic = mode.fields(x, z, reference_plane=0.5)

    assert electric.shape == (3, 2, 3)
    assert magnetic.shape == (3, 2, 3)
    expected_phase = np.exp(-1j * mode.beta * (z - 0.5))
    np.testing.assert_allclose(electric, mode.sample_E(np.broadcast_to(x, (2, 3))) * expected_phase)
    np.testing.assert_allclose(magnetic, mode.sample_H(np.broadcast_to(x, (2, 3))) * expected_phase)
    np.testing.assert_allclose(electric[:, :, 0], mode.sample_E(x[:, 0]))

    with pytest.raises(ValueError, match="broadcastable"):
        mode.fields(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        mode.fields(0.5, np.inf)


def test_counterpropagating_mode_is_the_exact_z_mirror() -> None:
    forward = _sample_mode()
    backward = forward.counterpropagating()

    assert backward.beta == -forward.beta
    assert backward.neff == -forward.neff
    assert backward.direction == "backward"
    assert backward.power == -forward.power
    assert backward.complex_power == -forward.complex_power
    assert backward.normalization == "unit-power"
    np.testing.assert_allclose(backward.E_x, forward.E_x)
    np.testing.assert_allclose(backward.E_y, forward.E_y)
    np.testing.assert_allclose(backward.E_z, -forward.E_z)
    np.testing.assert_allclose(backward.H_x, -forward.H_x)
    np.testing.assert_allclose(backward.H_y, -forward.H_y)
    np.testing.assert_allclose(backward.H_z, forward.H_z)

    restored = backward.counterpropagating()
    np.testing.assert_allclose(restored.sample_E([0.25, 1.5]), forward.sample_E([0.25, 1.5]))
    np.testing.assert_allclose(restored.sample_H([0.25, 1.5]), forward.sample_H([0.25, 1.5]))
    assert backward.backward() is backward


def test_solved_unit_power_mode_launches_from_either_side_with_correct_sign() -> None:
    wavelength = 1.0
    width = 0.2
    eps_r = 2.25
    ky = 0.35 * 2.0 * np.pi / wavelength
    expected_neff = np.sqrt(eps_r - 0.35**2)
    cross_section = CrossSection(
        x_span=(-width / 2.0, width / 2.0),
        background=Material(eps_r=eps_r),
        boundary="pec",
    )
    forward = ModeSolver(
        cross_section,
        wavelength=wavelength,
        ky=ky,
        num_elements=10,
        dense_linearization_limit=256,
    ).solve(
        max_refinements=0,
        num_modes=1,
        neff_guess=expected_neff,
        residual_tolerance=1e-9,
        divergence_tolerance=1e-9,
    )[0]

    left = IncidentMode(forward, side="left", reference_plane=-0.4, amplitude=2j)
    right = IncidentMode(forward, side="right", reference_plane=0.4, amplitude=2j)

    assert left.direction == "forward"
    assert right.direction == "backward"
    assert left.signed_power == pytest.approx(4.0)
    assert right.signed_power == pytest.approx(-4.0)
    assert right.beta == -left.beta

    x = np.array([-0.05, 0.0, 0.05])
    np.testing.assert_allclose(left.E(x, -0.4), 2j * forward.sample_E(x))
    np.testing.assert_allclose(left.H(x, -0.4), 2j * forward.sample_H(x))
    np.testing.assert_allclose(left(x, -0.4), left.E(x, -0.4))

    backward = forward.counterpropagating()
    np.testing.assert_allclose(right.E(x, 0.4), 2j * backward.sample_E(x))
    np.testing.assert_allclose(right.H(x, 0.4), 2j * backward.sample_H(x))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"side": "top"}, "side"),
        ({"reference_plane": np.inf}, "reference_plane"),
        ({"amplitude": np.nan}, "amplitude"),
        ({"amplitude": [1.0]}, "amplitude"),
    ],
)
def test_incident_configuration_is_validated(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        IncidentMode(_sample_mode(), **kwargs)  # type: ignore[arg-type]
