from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from wavefem.results import ScatteringResult
from wavefem.scattering import Scattering2D


_WAVELENGTH = 1.55e-6
_K0 = 2.0 * np.pi / _WAVELENGTH
_KY = 0.28 * _K0
_MONITOR = 1.25e-6

Layer = tuple[float, float, float]


def _solve_pec_guide(
    *,
    ky: float,
    layers: Sequence[Layer] = (),
    polygon: Sequence[tuple[float, float]] | None = None,
    x_span: tuple[float, float] = (0.0, 0.8e-6),
    layer_x: tuple[float, float] | None = None,
    maximum_edge: float = 0.20e-6,
    num_modes: int = 1,
) -> ScatteringResult:
    """Solve a compact reciprocal PEC-guide regression case."""

    simulation = Scattering2D(
        wavelength=_WAVELENGTH,
        ky=ky,
        x_span=x_span,
        z_span=(-2.8e-6, 2.8e-6),
        background_eps=1.0,
        transverse_boundary="pec",
    )
    layer_x = simulation.x_span if layer_x is None else layer_x
    for index, (z_min, z_max, eps_r) in enumerate(layers):
        simulation.add_rectangle(
            x=layer_x,
            z=(z_min, z_max),
            eps=eps_r,
            name=f"layer_{index}",
        )
    if polygon is not None:
        simulation.add_polygon(points=polygon, eps=1.14, name="compact_perturbation")
    simulation.add_pml(z=0.7e-6, order=3, target_reflection=1e-8)
    simulation.set_monitors(left=-_MONITOR, right=_MONITOR)
    simulation.mesh(max_element_size=maximum_edge, wavelength_elements=7)
    modes = simulation.solve_modes(
        num_modes=num_modes,
        neff_guess=np.sqrt(1.0 - (ky / _K0) ** 2),
        num_elements=48,
    )
    simulation.set_incident_mode(modes[0])
    return simulation.solve()


@pytest.mark.gmsh
@pytest.mark.slow
def test_nonzero_ky_sign_symmetry_of_scattering_and_power() -> None:
    layers = ((-0.42e-6, 0.42e-6, 1.12),)

    positive = _solve_pec_guide(ky=_KY, layers=layers)
    negative = _solve_pec_guide(ky=-_KY, layers=layers)

    assert positive.reflection > 1e-4
    np.testing.assert_allclose(
        [negative.S11, negative.S21],
        [positive.S11, positive.S21],
        rtol=2e-6,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        [negative.reflection, negative.transmission, negative.power_balance],
        [positive.reflection, positive.transmission, positive.power_balance],
        rtol=2e-6,
        atol=2e-9,
    )
    assert positive.power_balance_error < 1e-3
    assert negative.power_balance_error < 1e-3


@pytest.mark.gmsh
@pytest.mark.slow
def test_reciprocal_transmission_matches_z_mirrored_left_incidence() -> None:
    # Scattering2D currently solves only left incidence.  Left incidence on
    # the z-mirrored device is physically the original device's right-incidence
    # experiment, with the same power normalization and mirrored reference
    # planes.  Reciprocity therefore requires the complex transmissions to
    # agree even though the two complex reflection coefficients need not.
    polygon = (
        (0.12e-6, -0.50e-6),
        (0.38e-6, -0.36e-6),
        (0.34e-6, 0.18e-6),
        (0.17e-6, 0.11e-6),
    )
    mirrored_polygon = tuple((x, -z) for x, z in reversed(polygon))

    common = {
        "ky": _KY,
        "x_span": (0.0, 0.50e-6),
        "maximum_edge": 0.035e-6,
        "num_modes": 2,
    }
    original = _solve_pec_guide(polygon=polygon, **common)
    mirrored = _solve_pec_guide(polygon=mirrored_polygon, **common)

    assert original.reflection > 1e-4
    assert mirrored.S21 == pytest.approx(original.S21, rel=0.0, abs=8e-5)
    assert mirrored.transmission == pytest.approx(
        original.transmission, rel=0.0, abs=1.5e-4
    )
    # A passive reciprocal two-port with one propagating modal channel has
    # equal reflectance from either side, although the phases may differ.
    assert mirrored.reflection == pytest.approx(
        original.reflection, rel=0.0, abs=3e-5
    )
    assert original.power_balance_error < 5e-4
    assert mirrored.power_balance_error < 5e-4
    assert original.solve_info["left_projection_residual"] < 3e-3
    assert original.solve_info["right_projection_residual"] < 3e-3
    assert mirrored.solve_info["left_projection_residual"] < 3e-3
    assert mirrored.solve_info["right_projection_residual"] < 3e-3
