"""Waveguide solvers: phasor convention, field reconstruction, PML, guess, ordering."""

import numpy as np
import pytest

import fdfd
from fdfd import Material, materials, shapes
from fdfd.core.constants import C0

LOSSY = Material(name="lossy", epsilon=2.25 - 0.09j)


def parallel_plate(aperture_cells, *, frequency=30e9, aperture=10e-3):
    dx = aperture / aperture_cells
    solver = fdfd.ModeSolver1D(frequency=frequency, x_range=(-dx, aperture + dx), background_material=LOSSY)
    solver.add_layer(x_range=(-dx, 0.0), material=materials.PEC, name="left")
    solver.add_layer(x_range=(aperture, aperture + dx), material=materials.PEC, name="right")
    solver.mesh(resolution=aperture_cells + 2, subpixels=1)
    return solver


def test_1d_passive_bulk_loss_has_negative_imaginary_neff():
    frequency, aperture = 30e9, 10e-3
    solver = parallel_plate(80)
    k0 = 2 * np.pi * frequency / C0
    expected = {"TM": np.sqrt(LOSSY.epsilon), "TE": np.sqrt(LOSSY.epsilon - (np.pi / (k0 * aperture)) ** 2)}
    for pol, value in expected.items():
        result = solver.solve(num_modes=1, polarization=pol, guess=np.sqrt(LOSSY.epsilon))
        assert result.neff[0] == pytest.approx(value, abs=2e-5)
        assert result.neff[0].imag < 0
        assert result.attenuation_constant[0] == pytest.approx(-result.neff[0].imag * result.k0)


def test_2d_passive_bulk_loss_has_negative_imaginary_neff():
    frequency, width, height = 10e9, 22.86e-3, 10.16e-3
    dx, dy = width / 24, height / 12
    solver = fdfd.ModeSolver2D(
        frequency=frequency, x_range=(-dx, width + dx), y_range=(-dy, height + dy), background_material=LOSSY
    )
    frame = shapes.Difference(
        shape=shapes.Rectangle(bounds=((-dx, width + dx), (-dy, height + dy))),
        tool=shapes.Rectangle(bounds=((0.0, width), (0.0, height))),
    )
    solver.add_geometry(shape=frame, material=materials.PEC)
    solver.mesh(resolution=(26, 14), subpixels=1)
    expected = np.sqrt(LOSSY.epsilon - (C0 / (2 * width * frequency)) ** 2)
    result = solver.solve(num_modes=1, guess=expected)
    assert result.neff[0] == pytest.approx(expected, abs=5e-4)
    assert result.neff[0].imag < 0 and result.attenuation_constant[0] > 0
    mu_xx = solver._kernel._effective_materials_and_masks()[0]["mu_xx"]
    np.testing.assert_allclose(
        result.fields["Hx"][..., 0],
        1j * result.neff[0] / mu_xx * result.fields["Ey"][..., 0],
        rtol=5e-4,
        atol=1e-10,
    )


def test_1d_reconstructed_fields_follow_exp_plus_iwt_maxwell_phase():
    solver = parallel_plate(40)
    materials_ = solver._kernel._effective_materials_and_masks()[0]
    te = solver.solve(num_modes=1, polarization="TE", guess=1.5)
    tm = solver.solve(num_modes=1, polarization="TM", guess=1.5)
    np.testing.assert_allclose(
        te.fields["Hx"][:, 0],
        1j * te.neff[0] / materials_["mu_xx"] * te.fields["Ey"][:, 0],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tm.fields["Ex"][:, 0],
        1j * tm.neff[0] / materials_["eps_xx"] * tm.fields["Hy"][:, 0],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(te.fields["Hy"] == 0) and np.all(tm.fields["Ey"] == 0)


def test_1d_pml_is_a_negative_imaginary_coordinate_stretch():
    solver = fdfd.ModeSolver1D(frequency=10e9, x_range=(0.0, 6e-3))
    solver.add_pml(thickness=1e-3, sides="x-", strength=0.5, order=1)
    solver.mesh(resolution=6)
    eps, mu = solver._kernel.cell_tensors()
    stretch = 1 - 0.5j * 0.5  # depth at the first cell centre is half the layer
    for tensor in (eps, mu):
        assert tensor[0][0] == pytest.approx(1 / stretch)
        assert tensor[1][0] == pytest.approx(stretch)
        assert tensor[2][0] == pytest.approx(stretch)
        np.testing.assert_array_equal(tensor[0][1:], 1.0)
    np.testing.assert_array_equal(solver._kernel._pml_cell_mask, [True] + [False] * 5)


def test_2d_pml_uses_every_tensor_ratio():
    solver = fdfd.ModeSolver2D(frequency=10e9, x_range=(0.0, 4e-3), y_range=(0.0, 4e-3))
    solver.add_pml(thickness=1e-3, strength=0.5, order=1)
    solver.mesh(resolution=(4, 4))
    eps, mu = solver._kernel.cell_tensors()
    s = 1 - 0.25j
    # x-only, y-only and corner cells.
    assert (eps[0][0, 2], eps[1][0, 2], eps[2][0, 2]) == pytest.approx((1 / s, s, s))
    assert (eps[0][2, 0], eps[1][2, 0], eps[2][2, 0]) == pytest.approx((s, 1 / s, s))
    assert (eps[0][0, 0], eps[1][0, 0], eps[2][0, 0]) == pytest.approx((1.0, 1.0, s**2))
    for e, m in zip(eps, mu, strict=True):
        np.testing.assert_allclose(m, e)


@pytest.mark.parametrize("strength", [-1.0, np.inf, np.nan])
def test_active_or_nonfinite_pml_is_rejected(strength):
    solver = fdfd.ModeSolver2D(frequency=10e9, x_range=(0.0, 4e-3), y_range=(0.0, 4e-3))
    with pytest.raises(fdfd.ConfigurationError, match="strength"):
        solver.add_pml(thickness=1e-3, strength=strength)


def test_pml_thickness_is_not_rounded_to_cells():
    solver = fdfd.ModeSolver1D(frequency=10e9, x_range=(0.0, 10e-3))
    solver.add_pml(thickness=1.5e-3, sides="x+", order=1, strength=1.0)
    solver.mesh(resolution=10)
    eps_yy = solver._kernel.cell_eps_r_yy
    # Cell centres at 8.5 and 9.5 mm lie 0 and 1 mm into the 1.5 mm layer.
    np.testing.assert_allclose(eps_yy[-2:], [1.0, 1 - 1j / 1.5])
    assert eps_yy[-3] == 1.0


@pytest.mark.parametrize("dimension", [1, 2])
def test_automatic_guess_bounds_anisotropic_materials(dimension):
    background = Material(name="background", epsilon=(2.0, 5.0, 3.0), mu=(4.0, 1.0, 2.0))
    core = Material(name="core", epsilon=(9.0 - 0.5j, 2.0, 3.0), mu=(2.0, 3.0, 1.0))
    if dimension == 1:
        solver = fdfd.ModeSolver1D(frequency=30e9, x_range=(0.0, 20e-3), background_material=background)
        solver.add_layer(x_range=(5e-3, 10e-3), material=core)
    else:
        solver = fdfd.ModeSolver2D(
            frequency=30e9, x_range=(0.0, 20e-3), y_range=(0.0, 16e-3), background_material=background
        )
        solver.add_rectangle(x_range=(5e-3, 10e-3), y_range=(5e-3, 10e-3), material=core)
    bound = np.sqrt(abs(9.0 - 0.5j) * 3.0)
    assert solver._search_guess(None) == pytest.approx(1.01 * bound)
    assert solver._search_guess(0.5) == 0.5


def test_automatic_guess_returns_fundamental_modes_and_explicit_guess_is_honoured():
    width, cells, frequency = 20e-3, 80, 30e9
    step = width / cells
    solver = fdfd.ModeSolver1D(
        frequency=frequency,
        x_range=(-step, width + step),
        background_material=Material(name="fill", epsilon=4.0, mu=3.0),
    )
    solver.add_layer(x_range=(-step, 0.0), material=materials.PEC, name="left")
    solver.add_layer(x_range=(width, width + step), material=materials.PEC, name="right")
    solver.mesh(resolution=cells + 2, subpixels=1)
    automatic = solver.solve(num_modes=3, polarization="TE")
    assert automatic.info.guess.real > np.sqrt(12.0)
    assert np.all(np.diff(automatic.neff.real) < 0)
    k0 = 2 * np.pi * frequency / C0
    expected = np.sqrt(12.0 - (np.arange(1, 4) * np.pi / (k0 * width)) ** 2)
    np.testing.assert_allclose(automatic.neff.real, expected, rtol=2e-3)
    high = solver.solve(num_modes=3, polarization="TE", guess=5.0)
    np.testing.assert_allclose(high.neff, automatic.neff, atol=1e-10)
    # An explicit guess targets the modes nearest to it, e.g. higher orders.
    targeted = solver.solve(num_modes=1, polarization="TE", guess=expected[2])
    assert targeted.neff[0].real == pytest.approx(automatic.neff[2].real, rel=1e-8)


def test_vector_modes_are_returned_in_descending_neff_order():
    solver = fdfd.ModeSolver2D(
        frequency=30e9,
        x_range=(0.0, 20e-3),
        y_range=(0.0, 16e-3),
        background_material=Material(name="fill", epsilon=4.0, mu=3.0),
    )
    solver.mesh(resolution=(18, 14))
    result = solver.solve(num_modes=4)
    assert result.info.guess.real > np.sqrt(12.0)
    assert np.all(np.diff(result.neff.real) <= 0)


@pytest.mark.parametrize(("source", "implied"), [("pec_xx", "pmc_yy"), ("pec_yy", "pmc_xx")])
def test_tangential_pec_also_fixes_the_collocated_normal_h(source, implied):
    from fdfd.waveguide._kernel_2d import WaveguideKernel2D

    kernel = WaveguideKernel2D(10e9, 5e-3, 4e-3, 5, 4)
    mask = np.zeros((5, 4), dtype=bool)
    mask[2, 1] = True
    kernel.add_pec(mask)
    _, masks = kernel._effective_materials_and_masks()
    names = ("pec_xx", "pec_yy", "pec_zz", "pmc_xx", "pmc_yy", "pmc_zz")
    effective = dict(zip(names, masks, strict=True))
    assert np.any(effective[source])
    np.testing.assert_array_equal(effective[implied], effective[source])


@pytest.mark.parametrize("dimension", [1, 2])
def test_fields_keep_complete_transverse_yee_lattices(dimension):
    if dimension == 1:
        solver = fdfd.ModeSolver1D(frequency=20e9, x_range=(0.0, 10e-3))
        solver.mesh(resolution=12)
    else:
        solver = fdfd.ModeSolver2D(frequency=20e9, x_range=(0.0, 10e-3), y_range=(0.0, 8e-3))
        solver.mesh(resolution=(12, 10))
    result = solver.solve(num_modes=1)
    offsets = {
        "Ex": (0.5, 0.0),
        "Ey": (0.0, 0.5),
        "Ez": (0.0, 0.0),
        "Hx": (0.0, 0.5),
        "Hy": (0.5, 0.0),
        "Hz": (0.5, 0.5),
    }
    for name, field in result.fields.items():
        for n, coordinates, cells, bounds, offset in zip(
            field.shape[:-1],
            result.field_coordinates[name],
            result.grid.resolution,
            result.grid.bounds,
            offsets[name],
            strict=False,
        ):
            assert n == cells + (offset == 0.0)
            expected = bounds[0] + (np.arange(n) + offset) * (bounds[1] - bounds[0]) / cells
            np.testing.assert_allclose(coordinates, expected, atol=1e-15)
