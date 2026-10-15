"""Band structures: Yee sampling, discrete Bloch dispersion, phasor convention, Gamma point."""

import numpy as np
import pytest
from scipy import linalg

import fdfd
from fdfd import Material
from fdfd.bands._kernel import BandKernel
from fdfd.core import yee
from fdfd.core.constants import C0


def test_materials_are_sampled_at_six_distinct_yee_locations():
    kernel = BandKernel(12e-3, 10e-3, 6, 5)
    kernel.ER2[:] = 2 + kernel.X2 / 12e-3 + 2 * kernel.Y2 / 10e-3
    kernel.UR2[:] = 3 + 2 * kernel.X2 / 12e-3 + kernel.Y2 / 10e-3
    offsets = {
        "eps_xx": (0.5, 0.0),
        "eps_yy": (0.0, 0.5),
        "eps_zz": (0.0, 0.0),
        "mu_xx": (0.0, 0.5),
        "mu_yy": (0.5, 0.0),
        "mu_zz": (0.5, 0.5),
    }
    for name, (ox, oy) in offsets.items():
        x = -6e-3 + (np.arange(6) + ox) * kernel.dx
        y = -5e-3 + (np.arange(5) + oy) * kernel.dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        expected = (
            2 + xx / 12e-3 + 2 * yy / 10e-3 if name.startswith("eps") else 3 + 2 * xx / 12e-3 + yy / 10e-3
        )
        np.testing.assert_allclose(kernel.tensors()[name], expected)


def test_lossy_bands_obey_the_discrete_bloch_dispersion_with_positive_imaginary_frequency():
    eps, mu = 2.25 - 0.09j, 1.3 - 0.02j
    kernel = BandKernel(1.0, 1.2, 6, 5, background_epsilon=eps, background_mu=mu)
    beta = np.array([[0.3], [0.2]])
    frequencies, residuals = kernel.solve(
        beta, 1, polarizations=("TE", "TM"), sigma=0.1, tol=0.0, device="cpu", method="arpack", progress=False
    )
    k2 = (2 * np.sin(0.3 * kernel.dx / 2) / kernel.dx) ** 2 + (
        2 * np.sin(0.2 * kernel.dy / 2) / kernel.dy
    ) ** 2
    for pol in ("TE", "TM"):
        eigenvalue = (2 * np.pi * frequencies[pol] / C0) ** 2
        np.testing.assert_allclose(eigenvalue, [[k2 / (eps * mu)]], rtol=1e-10)
        assert frequencies[pol][0, 0].imag > 0
        assert residuals[pol][0, 0] < 1e-12


def test_bloch_wrap_uses_negative_spatial_phase():
    derivative = yee.bloch_difference((3, 1), 0, 0.2, 0.37)
    assert derivative[2, 0] == pytest.approx(np.exp(-1j * 0.37 * 3 * 0.2) / 0.2)
    single = yee.bloch_difference((1, 4), 0, 0.2, 0.37)
    np.testing.assert_allclose(single.diagonal(), -0.37j)


def test_gamma_point_matches_a_dense_eigensolve_and_is_deterministic():
    a = 10e-3
    solver = fdfd.BandStructureSolver2D(x_range=(-a / 2, a / 2), y_range=(-a / 2, a / 2))
    solver.add_circle(center=(0.0, 0.0), radius=0.2 * a, material=Material(name="rod", epsilon=8.9))
    solver.mesh(resolution=(16, 16), subpixels=1)
    gamma = np.zeros((2, 1))
    kernel = solver._kernel
    t = {name: yee.diag(values) for name, values in kernel.tensors().items()}
    dex = yee.bloch_difference((16, 16), 0, kernel.dx, 0.0)
    dey = yee.bloch_difference((16, 16), 1, kernel.dy, 0.0)
    operator = t["eps_zz"].power(-1) @ (
        dex.conj().T @ t["mu_yy"].power(-1) @ dex + dey.conj().T @ t["mu_xx"].power(-1) @ dey
    )
    dense = np.sort(linalg.eigvals(operator.toarray()).real)[:4]
    expected = C0 * np.sqrt(np.clip(dense, 0, None)) / (2 * np.pi)
    for _ in range(3):  # repeated calls must agree (ARPACK start vectors vary)
        result = solver.solve(beta_path=gamma, num_modes=4, polarization="TM", progress=False)
        np.testing.assert_allclose(result.frequencies["TM"][:, 0].real, expected, rtol=1e-9, atol=1e4)


def test_guess_selects_bands_near_a_frequency():
    a = 10e-3
    solver = fdfd.BandStructureSolver2D(x_range=(0.0, a), y_range=(0.0, a))
    solver.mesh(resolution=(12, 12))
    path = np.array([[np.pi / a], [0.0]])
    low = solver.solve(beta_path=path, num_modes=3, polarization="TE", progress=False)
    high = solver.solve(
        beta_path=path, num_modes=1, polarization="TE", guess=low.frequencies["TE"][2, 0].real, progress=False
    )
    assert high.frequencies["TE"][0, 0] == pytest.approx(low.frequencies["TE"][2, 0], rel=1e-9)
    assert high.info.guess == pytest.approx(low.frequencies["TE"][2, 0].real)


def test_bloch_path_samples_every_vertex():
    path = fdfd.bloch_path(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), num_points=5)
    assert path.shape == (2, 5)
    for vertex in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)):
        assert np.any(np.all(np.isclose(path.T, vertex), axis=1))
    with pytest.raises(fdfd.ConfigurationError):
        fdfd.bloch_path(((0.0, 0.0), (0.0, 0.0)), num_points=4)
