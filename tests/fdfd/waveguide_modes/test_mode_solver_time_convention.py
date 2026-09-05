import unittest

import numpy as np

from fdfd_waveguide_modes.solver_1d import _ModeSolver1D as ModeSolver1D
from fdfd_waveguide_modes.solver_2d import _ModeSolver2D as ModeSolver2D


class ModeSolverTimeConventionTests(unittest.TestCase):
    EPSILON0 = 8.854187817e-12
    MU0 = 4e-7 * np.pi

    def test_1d_passive_bulk_loss_has_negative_imaginary_neff(self):
        frequency = 30e9
        aperture = 10e-3
        aperture_cells = 80
        nx = aperture_cells + 2
        dx = aperture / aperture_cells
        epsilon_r = 2.25 - 0.09j
        c = 1 / np.sqrt(self.EPSILON0 * self.MU0)
        k0 = 2 * np.pi * frequency / c

        solver = ModeSolver1D(
            frequency,
            nx * dx,
            nx,
            num_modes=1,
            guess=-epsilon_r,
        )
        solver.add_layer(epsilon_r, 1.0, (0, nx))
        solver.add_pec((0, 1))
        solver.add_pec((nx - 1, nx))
        solver.solve()

        expected_tm = np.sqrt(epsilon_r)
        expected_te = np.sqrt(epsilon_r - (np.pi / (k0 * aperture)) ** 2)
        self.assertAlmostEqual(solver.neff_TM[0], expected_tm, delta=2e-5)
        self.assertAlmostEqual(solver.neff_TE[0], expected_te, delta=2e-5)
        self.assertLess(solver.neff_TM[0].imag, 0.0)
        self.assertLess(solver.neff_TE[0].imag, 0.0)
        self.assertAlmostEqual(
            solver.attenuation_constant_TM[0],
            -solver.neff_TM[0].imag,
        )
        self.assertAlmostEqual(
            solver.attenuation_constant_TE[0],
            -solver.neff_TE[0].imag,
        )

    def test_2d_passive_bulk_loss_has_negative_imaginary_neff(self):
        frequency = 10e9
        width = 22.86e-3
        height = 10.16e-3
        aperture_nx, aperture_ny = 24, 12
        dx, dy = width / aperture_nx, height / aperture_ny
        nx, ny = aperture_nx + 2, aperture_ny + 2
        epsilon_r = 2.25 - 0.09j
        c = 1 / np.sqrt(self.EPSILON0 * self.MU0)
        expected_squared = epsilon_r - (c / (2 * width * frequency)) ** 2
        expected_neff = np.sqrt(expected_squared)

        solver = ModeSolver2D(
            frequency,
            nx * dx,
            ny * dy,
            nx,
            ny,
            num_modes=1,
            guess=-expected_squared,
        )
        solver.add_rectangle(epsilon_r, 1.0, (0, nx), (0, ny))
        for x_range, y_range in (
            ((0, 1), (0, ny)),
            ((nx - 1, nx), (0, ny)),
            ((1, nx - 1), (0, 1)),
            ((1, nx - 1), (ny - 1, ny)),
        ):
            solver.add_pec(x_range, y_range)
        solver.solve()

        self.assertAlmostEqual(solver.neff[0], expected_neff, delta=5e-4)
        self.assertLess(solver.neff[0].imag, 0.0)
        self.assertGreater(solver.attenuation_constant[0], 0.0)
        self.assertAlmostEqual(
            solver.attenuation_constant[0],
            -solver.neff[0].imag,
        )
        self.assertFalse(np.shares_memory(solver.Ex, solver.eigenvectors))
        self.assertFalse(np.shares_memory(solver.Ey, solver.eigenvectors))
        np.testing.assert_allclose(
            solver.Hx[:, :, 0],
            1j * solver.neff[0] / solver.mu_r_xx * solver.Ey[:, :, 0],
            rtol=5e-4,
            atol=1e-10,
        )

    def test_1d_reconstructed_fields_follow_exp_plus_jwt_maxwell_phase(self):
        frequency = 30e9
        aperture = 10e-3
        aperture_cells = 40
        nx = aperture_cells + 2
        dx = aperture / aperture_cells
        epsilon_r = 2.25 - 0.09j

        solver = ModeSolver1D(
            frequency,
            nx * dx,
            nx,
            num_modes=1,
            guess=-epsilon_r,
        )
        solver.add_layer(epsilon_r, 1.0, (0, nx))
        solver.add_pec((0, 1))
        solver.add_pec((nx - 1, nx))
        solver.solve()

        self.assertFalse(np.shares_memory(solver.Ey, solver.eigenvectors_TE))
        self.assertFalse(np.shares_memory(solver.Hy, solver.eigenvectors_TM))

        np.testing.assert_allclose(
            solver.Hx[:, 0],
            1j * solver.neff_TE[0] / solver.mu_r_xx * solver.Ey[:, 0],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            solver.Ex[:, 0],
            1j * solver.neff_TM[0] / solver.eps_r_xx * solver.Hy[:, 0],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_1d_pml_uses_negative_imaginary_coordinate_stretch(self):
        frequency = 10e9
        sigma_max = 0.25
        solver = ModeSolver1D(frequency, 6e-3, 6, num_modes=1)
        stretch = 1 - 1j * sigma_max / (
            self.EPSILON0 * 2 * np.pi * frequency
        )

        solver.add_pml(
            pml_width=1,
            n=1,
            sigma_max=sigma_max,
            direction="x-",
        )

        self.assertAlmostEqual(solver.cell_eps_r_xx[0], 1 / stretch)
        self.assertAlmostEqual(solver.cell_eps_r_yy[0], stretch)
        self.assertAlmostEqual(solver.cell_eps_r_zz[0], stretch)
        self.assertAlmostEqual(solver.cell_mu_r_xx[0], 1 / stretch)
        self.assertAlmostEqual(solver.cell_mu_r_yy[0], stretch)
        self.assertAlmostEqual(solver.cell_mu_r_zz[0], stretch)
        np.testing.assert_array_equal(solver.cell_eps_r_xx[1:], 1.0)

    def test_2d_pml_uses_negative_imaginary_coordinate_stretches(self):
        frequency = 10e9
        sigma_max = 0.25
        solver = ModeSolver2D(frequency, 4e-3, 4e-3, 4, 4, num_modes=1)
        stretch = 1 - 1j * sigma_max / (
            self.EPSILON0 * 2 * np.pi * frequency
        )

        solver.add_pml(
            pml_width=1,
            n=1,
            sigma_max=sigma_max,
            direction="all",
        )

        # x-only, y-only, and x/y-corner cells exercise every tensor ratio.
        self.assertAlmostEqual(solver.cell_eps_r_xx[0, 2], 1 / stretch)
        self.assertAlmostEqual(solver.cell_eps_r_yy[0, 2], stretch)
        self.assertAlmostEqual(solver.cell_eps_r_zz[0, 2], stretch)
        self.assertAlmostEqual(solver.cell_eps_r_xx[2, 0], stretch)
        self.assertAlmostEqual(solver.cell_eps_r_yy[2, 0], 1 / stretch)
        self.assertAlmostEqual(solver.cell_eps_r_zz[2, 0], stretch)
        self.assertAlmostEqual(solver.cell_eps_r_xx[0, 0], 1.0)
        self.assertAlmostEqual(solver.cell_eps_r_yy[0, 0], 1.0)
        self.assertAlmostEqual(solver.cell_eps_r_zz[0, 0], stretch ** 2)
        np.testing.assert_allclose(solver.cell_mu_r_xx, solver.cell_eps_r_xx)
        np.testing.assert_allclose(solver.cell_mu_r_yy, solver.cell_eps_r_yy)
        np.testing.assert_allclose(solver.cell_mu_r_zz, solver.cell_eps_r_zz)

    def test_negative_or_nonfinite_pml_conductivity_is_rejected(self):
        solvers = (
            ModeSolver1D(10e9, 4e-3, 4, num_modes=1),
            ModeSolver2D(10e9, 4e-3, 4e-3, 4, 4, num_modes=1),
        )
        for solver in solvers:
            for sigma_max in (-1.0, np.inf, np.nan):
                with self.subTest(solver=type(solver).__name__, sigma_max=sigma_max):
                    with self.assertRaisesRegex(
                        ValueError,
                        "sigma_max must be finite and nonnegative",
                    ):
                        solver.add_pml(pml_width=1, sigma_max=sigma_max)


if __name__ == "__main__":
    unittest.main()
