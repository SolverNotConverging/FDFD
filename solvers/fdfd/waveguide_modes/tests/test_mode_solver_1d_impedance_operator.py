import unittest

import numpy as np

from fdfd_waveguide_modes import ModeSolver1D


class ModeSolver1DImpedanceOperatorTests(unittest.TestCase):
    FREQUENCY_HZ = 30e9
    EPSILON0 = 8.854187817e-12
    MU0 = 4e-7 * np.pi
    COPPER_RESISTIVITY_OHM_M = 1.676e-8
    APERTURE_M = 10e-3

    @classmethod
    def setUpClass(cls):
        cls.c = 1 / np.sqrt(cls.EPSILON0 * cls.MU0)
        cls.eta0 = np.sqrt(cls.MU0 / cls.EPSILON0)
        cls.omega = 2 * np.pi * cls.FREQUENCY_HZ
        cls.k0 = cls.omega / cls.c
        cls.copper_resistance = np.sqrt(
            np.pi
            * cls.FREQUENCY_HZ
            * cls.MU0
            * cls.COPPER_RESISTIVITY_OHM_M
        )
        cls.copper_impedance = cls.copper_resistance * (1 + 1j)
        cls.cutoff_wavenumber = np.pi / cls.APERTURE_M
        cls.beta_te1 = np.sqrt(cls.k0 ** 2 - cls.cutoff_wavenumber ** 2)
        cls.neff_te1 = cls.beta_te1 / cls.k0
        cls.tem_perturbation_coefficient = 1 / (cls.eta0 * cls.APERTURE_M)
        cls.te1_perturbation_coefficient = (
            2
            * cls.cutoff_wavenumber ** 2
            / (cls.omega * cls.MU0 * cls.beta_te1 * cls.APERTURE_M)
        )

    def parallel_plate_solver(
        self,
        *,
        aperture_cells=100,
        wall_cells=1,
        impedance=None,
        preset=None,
    ):
        dx = self.APERTURE_M / aperture_cells
        nx = aperture_cells + 2 * wall_cells
        solver = ModeSolver1D(
            frequency=self.FREQUENCY_HZ,
            x_range=nx * dx,
            Nx=nx,
            num_modes=1,
            guess=-1.0,
        )
        kwargs = {"Zs": impedance} if preset is None else {"preset": preset}
        solver.add_impedance_surface(
            **kwargs,
            x_range=(0, wall_cells),
        )
        solver.add_impedance_surface(
            **kwargs,
            x_range=(nx - wall_cells, nx),
        )
        solver.solve()
        return solver

    def test_installed_left_and_right_half_cell_rows_and_material_loads(self):
        nx = 6
        dx = 0.5e-3
        impedance = 25.0 + 7.0j
        solver = ModeSolver1D(
            self.FREQUENCY_HZ,
            nx * dx,
            nx,
            num_modes=1,
        )
        solver.cell_eps_r_yy[:] = np.array([91, 2.5, 3, 4, 4.5, 92])
        solver.cell_eps_r_zz[:] = np.array([81, 3.5, 4, 5, 5.5, 82])
        solver.cell_mu_r_xx[:] = np.array([71, 6, 1, 1, 7, 72])
        solver.add_impedance_surface(impedance, x_range=(0, 1))
        solver.add_impedance_surface(impedance, x_range=(nx - 1, nx))

        materials, *_ = solver._effective_materials_and_masks()
        _, d_h_to_e = solver._yeeder1d()

        surface_load = 2 / (
            1j * self.omega * self.EPSILON0 * dx * impedance
        )
        self.assertAlmostEqual(materials["eps_yy"][1], 2.5 + surface_load)
        self.assertAlmostEqual(materials["eps_zz"][1], 3.5 + surface_load)
        self.assertAlmostEqual(materials["eps_yy"][nx - 1], 4.5 + surface_load)
        self.assertAlmostEqual(materials["eps_zz"][nx - 1], 5.5 + surface_load)

        # Normal permeability must be sampled solely from the retained cell,
        # never averaged with the arbitrary material stored in an opaque cell.
        self.assertEqual(materials["mu_xx"][1], 6.0)
        self.assertEqual(materials["mu_xx"][nx - 1], 7.0)

        left_row = d_h_to_e.getrow(1)
        right_row = d_h_to_e.getrow(nx - 1)
        self.assertEqual(left_row.indices.tolist(), [1])
        self.assertEqual(right_row.indices.tolist(), [nx - 2])
        self.assertAlmostEqual(left_row.data[0], 2 / (self.k0 * dx))
        self.assertAlmostEqual(right_row.data[0], -2 / (self.k0 * dx))

    def test_copper_parallel_plate_attenuation_and_phase_match_perturbation(self):
        copper = self.parallel_plate_solver(preset="Cu")
        resistive_reference = self.parallel_plate_solver(
            impedance=self.copper_resistance
        )

        expected_alpha_tm = (
            self.tem_perturbation_coefficient * self.copper_resistance
        )
        expected_alpha_te = (
            self.te1_perturbation_coefficient * self.copper_resistance
        )
        calculated_alpha_tm = self.k0 * copper.attenuation_constant_TM[0]
        calculated_alpha_te = self.k0 * copper.attenuation_constant_TE[0]

        self.assertLess(copper.neff_TM[0].imag, 0.0)
        self.assertLess(copper.neff_TE[0].imag, 0.0)
        self.assertAlmostEqual(
            calculated_alpha_tm,
            expected_alpha_tm,
            delta=0.01 * expected_alpha_tm,
        )
        self.assertAlmostEqual(
            calculated_alpha_te,
            expected_alpha_te,
            delta=0.01 * expected_alpha_te,
        )

        calculated_phase_tm = self.k0 * (
            copper.neff_TM[0].real - resistive_reference.neff_TM[0].real
        )
        calculated_phase_te = self.k0 * (
            copper.neff_TE[0].real - resistive_reference.neff_TE[0].real
        )
        expected_phase_tm = (
            self.tem_perturbation_coefficient * self.copper_impedance.imag
        )
        expected_phase_te = (
            self.te1_perturbation_coefficient * self.copper_impedance.imag
        )
        self.assertAlmostEqual(
            calculated_phase_tm,
            expected_phase_tm,
            delta=0.01 * expected_phase_tm,
        )
        self.assertAlmostEqual(
            calculated_phase_te,
            expected_phase_te,
            delta=0.01 * expected_phase_te,
        )

    def test_purely_reactive_walls_are_lossless_and_match_phase_perturbation(self):
        reactance = self.copper_resistance
        inductive = self.parallel_plate_solver(impedance=1j * reactance)
        capacitive = self.parallel_plate_solver(impedance=-1j * reactance)

        for solver in (inductive, capacitive):
            self.assertAlmostEqual(solver.attenuation_constant_TM[0], 0.0, delta=1e-10)
            self.assertAlmostEqual(solver.attenuation_constant_TE[0], 0.0, delta=1e-10)

        calculated_phase_tm = self.k0 * (
            inductive.neff_TM[0].real - capacitive.neff_TM[0].real
        ) / 2
        calculated_phase_te = self.k0 * (
            inductive.neff_TE[0].real - capacitive.neff_TE[0].real
        ) / 2
        expected_phase_tm = self.tem_perturbation_coefficient * reactance
        expected_phase_te = self.te1_perturbation_coefficient * reactance
        self.assertAlmostEqual(
            calculated_phase_tm,
            expected_phase_tm,
            delta=0.01 * expected_phase_tm,
        )
        self.assertAlmostEqual(
            calculated_phase_te,
            expected_phase_te,
            delta=0.01 * expected_phase_te,
        )

    def test_opaque_wall_thickness_does_not_change_parallel_plate_modes(self):
        thin = self.parallel_plate_solver(
            aperture_cells=80,
            wall_cells=1,
            preset="copper",
        )
        thick = self.parallel_plate_solver(
            aperture_cells=80,
            wall_cells=4,
            preset="copper",
        )

        self.assertGreater(
            np.count_nonzero(thick.impedance_surface_mask),
            np.count_nonzero(thin.impedance_surface_mask),
        )
        np.testing.assert_allclose(
            thick.neff_TE,
            thin.neff_TE,
            rtol=1e-9,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            thick.neff_TM,
            thin.neff_TM,
            rtol=1e-9,
            atol=1e-11,
        )


if __name__ == "__main__":
    unittest.main()
