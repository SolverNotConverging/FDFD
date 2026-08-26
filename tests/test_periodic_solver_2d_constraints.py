import importlib
import unittest
from unittest.mock import patch

import numpy as np

from Periodic_Solver_2D import PeriodicModeSolver2D


periodic_solver_2d_module = importlib.import_module(
    "Periodic_Solver_2D.Periodic_Solver_2D"
)


class PeriodicModeSolver2DConstraintTests(unittest.TestCase):
    def make_solver(self, polarization="TM"):
        return PeriodicModeSolver2D(
            polarization,
            freq=10e9,
            x_range=1.0,
            z_range=1.0,
            Nx=4,
            Nz=3,
            num_modes=1,
            mode_filter=False,
            guess=0.0,
        )

    @staticmethod
    def effective_masks(solver):
        return solver._effective_materials_and_masks()[6:]

    @staticmethod
    def deterministic_eigs(captured):
        def fake_eigs(A, M, *, k, sigma, tol, ncv, v0):
            captured["A"] = A
            captured["B"] = M
            captured["v0"] = v0
            vector = np.arange(1, A.shape[0] + 1, dtype=float).reshape(-1, 1)
            return np.array([2.0 + 0.0j]), vector.astype(complex)

        return fake_eigs

    def test_x_normal_pmc_face_constrains_tangential_h_and_normal_e_only(self):
        solver = self.make_solver()
        solver.add_pmc((1, 2), (0, solver.Nz))

        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = (
            self.effective_masks(solver)
        )

        # A one-cell-thick slab spanning periodic z has only x-normal faces.
        # On either face PMC requires Hy = Hz = Ex = 0.  Hx is normal and
        # Ey/Ez are tangential electric components, so those DOFs stay free.
        np.testing.assert_array_equal(pec_xx[1, :], True)
        np.testing.assert_array_equal(pmc_yy[1, :], True)
        np.testing.assert_array_equal(pmc_zz[1, :], True)
        np.testing.assert_array_equal(pmc_xx[1:3, :], False)
        np.testing.assert_array_equal(pec_yy[1:3, :], False)
        np.testing.assert_array_equal(pec_zz[1:3, :], False)

    def test_z_normal_pmc_face_constrains_tangential_h_and_normal_e_only(self):
        solver = self.make_solver()
        solver.add_pmc((0, solver.Nx), (0, 1))

        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = (
            self.effective_masks(solver)
        )
        source = solver._pmc_cell_masks["zz"]
        z_faces = np.flatnonzero(solver._periodic_z_interface_mask(source)[1])

        self.assertGreater(z_faces.size, 0)
        np.testing.assert_array_equal(pmc_xx[1, z_faces], True)
        np.testing.assert_array_equal(pmc_yy[1, z_faces], True)
        np.testing.assert_array_equal(pec_zz[1, z_faces], True)
        np.testing.assert_array_equal(pmc_zz[1, z_faces], False)
        np.testing.assert_array_equal(pec_xx[1, z_faces], False)
        np.testing.assert_array_equal(pec_yy[1, z_faces], False)

    def test_x_z_pmc_corner_unions_face_constraints(self):
        solver = self.make_solver()
        solver.add_pmc((1, 2), (0, 1))

        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = (
            self.effective_masks(solver)
        )

        # At an x/z corner, Hx is tangential to z and Hz is tangential to x;
        # Ex and Ez are each normal to one face.  Ey is tangential to both.
        self.assertTrue(pmc_xx[1, 0])
        self.assertTrue(pmc_yy[1, 0])
        self.assertTrue(pmc_zz[1, 0])
        self.assertTrue(pec_xx[1, 0])
        self.assertTrue(pec_zz[1, 0])
        self.assertFalse(pec_yy[1, 0])

    def test_finite_z_pec_uses_normal_hz_without_zeroing_tangential_hy(self):
        solver = self.make_solver()
        solver.add_pec((1, 2), (0, 1), components=("xx",))

        pec_xx, _pec_yy, _pec_zz, _pmc_xx, pmc_yy, pmc_zz = (
            self.effective_masks(solver)
        )
        source = solver._pec_cell_masks["xx"]
        z_ex = solver._z_interface_component_masks(source, field="electric")[0]

        self.assertTrue(np.any(z_ex))
        np.testing.assert_array_equal(pec_xx[z_ex], True)
        np.testing.assert_array_equal(pmc_yy[z_ex], False)
        np.testing.assert_array_equal(
            pmc_zz[solver._periodic_z_interface_mask(source)],
            True,
        )

    def test_z_normal_pec_face_constrains_tangential_e_and_normal_h_only(self):
        solver = self.make_solver()
        solver.add_pec((0, solver.Nx), (0, 1))

        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = (
            self.effective_masks(solver)
        )
        source = solver._pec_cell_masks["zz"]
        z_faces = np.flatnonzero(solver._periodic_z_interface_mask(source)[1])

        self.assertGreater(z_faces.size, 0)
        np.testing.assert_array_equal(pec_xx[1, z_faces], True)
        np.testing.assert_array_equal(pec_yy[1, z_faces], True)
        np.testing.assert_array_equal(pmc_zz[1, z_faces], True)
        np.testing.assert_array_equal(pmc_xx[1, z_faces], False)
        np.testing.assert_array_equal(pmc_yy[1, z_faces], False)
        np.testing.assert_array_equal(pec_zz[1, z_faces], False)

    def test_x_normal_pec_face_constrains_tangential_e_and_normal_h_only(self):
        solver = self.make_solver()
        solver.add_pec((1, 2), (0, solver.Nz))

        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = (
            self.effective_masks(solver)
        )

        # On either x-normal face PEC requires Ey = Ez = Hx = 0.  Ex is
        # normal and Hy/Hz are tangential magnetic components, so they stay
        # free in the face-adjacent boundary cells.
        np.testing.assert_array_equal(pec_yy[1:3, :], True)
        np.testing.assert_array_equal(pec_zz[1:3, :], True)
        np.testing.assert_array_equal(pmc_xx[1:3, :], True)
        np.testing.assert_array_equal(pec_xx[1, :], False)
        np.testing.assert_array_equal(pmc_yy[1, :], False)
        np.testing.assert_array_equal(pmc_zz[1, :], False)

    def test_x_z_pec_corner_unions_face_constraints(self):
        solver = self.make_solver()
        solver.add_pec((1, 2), (0, 1))

        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = (
            self.effective_masks(solver)
        )

        # Ex is tangential to z, Ez is tangential to x, Hx is normal to x,
        # and Hz is normal to z.  Hy is tangential magnetic on both faces.
        self.assertTrue(pec_xx[1, 0])
        self.assertTrue(pec_yy[1, 0])
        self.assertTrue(pec_zz[1, 0])
        self.assertTrue(pmc_xx[1, 0])
        self.assertFalse(pmc_yy[1, 0])
        self.assertTrue(pmc_zz[1, 0])

    def test_finite_z_pmc_uses_normal_ez_without_zeroing_tangential_ey(self):
        solver = self.make_solver("TE")
        solver.add_pmc((1, 2), (0, 1), components=("xx",))

        _pec_xx, pec_yy, pec_zz, pmc_xx, _pmc_yy, _pmc_zz = (
            self.effective_masks(solver)
        )
        source = solver._pmc_cell_masks["xx"]
        z_hx = solver._z_interface_component_masks(source, field="magnetic")[0]
        direct_hx = solver.component_masks_from_cell_mask(source, field="magnetic")[0]
        z_only = z_hx & ~direct_hx

        self.assertTrue(np.any(z_only))
        np.testing.assert_array_equal(pmc_xx[z_only], True)
        np.testing.assert_array_equal(pec_yy[z_only], False)
        np.testing.assert_array_equal(
            pec_zz[solver._z_interface_component_masks(source, field="electric")[2]],
            True,
        )

    def test_adjacent_periodic_regions_do_not_leave_a_false_z_interface(self):
        solver = self.make_solver()
        solver.add_pec((1, 2), (0, 1), components=("xx",))
        solver.add_pec((1, 2), (1, solver.Nz), components=("xx",))

        pec_xx, _pec_yy, _pec_zz, _pmc_xx, pmc_yy, pmc_zz = (
            self.effective_masks(solver)
        )
        np.testing.assert_array_equal(pec_xx, pmc_yy)
        self.assertFalse(np.any(pmc_zz))

    def test_exact_boundaries_do_not_modify_material_tensors(self):
        solver = self.make_solver()
        solver.add_pec((1, 2), (0, solver.Nz))
        solver.add_pmc((2, 3), (0, solver.Nz))

        for values in (
            solver.cell_eps_r_xx,
            solver.cell_eps_r_yy,
            solver.cell_eps_r_zz,
            solver.cell_mu_r_xx,
            solver.cell_mu_r_yy,
            solver.cell_mu_r_zz,
        ):
            np.testing.assert_array_equal(values, np.ones_like(values))

    def test_single_cell_at_periodic_seam_uses_wrapped_yee_footprints(self):
        solver = self.make_solver()
        cell_mask = np.zeros(solver.shape_cell, dtype=bool)
        cell_mask[1, 0] = True

        ex, ey, ez = solver.component_masks_from_cell_mask(
            cell_mask, field="electric"
        )
        hx, hy, hz = solver.component_masks_from_cell_mask(
            cell_mask, field="magnetic"
        )

        expected_ex_hy = np.zeros(solver.shape_ex, dtype=bool)
        expected_ex_hy[1, 0] = True
        expected_ex_hy[1, solver.Nz - 1] = True
        expected_ey_ez_hx = np.zeros(solver.shape_ey, dtype=bool)
        expected_ey_ez_hx[1, 0] = True
        expected_ey_ez_hx[2, 0] = True
        expected_hz = np.zeros(solver.shape_hz, dtype=bool)
        expected_hz[1, 0] = True

        np.testing.assert_array_equal(ex, expected_ex_hy)
        np.testing.assert_array_equal(hy, expected_ex_hy)
        np.testing.assert_array_equal(ey, expected_ey_ez_hx)
        np.testing.assert_array_equal(ez, expected_ey_ez_hx)
        np.testing.assert_array_equal(hx, expected_ey_ez_hx)
        np.testing.assert_array_equal(hz, expected_hz)

    def test_tm_solve_reduces_both_matrices_and_restores_exact_zeros(self):
        solver = self.make_solver("TM")
        solver.add_pec((1, 2), (0, solver.Nz), components=("xx",))
        pec_xx, _pec_yy, _pec_zz, _pmc_xx, pmc_yy, _pmc_zz = (
            self.effective_masks(solver)
        )
        free = np.concatenate(
            (~pec_xx.ravel(order="F"), ~pmc_yy.ravel(order="F"))
        )
        captured = {}

        with patch.object(
            periodic_solver_2d_module,
            "eigs",
            side_effect=self.deterministic_eigs(captured),
        ):
            solver.solve()

        reduced_size = int(np.count_nonzero(free))
        self.assertEqual(captured["A"].shape, (reduced_size, reduced_size))
        self.assertEqual(captured["B"].shape, (reduced_size, reduced_size))
        self.assertEqual(captured["v0"].shape, (reduced_size,))

        expected = np.zeros(free.size, dtype=complex)
        expected[free] = np.arange(1, reduced_size + 1, dtype=float)
        np.testing.assert_array_equal(solver.eigenvectors[:, 0], expected)
        np.testing.assert_array_equal(
            solver.Ex[pec_xx.ravel(order="F"), 0],
            0.0,
        )
        np.testing.assert_array_equal(
            solver.Hy[pmc_yy.ravel(order="F"), 0],
            0.0,
        )

    def test_te_solve_reduces_both_matrices_and_restores_exact_zeros(self):
        solver = self.make_solver("TE")
        solver.add_pmc((1, 2), (0, solver.Nz), components=("xx",))
        _pec_xx, pec_yy, _pec_zz, pmc_xx, _pmc_yy, _pmc_zz = (
            self.effective_masks(solver)
        )
        free = np.concatenate(
            (~pmc_xx.ravel(order="F"), ~pec_yy.ravel(order="F"))
        )
        captured = {}

        with patch.object(
            periodic_solver_2d_module,
            "eigs",
            side_effect=self.deterministic_eigs(captured),
        ):
            solver.solve()

        reduced_size = int(np.count_nonzero(free))
        self.assertEqual(captured["A"].shape, (reduced_size, reduced_size))
        self.assertEqual(captured["B"].shape, (reduced_size, reduced_size))

        expected = np.zeros(free.size, dtype=complex)
        expected[free] = np.arange(1, reduced_size + 1, dtype=float)
        np.testing.assert_array_equal(solver.eigenvectors[:, 0], expected)
        np.testing.assert_array_equal(
            solver.Hx[pmc_xx.ravel(order="F"), 0],
            0.0,
        )
        np.testing.assert_array_equal(
            solver.Ey[pec_yy.ravel(order="F"), 0],
            0.0,
        )

    def test_spatial_eigenvalue_maps_to_exp_plus_jwt_effective_index(self):
        solver = self.make_solver("TM")
        gamma = solver.k0 * (0.02 + 1.5j)

        def fake_eigs(A, M, *, k, sigma, tol, ncv, v0):
            vector = np.ones((A.shape[0], 1), dtype=complex)
            return np.array([gamma]), vector

        with patch.object(periodic_solver_2d_module, "eigs", side_effect=fake_eigs):
            solver.solve()

        self.assertAlmostEqual(solver.eigenvalues[0], gamma)
        self.assertAlmostEqual(solver.gammas[0], 0.02 + 1.5j)
        self.assertAlmostEqual(solver.neff[0], 1.5 - 0.02j)
        self.assertAlmostEqual(solver.propagation_constant[0], 1.5)
        self.assertAlmostEqual(solver.attenuation_constant[0], 0.02)
        self.assertLess(solver.neff[0].imag, 0.0)

    def test_pml_stretch_uses_exp_plus_jwt_passive_sign(self):
        solver = self.make_solver("TM")
        sigma = 0.25 * solver.epsilon0 * solver.omega
        stretch = 1.0 - 0.25j

        solver.add_pml(pml_width=1, sigma_max=sigma, direction="x-")

        np.testing.assert_allclose(solver.cell_eps_r_xx[0, :], 1.0 / stretch)
        np.testing.assert_allclose(solver.cell_eps_r_yy[0, :], stretch)
        np.testing.assert_allclose(solver.cell_eps_r_zz[0, :], stretch)
        np.testing.assert_allclose(solver.cell_mu_r_xx[0, :], 1.0 / stretch)
        np.testing.assert_allclose(solver.cell_mu_r_yy[0, :], stretch)
        np.testing.assert_allclose(solver.cell_mu_r_zz[0, :], stretch)
        np.testing.assert_array_equal(
            solver.cell_eps_r_yy[1:, :],
            np.ones_like(solver.cell_eps_r_yy[1:, :]),
        )

    def test_pml_rejects_active_or_nonfinite_loss(self):
        for sigma_max in (-1.0, np.inf, np.nan):
            with self.subTest(sigma_max=sigma_max):
                solver = self.make_solver("TM")
                with self.assertRaisesRegex(
                    ValueError,
                    "sigma_max must be finite and nonnegative",
                ):
                    solver.add_pml(pml_width=1, sigma_max=sigma_max)

    def test_passive_material_uses_negative_imaginary_constitutive_values(self):
        solver = self.make_solver("TM")
        epsilon = 2.0 - 0.2j
        mu = 1.0 - 0.1j

        solver.add_rectangle(
            epsilon,
            mu,
            (0, solver.Nx),
            (0, solver.Nz),
            subpixels=1,
        )

        np.testing.assert_array_equal(
            solver.cell_eps_r_xx,
            np.full(solver.shape_cell, epsilon),
        )
        np.testing.assert_array_equal(
            solver.cell_mu_r_xx,
            np.full(solver.shape_cell, mu),
        )

    def test_uniform_passive_medium_has_negative_imaginary_neff(self):
        solver = self.make_solver("TE")
        epsilon = 2.0 - 0.1j
        expected_neff = np.sqrt(epsilon)
        solver.add_rectangle(
            epsilon,
            1.0,
            (0, solver.Nx),
            (0, solver.Nz),
            subpixels=1,
        )
        solver.guess = 1j * solver.k0 * expected_neff

        solver.solve()

        self.assertAlmostEqual(solver.neff[0], expected_neff, delta=1e-10)
        self.assertLess(solver.neff[0].imag, 0.0)
        self.assertAlmostEqual(
            solver.attenuation_constant[0],
            -expected_neff.imag,
            delta=1e-10,
        )

    def test_inverse_diagonal_is_exactly_zero_on_longitudinal_constraints(self):
        solver = self.make_solver()
        values = np.full(solver.shape_ez, 4.0 + 0.0j)
        constrained = np.zeros(solver.shape_ez, dtype=bool)
        constrained[1, 1] = True

        inverse = solver._inverse_diag_on_free(values, constrained).diagonal()
        constrained_flat = constrained.ravel(order="F")

        np.testing.assert_array_equal(inverse[constrained_flat], 0.0)
        np.testing.assert_array_equal(inverse[~constrained_flat], 0.25)

    def test_tm_and_te_pass_longitudinal_masks_to_the_schur_inverse(self):
        cases = (
            ("TM", "pec"),
            ("TE", "pmc"),
        )

        for polarization, boundary in cases:
            with self.subTest(polarization=polarization, boundary=boundary):
                solver = self.make_solver(polarization)
                add_boundary = solver.add_pec if boundary == "pec" else solver.add_pmc
                add_boundary((1, 2), (0, solver.Nz), components=("zz",))
                captured = {}

                with patch.object(
                    solver,
                    "_inverse_diag_on_free",
                    wraps=solver._inverse_diag_on_free,
                ) as inverse_mock, patch.object(
                    periodic_solver_2d_module,
                    "eigs",
                    side_effect=self.deterministic_eigs(captured),
                ):
                    solver.solve()

                inverse_mock.assert_called_once()
                constrained = inverse_mock.call_args.args[1]
                self.assertTrue(np.any(constrained))
                diagonal = solver._inverse_diag_on_free(
                    inverse_mock.call_args.args[0], constrained
                ).diagonal()
                np.testing.assert_array_equal(
                    diagonal[constrained.ravel(order="F")],
                    0.0,
                )

    def test_fully_constrained_domain_is_rejected_before_eigensolve(self):
        solver = self.make_solver("TM")
        solver.add_pec((0, solver.Nx), (0, solver.Nz))
        solver.add_pmc((0, solver.Nx), (0, solver.Nz))

        with patch.object(periodic_solver_2d_module, "eigs") as eigs_mock:
            with self.assertRaisesRegex(ValueError, "Not enough unconstrained DOFs"):
                solver.solve()

        eigs_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
