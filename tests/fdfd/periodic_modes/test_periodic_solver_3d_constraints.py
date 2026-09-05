import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fdfd_periodic_modes.solver_3d import _PeriodicModeSolver3D as PeriodicModeSolver3D


periodic_solver_3d_module = importlib.import_module(
    "fdfd_periodic_modes.solver_3d"
)


class PeriodicModeSolver3DConstraintTests(unittest.TestCase):
    def make_solver(self):
        return PeriodicModeSolver3D(
            Nx=3,
            Ny=3,
            Nz=2,
            x_range=1.0,
            y_range=1.0,
            z_range=1.0,
            freq=10e9,
            num_modes=1,
            sigma_guess=0.0,
            tol=0.0,
            ncv=None,
        )

    @staticmethod
    def effective_materials_and_masks(solver):
        result = solver._effective_materials_and_masks()
        if isinstance(result[0], dict):
            return result[0], result[1:]

        names = ("erxx", "eryy", "erzz", "mrxx", "mryy", "mrzz")
        return dict(zip(names, result[:6])), result[6:]

    @staticmethod
    def free_mask(masks):
        pec_xx, pec_yy, _pec_zz, pmc_xx, pmc_yy, _pmc_zz = masks
        return np.concatenate(
            tuple(
                mask.ravel(order="F")
                for mask in (~pec_xx, ~pec_yy, ~pmc_xx, ~pmc_yy)
            )
        )

    @staticmethod
    def deterministic_vector(size):
        return np.arange(1, size + 1, dtype=float).reshape(-1, 1).astype(complex)

    def test_x_normal_pmc_face_constrains_tangential_h_and_normal_e_only(self):
        solver = self.make_solver()
        solver.add_pmc(
            (1, 2),
            (0, solver.Ny),
            (0, solver.Nz),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks

        # Sample away from the y-domain edges.  The slab spans periodic z, so
        # these samples see an x-normal face only: Hy = Hz = Ex = 0.
        np.testing.assert_array_equal(pec_xx[1, 1, :], True)
        np.testing.assert_array_equal(pmc_yy[1, 1, :], True)
        np.testing.assert_array_equal(pmc_zz[1, 1, :], True)
        np.testing.assert_array_equal(pmc_xx[1:3, 1, :], False)
        np.testing.assert_array_equal(pec_yy[1:3, 1, :], False)
        np.testing.assert_array_equal(pec_zz[1:3, 1, :], False)

    def test_y_normal_pmc_face_constrains_tangential_h_and_normal_e_only(self):
        solver = self.make_solver()
        solver.add_pmc(
            (0, solver.Nx),
            (1, 2),
            (0, solver.Nz),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks

        # Sample away from the x-domain edges: Hx = Hz = Ey = 0, whereas Hy
        # is normal and Ex/Ez are tangential electric components.
        np.testing.assert_array_equal(pmc_xx[1, 1, :], True)
        np.testing.assert_array_equal(pmc_zz[1, 1, :], True)
        np.testing.assert_array_equal(pec_yy[1, 1, :], True)
        np.testing.assert_array_equal(pmc_yy[1, 1:3, :], False)
        np.testing.assert_array_equal(pec_xx[1, 1:3, :], False)
        np.testing.assert_array_equal(pec_zz[1, 1:3, :], False)

    def test_z_normal_pmc_face_constrains_tangential_h_and_normal_e_only(self):
        solver = self.make_solver()
        solver.add_pmc(
            (0, solver.Nx),
            (0, solver.Ny),
            (0, 1),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks
        source = solver._pmc_cell_masks["zz"]
        z_faces = np.flatnonzero(
            solver._periodic_z_interface_mask(source)[1, 1, :]
        )

        self.assertGreater(z_faces.size, 0)
        np.testing.assert_array_equal(pmc_xx[1, 1, z_faces], True)
        np.testing.assert_array_equal(pmc_yy[1, 1, z_faces], True)
        np.testing.assert_array_equal(pec_zz[1, 1, z_faces], True)
        np.testing.assert_array_equal(pmc_zz[1, 1, z_faces], False)
        np.testing.assert_array_equal(pec_xx[1, 1, z_faces], False)
        np.testing.assert_array_equal(pec_yy[1, 1, z_faces], False)

    def test_x_y_pmc_edge_unions_face_constraints(self):
        solver = self.make_solver()
        solver.add_pmc(
            (1, 2),
            (1, 2),
            (0, solver.Nz),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks

        # At the edge Hx is tangential to y, Hy is tangential to x, and Hz is
        # tangential to both.  Ex/Ey are normal to one face; Ez is tangent to
        # both and must remain unconstrained.
        self.assertTrue(pmc_xx[1, 1, 0])
        self.assertTrue(pmc_yy[1, 1, 0])
        self.assertTrue(pmc_zz[1, 1, 0])
        self.assertTrue(pec_xx[1, 1, 0])
        self.assertTrue(pec_yy[1, 1, 0])
        self.assertFalse(pec_zz[1, 1, 0])

    def test_x_y_z_pmc_corner_constrains_all_six_components(self):
        solver = self.make_solver()
        solver.add_pmc(
            (1, 2),
            (1, 2),
            (0, 1),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks

        # Every H component is tangential to at least one incident face, and
        # every E component is normal to one incident face.
        self.assertTrue(pmc_xx[1, 1, 0])
        self.assertTrue(pmc_yy[1, 1, 0])
        self.assertTrue(pmc_zz[1, 1, 0])
        self.assertTrue(pec_xx[1, 1, 0])
        self.assertTrue(pec_yy[1, 1, 0])
        self.assertTrue(pec_zz[1, 1, 0])

    def test_finite_z_pec_uses_normal_hz_without_zeroing_tangential_hy(self):
        solver = self.make_solver()
        solver.add_pec((1, 2), (1, 2), (0, 1), components=("xx",))

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, _pec_yy, _pec_zz, _pmc_xx, pmc_yy, pmc_zz = masks
        source = solver._pec_cell_masks["xx"]
        z_ex = solver._z_interface_component_masks(source, field="electric")[0]
        direct_ex = solver.component_masks_from_cell_mask(source, field="electric")[0]
        z_only = z_ex & ~direct_ex

        self.assertTrue(np.any(z_only))
        np.testing.assert_array_equal(pec_xx[z_only], True)
        np.testing.assert_array_equal(pmc_yy[z_only], False)
        np.testing.assert_array_equal(
            pmc_zz[solver._periodic_z_interface_mask(source)],
            True,
        )

    def test_z_normal_pec_face_constrains_tangential_e_and_normal_h_only(self):
        solver = self.make_solver()
        solver.add_pec(
            (0, solver.Nx),
            (0, solver.Ny),
            (0, 1),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks
        source = solver._pec_cell_masks["zz"]
        z_faces = np.flatnonzero(
            solver._periodic_z_interface_mask(source)[1, 1, :]
        )

        self.assertGreater(z_faces.size, 0)
        np.testing.assert_array_equal(pec_xx[1, 1, z_faces], True)
        np.testing.assert_array_equal(pec_yy[1, 1, z_faces], True)
        np.testing.assert_array_equal(pmc_zz[1, 1, z_faces], True)
        np.testing.assert_array_equal(pmc_xx[1, 1, z_faces], False)
        np.testing.assert_array_equal(pmc_yy[1, 1, z_faces], False)
        np.testing.assert_array_equal(pec_zz[1, 1, z_faces], False)

    def test_x_normal_pec_face_constrains_tangential_e_and_normal_h_only(self):
        solver = self.make_solver()
        solver.add_pec(
            (1, 2),
            (0, solver.Ny),
            (0, solver.Nz),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks

        # Sample away from y-domain edges.  Ey = Ez = Hx = 0; Ex is normal
        # and Hy/Hz are tangential magnetic components, so they stay free.
        np.testing.assert_array_equal(pec_yy[1:3, 1, :], True)
        np.testing.assert_array_equal(pec_zz[1:3, 1, :], True)
        np.testing.assert_array_equal(pmc_xx[1:3, 1, :], True)
        np.testing.assert_array_equal(pec_xx[1, 1, :], False)
        np.testing.assert_array_equal(pmc_yy[1, 1, :], False)
        np.testing.assert_array_equal(pmc_zz[1, 1, :], False)

    def test_y_normal_pec_face_constrains_tangential_e_and_normal_h_only(self):
        solver = self.make_solver()
        solver.add_pec(
            (0, solver.Nx),
            (1, 2),
            (0, solver.Nz),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks

        # Sample away from x-domain edges.  Ex = Ez = Hy = 0; Ey is normal
        # and Hx/Hz are tangential magnetic components, so they stay free.
        np.testing.assert_array_equal(pec_xx[1, 1:3, :], True)
        np.testing.assert_array_equal(pec_zz[1, 1:3, :], True)
        np.testing.assert_array_equal(pmc_yy[1, 1:3, :], True)
        np.testing.assert_array_equal(pec_yy[1, 1, :], False)
        np.testing.assert_array_equal(pmc_xx[1, 1, :], False)
        np.testing.assert_array_equal(pmc_zz[1, 1, :], False)

    def test_x_y_pec_edge_unions_face_constraints(self):
        solver = self.make_solver()
        solver.add_pec(
            (1, 2),
            (1, 2),
            (0, solver.Nz),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks

        # Ex/Ey are tangential to one incident face, Ez to both; Hx/Hy are
        # each normal to one face.  Hz is tangential magnetic to both.
        self.assertTrue(pec_xx[1, 1, 0])
        self.assertTrue(pec_yy[1, 1, 0])
        self.assertTrue(pec_zz[1, 1, 0])
        self.assertTrue(pmc_xx[1, 1, 0])
        self.assertTrue(pmc_yy[1, 1, 0])
        self.assertFalse(pmc_zz[1, 1, 0])

    def test_x_y_z_pec_corner_constrains_all_six_components(self):
        solver = self.make_solver()
        solver.add_pec(
            (1, 2),
            (1, 2),
            (0, 1),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, pec_yy, pec_zz, pmc_xx, pmc_yy, pmc_zz = masks

        # Every E component is tangential to at least one incident face, and
        # every H component is normal to one incident face.
        self.assertTrue(pec_xx[1, 1, 0])
        self.assertTrue(pec_yy[1, 1, 0])
        self.assertTrue(pec_zz[1, 1, 0])
        self.assertTrue(pmc_xx[1, 1, 0])
        self.assertTrue(pmc_yy[1, 1, 0])
        self.assertTrue(pmc_zz[1, 1, 0])

    def test_finite_z_pmc_uses_normal_ez_without_zeroing_tangential_ey(self):
        solver = self.make_solver()
        solver.add_pmc((1, 2), (1, 2), (0, 1), components=("xx",))

        _materials, masks = self.effective_materials_and_masks(solver)
        _pec_xx, pec_yy, pec_zz, pmc_xx, _pmc_yy, _pmc_zz = masks
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
        solver.add_pec((1, 2), (1, 2), (0, 1), components=("xx",))
        solver.add_pec(
            (1, 2),
            (1, 2),
            (1, solver.Nz),
            components=("xx",),
        )

        _materials, masks = self.effective_materials_and_masks(solver)
        pec_xx, _pec_yy, _pec_zz, _pmc_xx, pmc_yy, pmc_zz = masks
        np.testing.assert_array_equal(pec_xx, pmc_yy)
        self.assertFalse(np.any(pmc_zz))

    def test_exact_boundaries_do_not_modify_material_tensors(self):
        solver = self.make_solver()
        solver.add_pec((1, 2), (1, 2), (0, solver.Nz))
        solver.add_pmc((0, 1), (0, 1), (0, solver.Nz))

        for values in (
            solver.cell_Erxx_3D,
            solver.cell_Eryy_3D,
            solver.cell_Erzz_3D,
            solver.cell_Mrxx_3D,
            solver.cell_Mryy_3D,
            solver.cell_Mrzz_3D,
        ):
            np.testing.assert_array_equal(values, np.ones_like(values))

    def test_single_cell_masks_follow_yee_component_footprints(self):
        solver = self.make_solver()
        cell_mask = np.zeros(solver.shape_cell, dtype=bool)
        cell_mask[1, 1, 0] = True

        ex, ey, ez = solver.component_masks_from_cell_mask(
            cell_mask, field="electric"
        )
        hx, hy, hz = solver.component_masks_from_cell_mask(
            cell_mask, field="magnetic"
        )

        expected_ex_hy = np.zeros(solver.shape_ex, dtype=bool)
        expected_ex_hy[1, 1, 0] = True
        expected_ex_hy[1, 2, 0] = True
        expected_ey_hx = np.zeros(solver.shape_ey, dtype=bool)
        expected_ey_hx[1, 1, 0] = True
        expected_ey_hx[2, 1, 0] = True
        expected_ez = np.zeros(solver.shape_ez, dtype=bool)
        expected_ez[1:3, 1:3, 0] = True
        expected_hz = np.zeros(solver.shape_hz, dtype=bool)
        expected_hz[1, 1, 0] = True

        np.testing.assert_array_equal(ex, expected_ex_hy)
        np.testing.assert_array_equal(hy, expected_ex_hy)
        np.testing.assert_array_equal(ey, expected_ey_hx)
        np.testing.assert_array_equal(hx, expected_ey_hx)
        np.testing.assert_array_equal(ez, expected_ez)
        np.testing.assert_array_equal(hz, expected_hz)

    def test_eigs_reduces_both_matrices_and_restores_exact_zeros(self):
        solver = self.make_solver()
        solver.add_pec(
            (1, 2),
            (1, 2),
            (0, solver.Nz),
            components=("xx",),
        )
        _materials, masks = self.effective_materials_and_masks(solver)
        free = self.free_mask(masks)
        captured = {}

        def fake_eigs(A, M, *, k, sigma, tol, ncv):
            captured["A"] = A
            captured["B"] = M
            return np.array([2.0 + 0.0j]), self.deterministic_vector(A.shape[0])

        with patch.object(
            periodic_solver_3d_module, "eigs", side_effect=fake_eigs
        ):
            solver.solve(method="eigs")

        reduced_size = int(np.count_nonzero(free))
        self.assertEqual(captured["A"].shape, (reduced_size, reduced_size))
        self.assertEqual(captured["B"].shape, (reduced_size, reduced_size))

        expected = np.zeros(free.size, dtype=complex)
        expected[free] = np.arange(1, reduced_size + 1, dtype=float)
        np.testing.assert_array_equal(solver.eigenvectors[:, 0], expected)

        pec_xx, _pec_yy, _pec_zz, _pmc_xx, pmc_yy, _pmc_zz = masks
        np.testing.assert_array_equal(solver.fields["Ex"][0][pec_xx], 0.0)
        np.testing.assert_array_equal(solver.fields["Hy"][0][pmc_yy], 0.0)

    def test_refined_reduces_both_matrices_and_restores_exact_zeros(self):
        solver = self.make_solver()
        solver.add_pmc(
            (1, 2),
            (1, 2),
            (0, solver.Nz),
            components=("xx",),
        )
        _materials, masks = self.effective_materials_and_masks(solver)
        free = self.free_mask(masks)
        captured = {}

        def fake_refined(
            A,
            B,
            *,
            sigma,
            num_modes,
            tol,
            ncv,
            max_restarts,
            random_seed,
        ):
            captured["A"] = A
            captured["B"] = B
            captured["tol"] = tol
            from types import SimpleNamespace
            return SimpleNamespace(
                eigenvalues=np.array([2.0 + 0.0j]),
                eigenvectors=self.deterministic_vector(A.shape[0]),
                physical_residuals=np.array([0.0]),
                restart_count=0,
            )

        with patch.object(
            periodic_solver_3d_module,
            "solve_generalized",
            side_effect=fake_refined,
        ):
            solver.solve(method="refined")

        reduced_size = int(np.count_nonzero(free))
        self.assertEqual(captured["A"].shape, (reduced_size, reduced_size))
        self.assertEqual(captured["B"].shape, (reduced_size, reduced_size))
        self.assertEqual(captured["tol"], 1e-12)

        expected = np.zeros(free.size, dtype=complex)
        expected[free] = np.arange(1, reduced_size + 1, dtype=float)
        np.testing.assert_array_equal(solver.eigenvectors[:, 0], expected)

        _pec_xx, pec_yy, _pec_zz, pmc_xx, _pmc_yy, _pmc_zz = masks
        np.testing.assert_array_equal(solver.fields["Ey"][0][pec_yy], 0.0)
        np.testing.assert_array_equal(solver.fields["Hx"][0][pmc_xx], 0.0)

    def test_spatial_eigenvalue_maps_to_exp_plus_jwt_effective_index(self):
        solver = self.make_solver()
        gamma = solver.k0 * (0.02 + 1.5j)

        def fake_eigs(A, M, *, k, sigma, tol, ncv):
            return np.array([gamma]), self.deterministic_vector(A.shape[0])

        with patch.object(periodic_solver_3d_module, "eigs", side_effect=fake_eigs):
            solver.solve(method="eigs")

        self.assertAlmostEqual(solver.eigenvalues[0], gamma)
        self.assertAlmostEqual(solver.gammas[0], 0.02 + 1.5j)
        self.assertAlmostEqual(solver.neff[0], 1.5 - 0.02j)
        self.assertAlmostEqual(solver.propagation_constant[0], 1.5)
        self.assertAlmostEqual(solver.attenuation_constant[0], 0.02)
        self.assertLess(solver.neff[0].imag, 0.0)

    def test_upml_stretch_uses_exp_plus_jwt_passive_sign(self):
        solver = self.make_solver()
        sigma = 0.25 * solver.epsilon0 * solver.omega
        stretch = 1.0 - 0.25j

        solver.add_UPML(sides=("-x",), width=1, max_loss=sigma, n=1)

        np.testing.assert_allclose(solver.cell_Erxx_3D[0, :, :], 1.0 / stretch)
        np.testing.assert_allclose(solver.cell_Eryy_3D[0, :, :], stretch)
        np.testing.assert_allclose(solver.cell_Erzz_3D[0, :, :], stretch)
        np.testing.assert_allclose(solver.cell_Mrxx_3D[0, :, :], 1.0 / stretch)
        np.testing.assert_allclose(solver.cell_Mryy_3D[0, :, :], stretch)
        np.testing.assert_allclose(solver.cell_Mrzz_3D[0, :, :], stretch)
        np.testing.assert_array_equal(
            solver.cell_Eryy_3D[1:, :, :],
            np.ones_like(solver.cell_Eryy_3D[1:, :, :]),
        )

    def test_upml_rejects_active_or_nonfinite_loss(self):
        for max_loss in (-1.0, np.inf, np.nan):
            with self.subTest(max_loss=max_loss):
                solver = self.make_solver()
                with self.assertRaisesRegex(
                    ValueError,
                    "max_loss must be finite and nonnegative",
                ):
                    solver.add_UPML(sides=("-x",), width=1, max_loss=max_loss)

    def test_passive_material_uses_negative_imaginary_constitutive_values(self):
        solver = self.make_solver()
        epsilon = 2.0 - 0.2j
        mu = 1.0 - 0.1j

        solver.add_block(
            epsilon,
            mu,
            (0, solver.Nx),
            (0, solver.Ny),
            (0, solver.Nz),
            subpixels=1,
        )

        np.testing.assert_array_equal(
            solver.cell_Erxx_3D,
            np.full(solver.shape_cell, epsilon),
        )
        np.testing.assert_array_equal(
            solver.cell_Mrxx_3D,
            np.full(solver.shape_cell, mu),
        )

    def test_uniform_passive_medium_has_negative_imaginary_neff(self):
        solver = self.make_solver()
        epsilon = 2.0 - 0.1j
        expected_neff = np.sqrt(epsilon)
        solver.add_block(
            epsilon,
            1.0,
            (0, solver.Nx),
            (0, solver.Ny),
            (0, solver.Nz),
            subpixels=1,
        )
        # Stay slightly off the analytically degenerate pole; asking ARPACK
        # to factor exactly at the uniform-medium eigenvalue is numerically
        # singular and can fail depending on the preceding test order.
        solver.sigma_guess = 0.97j * solver.k0 * expected_neff

        solver.solve(method="eigs", ncv=30, tol=1e-10)

        self.assertGreater(solver.neff[0].real, 0.0)
        self.assertLess(solver.neff[0].imag, 0.0)
        self.assertGreater(solver.attenuation_constant[0], 0.0)

    def test_inverse_diagonal_is_exactly_zero_on_longitudinal_constraints(self):
        solver = self.make_solver()

        for shape in (solver.shape_ez, solver.shape_hz):
            with self.subTest(shape=shape):
                values = np.full(shape, 4.0 + 0.0j)
                constrained = np.zeros(shape, dtype=bool)
                constrained[(1,) * len(shape)] = True

                inverse = solver._inverse_diag_on_free(values, constrained).diagonal()
                constrained_flat = constrained.ravel(order="F")

                np.testing.assert_array_equal(inverse[constrained_flat], 0.0)
                np.testing.assert_array_equal(inverse[~constrained_flat], 0.25)

    def test_solve_passes_both_longitudinal_masks_to_the_schur_inverses(self):
        solver = self.make_solver()
        solver.add_pec(
            (1, 2),
            (1, 2),
            (0, solver.Nz),
            components=("zz",),
        )
        solver.add_pmc(
            (0, 1),
            (0, 1),
            (0, solver.Nz),
            components=("zz",),
        )
        _materials, masks = self.effective_materials_and_masks(solver)
        pec_zz = masks[2]
        pmc_zz = masks[5]

        def fake_eigs(A, M, *, k, sigma, tol, ncv):
            return np.array([2.0 + 0.0j]), self.deterministic_vector(A.shape[0])

        with patch.object(
            solver,
            "_inverse_diag_on_free",
            wraps=solver._inverse_diag_on_free,
        ) as inverse_mock, patch.object(
            periodic_solver_3d_module, "eigs", side_effect=fake_eigs
        ):
            solver.solve(method="eigs")

        self.assertEqual(inverse_mock.call_count, 2)
        called_masks = [call.args[1] for call in inverse_mock.call_args_list]
        self.assertTrue(any(np.array_equal(mask, pec_zz) for mask in called_masks))
        self.assertTrue(any(np.array_equal(mask, pmc_zz) for mask in called_masks))
        for call in inverse_mock.call_args_list:
            values, constrained = call.args
            diagonal = solver._inverse_diag_on_free(values, constrained).diagonal()
            np.testing.assert_array_equal(
                diagonal[constrained.ravel(order="F")],
                0.0,
            )

    def test_fully_constrained_domain_is_rejected_before_eigensolve(self):
        solver = self.make_solver()
        solver.add_pec(
            (0, solver.Nx),
            (0, solver.Ny),
            (0, solver.Nz),
        )
        solver.add_pmc(
            (0, solver.Nx),
            (0, solver.Ny),
            (0, solver.Nz),
        )

        with patch.object(periodic_solver_3d_module, "eigs") as eigs_mock:
            with self.assertRaisesRegex(ValueError, "Not enough unconstrained DOFs"):
                solver.solve(method="eigs")

        eigs_mock.assert_not_called()

    def test_save_load_round_trip_preserves_constraint_masks(self):
        solver = self.make_solver()
        solver.add_pec(
            (1, 2),
            (1, 2),
            (0, solver.Nz),
            components=("xx", "zz"),
        )
        solver.add_pmc(
            (0, 1),
            (0, 1),
            (0, solver.Nz),
            components=("yy", "zz"),
        )
        solver.eigenvalues = np.array([2.0 + 0.0j])
        solver._update_propagation_outputs()
        solver.fields = {
            "Ex": np.zeros((1, *solver.shape_ex), dtype=complex),
            "Ey": np.zeros((1, *solver.shape_ey), dtype=complex),
            "Hx": np.zeros((1, *solver.shape_hx), dtype=complex),
            "Hy": np.zeros((1, *solver.shape_hy), dtype=complex),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "periodic_constraints.npz"
            solver.save_results(path)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(
                    saved["time_convention"].item(),
                    "exp(+j*omega*t)",
                )
                self.assertEqual(
                    saved["fourier_transform_convention"].item(),
                    "forward exp(-j*omega*t)",
                )
            loaded = PeriodicModeSolver3D.load_results(path)

        np.testing.assert_array_equal(loaded.eigenvalues, solver.eigenvalues)
        np.testing.assert_array_equal(loaded.gammas, solver.gammas)
        np.testing.assert_array_equal(loaded.neff, solver.neff)
        np.testing.assert_array_equal(
            loaded.propagation_constant,
            solver.propagation_constant,
        )
        np.testing.assert_array_equal(
            loaded.attenuation_constant,
            solver.attenuation_constant,
        )

        for name in (
            "pec_xx_mask",
            "pec_yy_mask",
            "pec_zz_mask",
            "pmc_xx_mask",
            "pmc_yy_mask",
            "pmc_zz_mask",
        ):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(solver, name))
        for component in ("xx", "yy", "zz"):
            np.testing.assert_array_equal(
                loaded._pec_cell_masks[component],
                solver._pec_cell_masks[component],
            )
            np.testing.assert_array_equal(
                loaded._pmc_cell_masks[component],
                solver._pmc_cell_masks[component],
            )

if __name__ == "__main__":
    unittest.main()
