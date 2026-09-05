import unittest

import numpy as np

from fdfd_waveguide_modes.solver_2d import _ModeSolver2D as ModeSolver2D


class ModeSolver2DCrossConstraintTests(unittest.TestCase):
    def setUp(self):
        self.solver = ModeSolver2D(10e9, 1.0, 1.0, 4, 3, num_modes=1)

    def effective_masks(self):
        return self.solver._effective_materials_and_masks()[1:]

    def test_pec_ex_implies_collocated_pmc_hy(self):
        self.solver.add_pec((1, 2), (1, 2), components=("xx",))

        pec_xx, pec_yy, _pec_zz, pmc_xx, pmc_yy, _pmc_zz = self.effective_masks()

        np.testing.assert_array_equal(pmc_yy, pec_xx)
        self.assertTrue(np.any(pec_xx))
        self.assertFalse(np.any(pec_yy))
        self.assertFalse(np.any(pmc_xx))

    def test_pec_ey_implies_collocated_pmc_hx(self):
        self.solver.add_pec((1, 2), (1, 2), components=("yy",))

        pec_xx, pec_yy, _pec_zz, pmc_xx, pmc_yy, _pmc_zz = self.effective_masks()

        np.testing.assert_array_equal(pmc_xx, pec_yy)
        self.assertTrue(np.any(pec_yy))
        self.assertFalse(np.any(pec_xx))
        self.assertFalse(np.any(pmc_yy))

    def test_pmc_hx_implies_collocated_pec_ey(self):
        self.solver.add_pmc((1, 2), (1, 2), components=("xx",))

        _pec_xx, pec_yy, _pec_zz, pmc_xx, _pmc_yy, _pmc_zz = self.effective_masks()

        np.testing.assert_array_equal(pec_yy, pmc_xx)
        self.assertTrue(np.any(pmc_xx))

    def test_pmc_hy_implies_collocated_pec_ex(self):
        self.solver.add_pmc((1, 2), (1, 2), components=("yy",))

        pec_xx, _pec_yy, _pec_zz, _pmc_xx, pmc_yy, _pmc_zz = self.effective_masks()

        np.testing.assert_array_equal(pec_xx, pmc_yy)
        self.assertTrue(np.any(pmc_yy))

    def test_nonfinite_material_constraints_are_cross_constrained(self):
        self.solver.cell_eps_r_xx[1, 1] = np.inf
        self.solver.cell_mu_r_xx[2, 1] = np.inf

        pec_xx, pec_yy, _pec_zz, pmc_xx, pmc_yy, _pmc_zz = self.effective_masks()

        np.testing.assert_array_equal(pmc_yy, pec_xx)
        np.testing.assert_array_equal(pec_yy, pmc_xx)
        self.assertTrue(np.any(pec_xx))
        self.assertTrue(np.any(pmc_xx))


if __name__ == "__main__":
    unittest.main()
