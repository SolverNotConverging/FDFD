import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import hankel2


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


from importlib import import_module
SCATTERING_MODULE = import_module("fdfd_scattering.solver_2d")
BAND_MODULE = import_module("fdfd_band_structure.solver_2d")


class ScatteringTimeConventionTests(unittest.TestCase):
    def make_solver(self):
        return SCATTERING_MODULE.ScatteringSolver2D(
            frequency=2.0e9,
            x_range=0.12,
            y_range=0.08,
            Nx=6,
            Ny=4,
        )

    def test_positive_x_plane_wave_uses_negative_spatial_phase(self):
        solver = self.make_solver()
        solver.add_source(src_type="plane_wave", angle_deg=0.0)
        field = solver.source.reshape(solver.Ny, solver.Nx)

        expected_step = np.exp(-1j * solver.k0 * solver.dx)
        np.testing.assert_allclose(field[:, 1:] / field[:, :-1], expected_step)

    def test_point_sources_use_outgoing_hankel_second_kind(self):
        solver = self.make_solver()
        location = (0.013, -0.007)
        radius = np.hypot(solver.X - location[0], solver.Y - location[1])

        solver.add_source(src_type="point", polarization="TE", location=location)
        np.testing.assert_allclose(
            solver.source.reshape(solver.Ny, solver.Nx),
            hankel2(0, solver.k0 * radius),
        )

        solver.add_source(src_type="point", polarization="TM", location=location)
        np.testing.assert_allclose(
            solver.source.reshape(solver.Ny, solver.Nx),
            -1j / 4 * hankel2(0, solver.k0 * radius),
        )

    def test_positive_pml_conductivity_gives_negative_imaginary_stretch(self):
        solver = self.make_solver()
        sigma_max = 3.0
        solver.add_UPML(
            pml_width=1,
            n=2,
            sigma_max=sigma_max,
            direction="x",
        )

        stretch = 1.0 - 1j * sigma_max / (solver.eps0 * solver.omega)
        np.testing.assert_allclose(solver.ERzz[:, (0, -1)], stretch)
        np.testing.assert_allclose(solver.MRzz[:, (0, -1)], stretch)
        np.testing.assert_allclose(solver.ERzz[:, 1:-1], 1.0)
        self.assertLess(stretch.imag, 0.0)

    def test_pml_rejects_active_nonfinite_or_unknown_parameters(self):
        for sigma_max in (-1.0, np.inf, np.nan):
            with self.subTest(sigma_max=sigma_max):
                with self.assertRaisesRegex(
                    ValueError,
                    "sigma_max must be finite and nonnegative",
                ):
                    self.make_solver().add_UPML(
                        pml_width=1,
                        sigma_max=sigma_max,
                    )

        with self.assertRaisesRegex(ValueError, "direction must be one of"):
            self.make_solver().add_UPML(pml_width=1, direction="sideways")


class BandDiagramTimeConventionTests(unittest.TestCase):
    def test_bloch_wrap_uses_negative_spatial_phase(self):
        beta_x = 0.37
        nx = 3
        dx = 0.2
        derivative_x, _, _, _ = BAND_MODULE.yeeder2d(
            [nx, 1],
            [dx, 1.0],
            [1, 0],
            [beta_x, 0.0],
        )

        expected_wrap = np.exp(-1j * beta_x * nx * dx) / dx
        self.assertAlmostEqual(derivative_x[-1, 0], expected_wrap)

    def test_normalised_frequency_preserves_complex_decay_sign(self):
        solver = BAND_MODULE.BandStructureSolver2D(a=2 * np.pi, Nx=2)
        eigenvalues = np.array([4.0 + 0.4j, 4.0 - 0.4j, 4.0 + 0.0j])

        frequencies = solver._normalise_eigenvalues(eigenvalues)

        np.testing.assert_allclose(frequencies, np.sqrt(eigenvalues))
        self.assertTrue(np.iscomplexobj(frequencies))
        self.assertGreater(frequencies[0].imag, 0.0)
        self.assertLess(frequencies[1].imag, 0.0)
        self.assertEqual(frequencies[2].imag, 0.0)

    def test_uniform_passive_medium_has_positive_imaginary_frequency(self):
        solver = BAND_MODULE.BandStructureSolver2D(
            a=1.0,
            Nx=6,
            background_er=2.25 - 0.09j,
        )
        beta_path = np.array([[0.3], [0.2]])

        result = solver.compute_band_structure(
            beta_path,
            num_bands=1,
            polarisations=("TE", "TM"),
            eig_sigma=0.1,
        )

        for polarization in ("TE", "TM"):
            with self.subTest(polarization=polarization):
                self.assertGreater(
                    result.frequencies[polarization][0, 0].imag,
                    0.0,
                )


if __name__ == "__main__":
    unittest.main()
