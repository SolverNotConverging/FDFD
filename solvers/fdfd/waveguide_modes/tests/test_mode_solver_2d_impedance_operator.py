import unittest

import numpy as np

from fdfd_waveguide_modes import ModeSolver2D, good_conductor_surface_impedance
from fdfd_waveguide_modes.impedance_2d import (
    SurfaceImpedanceDefinition,
    compile_impedance_boundary,
)


class ModeSolver2DImpedanceOperatorTests(unittest.TestCase):
    FREQUENCY_HZ = 10e9
    EPSILON0 = 8.854187817e-12

    @staticmethod
    def definition(impedance=40.0 + 5.0j):
        impedance = complex(impedance)
        return SurfaceImpedanceDefinition(
            key=("constant", impedance.real, impedance.imag),
            impedance=impedance,
            label="test",
        )

    def compile(self, owner, *, impedance=40.0 + 5.0j, dx=0.7e-3, dy=0.9e-3):
        ones = np.ones(owner.shape, dtype=complex)
        return compile_impedance_boundary(
            owner=owner,
            definitions=(self.definition(impedance),),
            cell_eps_r_xx=ones,
            cell_eps_r_yy=ones,
            cell_eps_r_zz=ones,
            dx=dx,
            dy=dy,
            frequency=self.FREQUENCY_HZ,
            epsilon0=self.EPSILON0,
        )

    def test_flat_wall_has_exact_half_cell_ampere_row(self):
        nx, ny = 4, 4
        dx, dy = 0.7e-3, 0.9e-3
        impedance = 40.0 + 5.0j
        owner = np.full((nx, ny), -1, dtype=np.int32)
        owner[:, 0] = 0

        boundary = self.compile(owner, impedance=impedance, dx=dx, dy=dy)
        row = next(
            row
            for row in boundary.rows
            if row.electric_component == 0 and row.electric_index == (2, 1)
        )

        expected_epsilon = 1.0 + 2.0 / (
            1j * 2 * np.pi * self.FREQUENCY_HZ * self.EPSILON0 * dy * impedance
        )
        self.assertAlmostEqual(row.retained_dual_area, dy / 2)
        self.assertAlmostEqual(row.relative_permittivity, expected_epsilon)
        self.assertEqual(len(row.magnetic_terms), 1)
        self.assertEqual(row.magnetic_terms[0].component, 2)
        self.assertEqual(row.magnetic_terms[0].index, (2, 1))
        self.assertEqual(row.magnetic_terms[0].length, 1.0)

        self.assertFalse(boundary.electric_retained[0][2, 0])
        self.assertTrue(boundary.electric_retained[0][2, 1])
        self.assertFalse(boundary.magnetic_retained[2][2, 0])
        self.assertTrue(boundary.magnetic_retained[2][2, 1])

    def test_all_longitudinal_quadrant_patterns(self):
        dx, dy = 0.8e-3, 1.2e-3
        impedance = 35.0 + 9.0j
        quadrants = ((0, 0), (1, 0), (1, 1), (0, 1))
        diagonal_patterns = {0b0101, 0b1010}

        for pattern in range(16):
            with self.subTest(pattern=f"{pattern:04b}"):
                owner = np.full((3, 3), -1, dtype=np.int32)
                for bit, cell in enumerate(quadrants):
                    if pattern & (1 << bit):
                        owner[cell] = 0

                if pattern in diagonal_patterns:
                    with self.assertRaisesRegex(ValueError, "non-manifold"):
                        self.compile(owner, impedance=impedance, dx=dx, dy=dy)
                    continue

                boundary = self.compile(owner, impedance=impedance, dx=dx, dy=dy)
                rows = [
                    row
                    for row in boundary.rows
                    if row.electric_component == 2 and row.electric_index == (1, 1)
                ]
                metal_count = pattern.bit_count()
                if metal_count in (0, 4):
                    self.assertEqual(rows, [])
                    self.assertEqual(
                        boundary.electric_retained[2][1, 1],
                        metal_count == 0,
                    )
                    continue

                self.assertEqual(len(rows), 1)
                row = rows[0]
                expected_area = (4 - metal_count) * dx * dy / 4
                self.assertAlmostEqual(row.retained_dual_area, expected_area)

                states = [bool(pattern & (1 << bit)) for bit in range(4)]
                transitions = (
                    (0, 1, dy / 2),
                    (3, 2, dy / 2),
                    (0, 3, dx / 2),
                    (1, 2, dx / 2),
                )
                exposed_length = sum(
                    length
                    for first, second, length in transitions
                    if states[first] != states[second]
                )
                represented_load = (
                    1j
                    * 2
                    * np.pi
                    * self.FREQUENCY_HZ
                    * self.EPSILON0
                    * expected_area
                    * (row.relative_permittivity - 1.0)
                )
                self.assertAlmostEqual(represented_load, exposed_length / impedance)

    def test_single_lower_left_opaque_quadrant_has_exact_ez_circulation(self):
        """Check raw and installed Hx/Hy signs at a convex SIBC corner."""
        dx, dy = 0.8e-3, 1.2e-3
        owner = np.full((4, 4), -1, dtype=np.int32)
        owner[1, 1] = 0

        boundary = self.compile(owner, dx=dx, dy=dy)
        row = next(
            row
            for row in boundary.rows
            if row.electric_component == 2 and row.electric_index == (2, 2)
        )
        raw_terms = {
            (term.component, term.index): term.length
            for term in row.magnetic_terms
        }
        self.assertEqual(
            raw_terms,
            {
                (0, (2, 1)): dx / 2,
                (0, (2, 2)): -dx,
                (1, (1, 2)): -dy / 2,
                (1, (2, 2)): dy,
            },
        )

        solver = ModeSolver2D(
            self.FREQUENCY_HZ,
            4 * dx,
            4 * dy,
            4,
            4,
            num_modes=1,
        )
        solver.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(1, 2),
            y_range=(1, 2),
        )
        solver._effective_materials_and_masks()
        solver._yeeder2d()

        electric_row = solver._flat_index(2, 2, solver.shape_ez[0])
        area_scale = row.retained_dual_area * solver.k_0
        hx_bottom = solver._flat_index(2, 1, solver.shape_hx[0])
        hx_top = solver._flat_index(2, 2, solver.shape_hx[0])
        hy_left = solver._flat_index(1, 2, solver.shape_hy[0])
        hy_right = solver._flat_index(2, 2, solver.shape_hy[0])

        hx_row = solver.DHY_HX_TO_EZ.getrow(electric_row)
        hx_entries = dict(zip(hx_row.indices.tolist(), hx_row.data.tolist()))
        self.assertEqual(set(hx_entries), {hx_bottom, hx_top})
        self.assertAlmostEqual(hx_entries[hx_bottom], -dx / 2 / area_scale)
        self.assertAlmostEqual(hx_entries[hx_top], dx / area_scale)

        hy_row = solver.DHX_HY_TO_EZ.getrow(electric_row)
        hy_entries = dict(zip(hy_row.indices.tolist(), hy_row.data.tolist()))
        self.assertEqual(set(hy_entries), {hy_left, hy_right})
        self.assertAlmostEqual(hy_entries[hy_left], -dy / 2 / area_scale)
        self.assertAlmostEqual(hy_entries[hy_right], dy / area_scale)

    def test_ez_corner_sums_ports_from_different_impedance_definitions(self):
        dx, dy = 0.8e-3, 1.2e-3
        first_impedance = 30.0 + 4.0j
        second_impedance = 70.0 + 11.0j
        definitions = (
            self.definition(first_impedance),
            SurfaceImpedanceDefinition(
                key=("constant", second_impedance.real, second_impedance.imag),
                impedance=second_impedance,
                label="second",
            ),
        )
        owner = np.full((3, 3), -1, dtype=np.int32)
        owner[0, 0] = 0
        owner[0, 1] = 1
        ones = np.ones(owner.shape, dtype=complex)

        boundary = compile_impedance_boundary(
            owner=owner,
            definitions=definitions,
            cell_eps_r_xx=ones,
            cell_eps_r_yy=ones,
            cell_eps_r_zz=ones,
            dx=dx,
            dy=dy,
            frequency=self.FREQUENCY_HZ,
            epsilon0=self.EPSILON0,
        )
        row = next(
            row
            for row in boundary.rows
            if row.electric_component == 2 and row.electric_index == (1, 1)
        )
        represented_load = (
            1j
            * 2
            * np.pi
            * self.FREQUENCY_HZ
            * self.EPSILON0
            * row.retained_dual_area
            * (row.relative_permittivity - 1.0)
        )
        expected_load = dy / (2 * first_impedance) + dy / (2 * second_impedance)
        self.assertAlmostEqual(represented_load, expected_load)

    def test_interface_normal_permeability_comes_only_from_retained_cell(self):
        horizontal = ModeSolver2D(
            self.FREQUENCY_HZ,
            4.0e-3,
            4.0e-3,
            4,
            4,
            num_modes=1,
        )
        horizontal.cell_mu_r_yy[:, 0] = 99.0
        horizontal.cell_mu_r_yy[:, 1] = 4.0
        horizontal.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(0, 4),
            y_range=(0, 1),
        )
        materials, *_ = horizontal._effective_materials_and_masks()
        self.assertEqual(materials["mu_yy"][2, 1], 4.0)

        vertical = ModeSolver2D(
            self.FREQUENCY_HZ,
            4.0e-3,
            4.0e-3,
            4,
            4,
            num_modes=1,
        )
        vertical.cell_mu_r_xx[0, :] = 88.0
        vertical.cell_mu_r_xx[1, :] = 3.0
        vertical.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(0, 1),
            y_range=(0, 4),
        )
        materials, *_ = vertical._effective_materials_and_masks()
        self.assertEqual(materials["mu_xx"][1, 2], 3.0)

    def test_compiled_transverse_row_replaces_the_sparse_ampere_curl(self):
        solver = ModeSolver2D(
            self.FREQUENCY_HZ,
            4.0e-3,
            4.0e-3,
            4,
            4,
            num_modes=1,
        )
        solver.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(0, 4),
            y_range=(0, 1),
        )
        solver._effective_materials_and_masks()
        solver._yeeder2d()

        electric_row = solver._flat_index(2, 1, solver.shape_ex[0])
        magnetic_column = solver._flat_index(2, 1, solver.shape_hz[0])
        row = solver.DHY_HZ_TO_HY.getrow(electric_row)
        entries = dict(zip(row.indices.tolist(), row.data.tolist()))
        self.assertEqual(set(entries), {magnetic_column})
        self.assertAlmostEqual(
            entries[magnetic_column],
            2.0 / (solver.k_0 * solver.dy),
        )

    def test_copper_te10_wall_loss_matches_first_order_theory(self):
        width = 22.86e-3
        height = 10.16e-3
        aperture_nx, aperture_ny = 50, 25
        dx, dy = width / aperture_nx, height / aperture_ny
        nx, ny = aperture_nx + 2, aperture_ny + 2
        c = 1 / np.sqrt(self.EPSILON0 * (4e-7 * np.pi))
        cutoff = c / (2 * width)
        expected_neff = np.sqrt(1 - (cutoff / self.FREQUENCY_HZ) ** 2)

        solver = ModeSolver2D(
            self.FREQUENCY_HZ,
            nx * dx,
            ny * dy,
            nx,
            ny,
            num_modes=1,
            guess=-(expected_neff ** 2),
        )
        for x_range, y_range in (
            ((0, 1), (0, ny)),
            ((nx - 1, nx), (0, ny)),
            ((1, nx - 1), (0, 1)),
            ((1, nx - 1), (ny - 1, ny)),
        ):
            solver.add_impedance_surface(
                preset="Cu",
                x_range=x_range,
                y_range=y_range,
            )

        solver.solve()

        k0 = 2 * np.pi * self.FREQUENCY_HZ / c
        beta = k0 * expected_neff
        cutoff_wavenumber = np.pi / width
        impedance = good_conductor_surface_impedance("copper", self.FREQUENCY_HZ)
        eta0 = np.sqrt((4e-7 * np.pi) / self.EPSILON0)
        expected_alpha = (
            impedance.real
            / eta0
            * (
                k0 / (beta * height)
                + 2 * cutoff_wavenumber ** 2 / (k0 * beta * width)
            )
        )
        calculated_alpha = -k0 * solver.neff[0].imag

        pec_solver = ModeSolver2D(
            self.FREQUENCY_HZ,
            nx * dx,
            ny * dy,
            nx,
            ny,
            num_modes=1,
            guess=-(expected_neff ** 2),
        )
        for x_range, y_range in (
            ((0, 1), (0, ny)),
            ((nx - 1, nx), (0, ny)),
            ((1, nx - 1), (0, 1)),
            ((1, nx - 1), (ny - 1, ny)),
        ):
            pec_solver.add_pec(x_range, y_range)
        pec_solver.solve()

        perturbation_coefficient = expected_alpha / impedance.real
        expected_phase_shift_neff = (
            perturbation_coefficient * impedance.imag / k0
        )
        calculated_phase_shift_neff = (
            solver.neff[0].real - pec_solver.neff[0].real
        )

        self.assertAlmostEqual(solver.neff[0].real, expected_neff, delta=5e-4)
        self.assertAlmostEqual(
            k0 * solver.attenuation_constant[0],
            calculated_alpha,
        )
        self.assertAlmostEqual(pec_solver.attenuation_constant[0], 0.0)
        self.assertGreater(calculated_alpha, 0.0)
        self.assertAlmostEqual(
            calculated_alpha,
            expected_alpha,
            delta=0.01 * expected_alpha,
        )
        self.assertAlmostEqual(
            calculated_phase_shift_neff,
            expected_phase_shift_neff,
            delta=0.01 * expected_phase_shift_neff,
        )

    def test_opaque_wall_thickness_does_not_change_the_boundary_problem(self):
        width = 22.86e-3
        height = 10.16e-3
        aperture_nx, aperture_ny = 24, 12
        dx, dy = width / aperture_nx, height / aperture_ny
        c = 1 / np.sqrt(self.EPSILON0 * (4e-7 * np.pi))
        expected_neff = np.sqrt(
            1 - (c / (2 * width * self.FREQUENCY_HZ)) ** 2
        )

        results = []
        for wall_cells in (1, 2):
            nx = aperture_nx + 2 * wall_cells
            ny = aperture_ny + 2 * wall_cells
            solver = ModeSolver2D(
                self.FREQUENCY_HZ,
                nx * dx,
                ny * dy,
                nx,
                ny,
                num_modes=1,
                guess=-(expected_neff ** 2),
            )
            for x_range, y_range in (
                ((0, wall_cells), (0, ny)),
                ((nx - wall_cells, nx), (0, ny)),
                ((wall_cells, nx - wall_cells), (0, wall_cells)),
                ((wall_cells, nx - wall_cells), (ny - wall_cells, ny)),
            ):
                solver.add_impedance_surface(
                    preset="Cu",
                    x_range=x_range,
                    y_range=y_range,
                )

            solver.solve()
            boundary = solver._compiled_impedance_boundary
            results.append(
                (
                    solver.neff.copy(),
                    int(np.count_nonzero(solver.impedance_surface_mask)),
                    len(boundary.rows),
                    tuple(np.count_nonzero(mask) for mask in boundary.electric_retained),
                    tuple(np.count_nonzero(mask) for mask in boundary.magnetic_retained),
                )
            )

        thin, thick = results
        self.assertGreater(thick[1], thin[1])
        self.assertEqual(thick[2:], thin[2:])
        np.testing.assert_allclose(thick[0], thin[0], rtol=1e-9, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
