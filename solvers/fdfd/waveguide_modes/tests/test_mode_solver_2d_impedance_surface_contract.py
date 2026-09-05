"""Breaking API contract for true ModeSolver2D impedance boundaries.

These tests intentionally describe the opaque, one-sided boundary API.  The
legacy implementation smeared a sheet admittance through one or more material
cells; that behaviour belongs under a differently named thin-layer API.
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from fdfd_waveguide_modes import ModeSolver2D


class ModeSolver2DImpedanceSurfaceContractTests(unittest.TestCase):
    FREQUENCY_HZ = 10e9

    @classmethod
    def make_solver(cls, *, nx=6, ny=5, x_range=0.06, y_range=0.05, guess=None):
        return ModeSolver2D(
            cls.FREQUENCY_HZ,
            x_range,
            y_range,
            nx,
            ny,
            num_modes=1,
            guess=guess,
        )

    @staticmethod
    def surface_state(solver):
        """Return the mutation-sensitive portion of the boundary registry."""
        return (
            solver._surface_impedance_owner.copy(),
            tuple(id(definition) for definition in solver._surface_impedance_definitions),
            tuple(solver._surface_impedance_regions),
        )

    def assert_surface_state_equal(self, solver, state):
        owner, definition_ids, regions = state
        np.testing.assert_array_equal(solver._surface_impedance_owner, owner)
        self.assertEqual(
            tuple(id(definition) for definition in solver._surface_impedance_definitions),
            definition_ids,
        )
        self.assertEqual(tuple(solver._surface_impedance_regions), regions)

    @staticmethod
    def constraint_state(solver):
        return (
            tuple(
                values.copy()
                for values in (
                    solver.pec_xx_mask,
                    solver.pec_yy_mask,
                    solver.pec_zz_mask,
                    solver.pmc_xx_mask,
                    solver.pmc_yy_mask,
                    solver.pmc_zz_mask,
                    solver._pec_cell_mask,
                    solver._pmc_cell_mask,
                )
            ),
            len(solver._pec_regions),
            len(solver._pmc_regions),
        )

    def assert_constraint_state_equal(self, solver, state):
        masks, pec_regions, pmc_regions = state
        for actual, expected in zip(
            (
                solver.pec_xx_mask,
                solver.pec_yy_mask,
                solver.pec_zz_mask,
                solver.pmc_xx_mask,
                solver.pmc_yy_mask,
                solver.pmc_zz_mask,
                solver._pec_cell_mask,
                solver._pmc_cell_mask,
            ),
            masks,
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(solver._pec_regions), pec_regions)
        self.assertEqual(len(solver._pmc_regions), pmc_regions)

    def test_invalid_pec_pmc_components_are_rejected_atomically(self):
        for boundary_kind in ("pec", "pmc"):
            with self.subTest(boundary_kind=boundary_kind):
                solver = self.make_solver()
                initial = self.constraint_state(solver)

                with self.assertRaisesRegex(ValueError, "invalid tensor component"):
                    getattr(solver, f"add_{boundary_kind}")(
                        (1, 3),
                        (1, 3),
                        components=("invalid",),
                    )

                self.assert_constraint_state_equal(solver, initial)

    def test_breaking_signature_is_explicit_and_removes_thin_layer_options(self):
        signature = inspect.signature(ModeSolver2D.add_impedance_surface)
        parameters = signature.parameters

        self.assertEqual(
            tuple(parameters),
            ("self", "Zs", "preset", "x_range", "y_range"),
        )
        self.assertEqual(
            parameters["Zs"].kind,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        self.assertIsNone(parameters["Zs"].default)
        self.assertEqual(parameters["preset"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameters["preset"].default)
        for name in ("x_range", "y_range"):
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIs(parameters[name].default, inspect.Parameter.empty)

        for removed in ("position", "orientation", "thickness_cells", "eps_components"):
            self.assertNotIn(removed, parameters)

    def test_legacy_orientation_thickness_and_component_keywords_are_rejected(self):
        for removed in ("orientation", "thickness_cells", "eps_components"):
            solver = self.make_solver()
            with self.subTest(keyword=removed), self.assertRaises(TypeError):
                solver.add_impedance_surface(
                    40.0 + 5.0j,
                    x_range=(1, 3),
                    y_range=(1, 4),
                    **{removed: "x" if removed == "orientation" else 1},
                )

    def test_exactly_one_of_direct_impedance_and_preset_is_required_atomically(self):
        solver = self.make_solver()
        initial = self.surface_state(solver)
        invalid_calls = (
            {},
            {"Zs": 40.0 + 5.0j, "preset": "copper"},
        )

        for arguments in invalid_calls:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError) as raised:
                    solver.add_impedance_surface(
                        x_range=(1, 3),
                        y_range=(1, 4),
                        **arguments,
                    )
                message = str(raised.exception).casefold()
                self.assertIn("exactly one", message)
                self.assertIn("zs", message)
                self.assertIn("preset", message)
                self.assert_surface_state_equal(solver, initial)

    def test_direct_impedance_may_be_positional_but_other_arguments_are_keyword_only(self):
        solver = self.make_solver()
        solver.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(1, 3),
            y_range=(1, 4),
        )
        self.assertTrue(np.any(solver.impedance_surface_mask))

        with self.assertRaises(TypeError):
            self.make_solver().add_impedance_surface(
                None,
                "copper",
                x_range=(1, 3),
                y_range=(1, 4),
            )

    def test_rectangle_marks_opaque_cells_without_smearing_bulk_permittivity(self):
        solver = self.make_solver()
        bulk_permittivity = tuple(
            values.copy()
            for values in (
                solver.cell_eps_r_xx,
                solver.cell_eps_r_yy,
                solver.cell_eps_r_zz,
            )
        )

        solver.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(1, 4),
            y_range=(2, 5),
        )

        expected = np.zeros(solver.shape_cell, dtype=bool)
        expected[1:4, 2:5] = True
        np.testing.assert_array_equal(solver.impedance_surface_mask, expected)
        self.assertEqual(solver._surface_impedance_owner.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(
            solver._surface_impedance_owner[~expected],
            -1,
        )
        self.assertEqual(
            np.unique(solver._surface_impedance_owner[expected]).size,
            1,
        )

        for actual, original in zip(
            (
                solver.cell_eps_r_xx,
                solver.cell_eps_r_yy,
                solver.cell_eps_r_zz,
            ),
            bulk_permittivity,
        ):
            np.testing.assert_array_equal(actual, original)

        # The public diagnostic must not expose mutable ownership state.
        returned_mask = solver.impedance_surface_mask
        returned_mask[:] = False
        np.testing.assert_array_equal(solver.impedance_surface_mask, expected)

    def test_same_direct_definition_is_idempotent_and_unions_overlapping_rectangles(self):
        solver = self.make_solver(nx=6, ny=6, y_range=0.06)
        impedance = 40.0 + 5.0j
        solver.add_impedance_surface(
            impedance,
            x_range=(0, 3),
            y_range=(0, 3),
        )
        first_owner = int(solver._surface_impedance_owner[1, 1])

        solver.add_impedance_surface(
            impedance,
            x_range=(2, 5),
            y_range=(2, 5),
        )

        expected = np.zeros(solver.shape_cell, dtype=bool)
        expected[0:3, 0:3] = True
        expected[2:5, 2:5] = True
        np.testing.assert_array_equal(solver.impedance_surface_mask, expected)
        np.testing.assert_array_equal(
            solver._surface_impedance_owner[expected],
            first_owner,
        )
        self.assertEqual(len(solver._surface_impedance_definitions), 1)

    def test_preset_aliases_share_one_normalized_definition(self):
        alias_groups = (
            ("Cu", "copper", " cU "),
            ("Al", "aluminum", "aluminium"),
        )
        for aliases in alias_groups:
            with self.subTest(aliases=aliases):
                solver = self.make_solver(nx=6, ny=6, y_range=0.06)
                rectangles = (
                    ((0, 3), (0, 3)),
                    ((2, 5), (2, 5)),
                    ((1, 4), (3, 6)),
                )
                for alias, (x_range, y_range) in zip(aliases, rectangles):
                    solver.add_impedance_surface(
                        preset=alias,
                        x_range=x_range,
                        y_range=y_range,
                    )

                owners = solver._surface_impedance_owner[
                    solver.impedance_surface_mask
                ]
                self.assertEqual(np.unique(owners).size, 1)
                self.assertEqual(len(solver._surface_impedance_definitions), 1)

    def test_unknown_preset_is_rejected_without_mutation(self):
        solver = self.make_solver()
        initial = self.surface_state(solver)
        with self.assertRaisesRegex(ValueError, r"(?i)unknown.*metal"):
            solver.add_impedance_surface(
                preset="unobtainium",
                x_range=(1, 3),
                y_range=(1, 4),
            )
        self.assert_surface_state_equal(solver, initial)

    def test_invalid_direct_impedances_are_rejected_without_mutation(self):
        invalid_values = (
            0.0,
            complex(np.inf, 0.0),
            complex(np.nan, 0.0),
            -1.0 + 2.0j,
            True,
            np.asarray([40.0 + 5.0j]),
        )
        for impedance in invalid_values:
            with self.subTest(impedance=impedance):
                solver = self.make_solver()
                initial = self.surface_state(solver)
                with self.assertRaises((TypeError, ValueError)):
                    solver.add_impedance_surface(
                        impedance,
                        x_range=(1, 3),
                        y_range=(1, 4),
                    )
                self.assert_surface_state_equal(solver, initial)

    def test_different_definitions_may_be_disjoint(self):
        solver = self.make_solver()
        solver.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(0, 2),
            y_range=(0, 2),
        )
        solver.add_impedance_surface(
            60.0 + 8.0j,
            x_range=(3, 5),
            y_range=(3, 5),
        )

        owners = solver._surface_impedance_owner[solver.impedance_surface_mask]
        self.assertEqual(set(np.unique(owners)), {0, 1})
        self.assertEqual(len(solver._surface_impedance_definitions), 2)

    def test_different_impedance_overlap_is_rejected_atomically(self):
        solver = self.make_solver(nx=6, ny=6, y_range=0.06)
        solver.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(0, 3),
            y_range=(0, 3),
        )
        initial = self.surface_state(solver)

        with self.assertRaisesRegex(ValueError, r"(?i)impedance.*overlap"):
            solver.add_impedance_surface(
                41.0 + 5.0j,
                x_range=(2, 5),
                y_range=(2, 5),
            )

        self.assert_surface_state_equal(solver, initial)

    def test_impedance_and_pec_pmc_overlap_is_rejected_in_either_order_atomically(self):
        for boundary_kind in ("pec", "pmc"):
            with self.subTest(boundary_kind=boundary_kind, order="constraint-first"):
                solver = self.make_solver()
                add_constraint = getattr(solver, f"add_{boundary_kind}")
                add_constraint((1, 3), (1, 3))
                surface_before = self.surface_state(solver)
                constraints_before = self.constraint_state(solver)
                with self.assertRaisesRegex(ValueError, rf"(?i){boundary_kind}.*overlap|overlap.*{boundary_kind}"):
                    solver.add_impedance_surface(
                        40.0 + 5.0j,
                        x_range=(2, 4),
                        y_range=(2, 4),
                    )
                self.assert_surface_state_equal(solver, surface_before)
                self.assert_constraint_state_equal(solver, constraints_before)

            with self.subTest(boundary_kind=boundary_kind, order="impedance-first"):
                solver = self.make_solver()
                solver.add_impedance_surface(
                    40.0 + 5.0j,
                    x_range=(1, 4),
                    y_range=(1, 4),
                )
                surface_before = self.surface_state(solver)
                constraints_before = self.constraint_state(solver)
                add_constraint = getattr(solver, f"add_{boundary_kind}")
                with self.assertRaisesRegex(ValueError, rf"(?i){boundary_kind}.*overlap|overlap.*{boundary_kind}"):
                    add_constraint((3, 5), (3, 5))
                self.assert_surface_state_equal(solver, surface_before)
                self.assert_constraint_state_equal(solver, constraints_before)

    def test_adjacent_pec_pmc_row_conflicts_roll_back_atomically(self):
        for boundary_kind in ("pec", "pmc"):
            with self.subTest(boundary_kind=boundary_kind):
                solver = self.make_solver()
                solver.add_impedance_surface(
                    40.0 + 5.0j,
                    x_range=(0, 1),
                    y_range=(0, solver.Ny),
                )
                solver._effective_materials_and_masks()
                compiled_before = solver._compiled_impedance_boundary
                constraints_before = self.constraint_state(solver)

                with self.assertRaisesRegex(ValueError, r"(?i)surface-impedance.*constraint"):
                    getattr(solver, f"add_{boundary_kind}")(
                        (1, 2),
                        (0, solver.Ny),
                    )

                self.assert_constraint_state_equal(solver, constraints_before)
                self.assertIs(solver._compiled_impedance_boundary, compiled_before)

    @staticmethod
    def pml_state(solver):
        return (
            solver._pml_cell_mask.copy(),
            tuple(
                values.copy()
                for values in (
                    solver.cell_eps_r_xx,
                    solver.cell_eps_r_yy,
                    solver.cell_eps_r_zz,
                    solver.cell_mu_r_xx,
                    solver.cell_mu_r_yy,
                    solver.cell_mu_r_zz,
                    solver.eps_r_xx,
                    solver.eps_r_yy,
                    solver.eps_r_zz,
                    solver.mu_r_xx,
                    solver.mu_r_yy,
                    solver.mu_r_zz,
                )
            ),
        )

    def assert_pml_state_equal(self, solver, state):
        pml_mask, material_arrays = state
        np.testing.assert_array_equal(solver._pml_cell_mask, pml_mask)
        for actual, expected in zip(
            (
                solver.cell_eps_r_xx,
                solver.cell_eps_r_yy,
                solver.cell_eps_r_zz,
                solver.cell_mu_r_xx,
                solver.cell_mu_r_yy,
                solver.cell_mu_r_zz,
                solver.eps_r_xx,
                solver.eps_r_yy,
                solver.eps_r_zz,
                solver.mu_r_xx,
                solver.mu_r_yy,
                solver.mu_r_zz,
            ),
            material_arrays,
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_adjacent_pml_and_impedance_are_rejected_atomically_in_either_order(self):
        solver = self.make_solver()
        solver.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(1, 2),
            y_range=(0, solver.Ny),
        )
        pml_before = self.pml_state(solver)

        with self.assertRaisesRegex(ValueError, r"(?i)impedance.*pml"):
            solver.add_pml(pml_width=1, direction="x-")

        self.assert_pml_state_equal(solver, pml_before)

        solver = self.make_solver()
        solver.add_pml(pml_width=1, direction="x-")
        surface_before = self.surface_state(solver)

        with self.assertRaisesRegex(ValueError, r"(?i)impedance.*pml"):
            solver.add_impedance_surface(
                40.0 + 5.0j,
                x_range=(1, 2),
                y_range=(0, solver.Ny),
            )

        self.assert_surface_state_equal(solver, surface_before)

    def test_purely_reactive_closed_wall_has_no_modal_attenuation(self):
        # A 4-by-4-cell air aperture is enclosed by an opaque, purely reactive
        # wall.  With lossless bulk materials and Re(Zs)=0 the assembled
        # operator is lossless, so every propagating selected mode must have a
        # real effective index to numerical precision.
        solver = self.make_solver(
            nx=8,
            ny=8,
            x_range=0.08,
            y_range=0.08,
            guess=-0.85,
        )
        reactance = 25.0j
        for x_range, y_range in (
            ((0, 2), (0, 8)),
            ((6, 8), (0, 8)),
            ((2, 6), (0, 2)),
            ((2, 6), (6, 8)),
        ):
            solver.add_impedance_surface(
                reactance,
                x_range=x_range,
                y_range=y_range,
            )

        solver.solve()

        self.assertTrue(np.isfinite(solver.neff[0]))
        self.assertGreater(solver.neff[0].real, 0.0)
        self.assertLess(abs(solver.neff[0].imag), 1e-9)
        np.testing.assert_allclose(
            solver.Hz[solver.impedance_surface_mask, :],
            0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
