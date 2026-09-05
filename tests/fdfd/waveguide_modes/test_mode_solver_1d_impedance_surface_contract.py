"""Breaking API contract for true ModeSolver1D impedance boundaries.

The 1D API marks opaque cells.  A boundary is compiled only at a transition
between an opaque cell and a retained cell; the marked interval is therefore
not a thin material film.
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from fdfd_waveguide_modes.solver_1d import _ModeSolver1D as ModeSolver1D


class ModeSolver1DImpedanceSurfaceContractTests(unittest.TestCase):
    FREQUENCY_HZ = 10e9

    @classmethod
    def make_solver(cls, *, nx=8, x_range=0.08, guess=None):
        return ModeSolver1D(
            cls.FREQUENCY_HZ,
            x_range,
            nx,
            num_modes=1,
            guess=guess,
        )

    @staticmethod
    def surface_state(solver):
        """Return every mutation-sensitive part of the SIBC registry."""
        return (
            solver._surface_impedance_owner.copy(),
            tuple(id(definition) for definition in solver._surface_impedance_definitions),
            tuple(solver._surface_impedance_regions),
            solver._compiled_impedance_boundary,
        )

    def assert_surface_state_equal(self, solver, state):
        owner, definition_ids, regions, compiled = state
        np.testing.assert_array_equal(solver._surface_impedance_owner, owner)
        self.assertEqual(
            tuple(id(definition) for definition in solver._surface_impedance_definitions),
            definition_ids,
        )
        self.assertEqual(tuple(solver._surface_impedance_regions), regions)
        self.assertIs(solver._compiled_impedance_boundary, compiled)

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
            tuple(solver._pec_regions),
            tuple(solver._pmc_regions),
            solver._compiled_impedance_boundary,
        )

    def assert_constraint_state_equal(self, solver, state):
        masks, pec_regions, pmc_regions, compiled = state
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
        self.assertEqual(tuple(solver._pec_regions), pec_regions)
        self.assertEqual(tuple(solver._pmc_regions), pmc_regions)
        self.assertIs(solver._compiled_impedance_boundary, compiled)

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

    @staticmethod
    def compile_boundary(solver):
        solver._effective_materials_and_masks()
        boundary = solver._compiled_impedance_boundary
        if boundary is None:
            raise AssertionError("The impedance boundary was not compiled.")
        return boundary

    def test_breaking_signature_is_explicit(self):
        signature = inspect.signature(ModeSolver1D.add_impedance_surface)
        parameters = signature.parameters

        self.assertEqual(tuple(parameters), ("self", "Zs", "preset", "x_range"))
        self.assertEqual(
            parameters["Zs"].kind,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        self.assertIsNone(parameters["Zs"].default)
        self.assertEqual(parameters["preset"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameters["preset"].default)
        self.assertEqual(parameters["x_range"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["x_range"].default, inspect.Parameter.empty)

        for removed in ("position", "orientation", "thickness_cells", "eps_components"):
            self.assertNotIn(removed, parameters)

    def test_legacy_thin_layer_keywords_are_rejected(self):
        for removed in ("position", "orientation", "thickness_cells", "eps_components"):
            with self.subTest(keyword=removed), self.assertRaises(TypeError):
                self.make_solver().add_impedance_surface(
                    40.0 + 5.0j,
                    x_range=(0, 1),
                    **{removed: 1},
                )

    def test_exactly_one_of_direct_impedance_and_preset_is_required_atomically(self):
        solver = self.make_solver()
        initial = self.surface_state(solver)
        for arguments in ({}, {"Zs": 40.0 + 5.0j, "preset": "copper"}):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, r"(?i)exactly one.*Zs.*preset"):
                    solver.add_impedance_surface(x_range=(0, 1), **arguments)
                self.assert_surface_state_equal(solver, initial)

    def test_direct_impedance_may_be_positional_but_range_is_keyword_only(self):
        solver = self.make_solver()
        solver.add_impedance_surface(40.0 + 5.0j, x_range=(0, 1))
        self.assertTrue(np.any(solver.impedance_surface_mask))

        with self.assertRaises(TypeError):
            self.make_solver().add_impedance_surface(40.0 + 5.0j, (0, 1))

    def test_invalid_direct_values_and_unknown_preset_are_atomic(self):
        invalid_values = (
            0.0,
            complex(np.inf, 0.0),
            complex(np.nan, 0.0),
            -1.0 + 2.0j,
            True,
            "40+5j",
            np.asarray([40.0 + 5.0j]),
        )
        for value in invalid_values:
            with self.subTest(value=value):
                solver = self.make_solver()
                initial = self.surface_state(solver)
                with self.assertRaises((TypeError, ValueError)):
                    solver.add_impedance_surface(value, x_range=(0, 1))
                self.assert_surface_state_equal(solver, initial)

        solver = self.make_solver()
        initial = self.surface_state(solver)
        with self.assertRaisesRegex(ValueError, r"(?i)unknown.*metal"):
            solver.add_impedance_surface(preset="unobtainium", x_range=(0, 1))
        self.assert_surface_state_equal(solver, initial)

    def test_every_metal_full_name_and_symbol_alias_is_accepted_and_normalized(self):
        alias_groups = (
            ("aluminium", "Al", "ALUMINUM", " aluminium "),
            ("copper", "Cu", "CU", " copper "),
            ("gold", "Au", "AU", " gold "),
            ("molybdenum", "Mo", "MO", " molybdenum "),
            ("palladium", "Pd", "PD", " palladium "),
            ("silver", "Ag", "AG", " silver "),
            ("tungsten", "W", "w", " tungsten "),
            ("zinc", "Zn", "ZN", " zinc "),
        )
        for aliases in alias_groups:
            with self.subTest(aliases=aliases):
                solver = self.make_solver()
                for alias in aliases:
                    solver.add_impedance_surface(
                        preset=alias,
                        x_range=(0, 1),
                    )
                self.assertEqual(len(solver._surface_impedance_definitions), 1)
                self.assertEqual(
                    np.unique(solver._surface_impedance_owner[solver.impedance_surface_mask]).size,
                    1,
                )

    def test_region_marks_opaque_cells_without_smearing_bulk_permittivity(self):
        solver = self.make_solver()
        original_permittivity = tuple(
            values.copy()
            for values in (
                solver.cell_eps_r_xx,
                solver.cell_eps_r_yy,
                solver.cell_eps_r_zz,
            )
        )

        solver.add_impedance_surface(
            40.0 + 5.0j,
            x_range=(2, 5),
        )

        expected = np.zeros(solver.shape_cell, dtype=bool)
        expected[2:5] = True
        np.testing.assert_array_equal(solver.impedance_surface_mask, expected)
        self.assertEqual(solver._surface_impedance_owner.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(solver._surface_impedance_owner[~expected], -1)
        for actual, original in zip(
            (
                solver.cell_eps_r_xx,
                solver.cell_eps_r_yy,
                solver.cell_eps_r_zz,
            ),
            original_permittivity,
        ):
            np.testing.assert_array_equal(actual, original)

        returned_mask = solver.impedance_surface_mask
        returned_mask[:] = False
        np.testing.assert_array_equal(solver.impedance_surface_mask, expected)

    def test_same_definition_is_idempotent_and_unions_overlapping_ranges(self):
        solver = self.make_solver()
        impedance = 40.0 + 5.0j
        solver.add_impedance_surface(impedance, x_range=(0, 3))
        first_owner = int(solver._surface_impedance_owner[1])
        solver.add_impedance_surface(impedance, x_range=(2, 5))

        expected = np.zeros(solver.shape_cell, dtype=bool)
        expected[:5] = True
        np.testing.assert_array_equal(solver.impedance_surface_mask, expected)
        np.testing.assert_array_equal(solver._surface_impedance_owner[expected], first_owner)
        self.assertEqual(len(solver._surface_impedance_definitions), 1)

    def test_different_definition_overlap_is_rejected_atomically(self):
        solver = self.make_solver()
        solver.add_impedance_surface(40.0 + 5.0j, x_range=(0, 3))
        initial = self.surface_state(solver)

        with self.assertRaisesRegex(ValueError, r"(?i)impedance.*overlap"):
            solver.add_impedance_surface(41.0 + 5.0j, x_range=(2, 5))
        self.assert_surface_state_equal(solver, initial)

    def test_different_definitions_may_be_disjoint_or_adjacent(self):
        disjoint = self.make_solver()
        disjoint.add_impedance_surface(40.0 + 5.0j, x_range=(0, 2))
        disjoint.add_impedance_surface(60.0 + 8.0j, x_range=(4, 6))
        self.assertEqual(
            set(np.unique(disjoint._surface_impedance_owner[disjoint.impedance_surface_mask])),
            {0, 1},
        )

        adjacent = self.make_solver()
        adjacent.add_impedance_surface(40.0 + 5.0j, x_range=(0, 2))
        adjacent.add_impedance_surface(60.0 + 8.0j, x_range=(2, 4))
        np.testing.assert_array_equal(
            adjacent.impedance_surface_mask,
            np.asarray([True, True, True, True, False, False, False, False]),
        )
        self.assertEqual(len(adjacent._surface_impedance_definitions), 2)

    def test_all_opaque_geometry_is_rejected_atomically(self):
        solver = self.make_solver()
        initial = self.surface_state(solver)
        with self.assertRaisesRegex(ValueError, r"(?i)no retained|all.*opaque"):
            solver.add_impedance_surface(
                40.0 + 5.0j,
                x_range=(0, solver.Nx),
            )
        self.assert_surface_state_equal(solver, initial)

    def test_pec_pmc_overlap_and_shared_interface_conflicts_are_atomic(self):
        for boundary_kind in ("pec", "pmc"):
            add_name = f"add_{boundary_kind}"
            for constraint_range, geometry in (((0, 1), "overlap"), ((1, 2), "adjacent")):
                with self.subTest(
                    boundary_kind=boundary_kind,
                    geometry=geometry,
                    order="constraint-first",
                ):
                    solver = self.make_solver()
                    getattr(solver, add_name)(constraint_range)
                    surface_before = self.surface_state(solver)
                    constraints_before = self.constraint_state(solver)
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"(?i){boundary_kind}.*(?:overlap|constraint|interface)|(?:overlap|constraint|interface).*{boundary_kind}",
                    ):
                        solver.add_impedance_surface(40.0 + 5.0j, x_range=(0, 1))
                    self.assert_surface_state_equal(solver, surface_before)
                    self.assert_constraint_state_equal(solver, constraints_before)

                with self.subTest(
                    boundary_kind=boundary_kind,
                    geometry=geometry,
                    order="impedance-first",
                ):
                    solver = self.make_solver()
                    solver.add_impedance_surface(40.0 + 5.0j, x_range=(0, 1))
                    self.compile_boundary(solver)
                    surface_before = self.surface_state(solver)
                    constraints_before = self.constraint_state(solver)
                    with self.assertRaisesRegex(
                        ValueError,
                        r"(?i)surface-impedance.*constraint|constraint.*surface-impedance|impedance.*overlap",
                    ):
                        getattr(solver, add_name)(constraint_range)
                    self.assert_surface_state_equal(solver, surface_before)
                    self.assert_constraint_state_equal(solver, constraints_before)

    def test_transverse_pec_pmc_cross_constraint_closure(self):
        cases = (
            ("pec", "xx", 0, 4),
            ("pec", "yy", 1, 3),
            ("pmc", "xx", 3, 1),
            ("pmc", "yy", 4, 0),
        )
        for boundary_kind, component, source_index, implied_index in cases:
            with self.subTest(boundary_kind=boundary_kind, component=component):
                solver = self.make_solver()
                getattr(solver, f"add_{boundary_kind}")(
                    (2, 3),
                    components=(component,),
                )
                effective_masks = solver._effective_materials_and_masks()[1:]
                source = effective_masks[source_index]
                implied = effective_masks[implied_index]
                np.testing.assert_array_equal(implied, source)
                self.assertTrue(np.any(source))

    def test_xx_cross_constraints_conflict_at_impedance_interface_atomically(self):
        for boundary_kind in ("pec", "pmc"):
            with self.subTest(boundary_kind=boundary_kind, order="constraint-first"):
                solver = self.make_solver()
                getattr(solver, f"add_{boundary_kind}")(
                    (1, 2),
                    components=("xx",),
                )
                surface_before = self.surface_state(solver)
                constraints_before = self.constraint_state(solver)
                with self.assertRaisesRegex(
                    ValueError,
                    r"(?i)surface-impedance.*constraint|constraint.*surface-impedance",
                ):
                    solver.add_impedance_surface(40.0 + 5.0j, x_range=(0, 1))
                self.assert_surface_state_equal(solver, surface_before)
                self.assert_constraint_state_equal(solver, constraints_before)

            with self.subTest(boundary_kind=boundary_kind, order="impedance-first"):
                solver = self.make_solver()
                solver.add_impedance_surface(40.0 + 5.0j, x_range=(0, 1))
                self.compile_boundary(solver)
                surface_before = self.surface_state(solver)
                constraints_before = self.constraint_state(solver)
                with self.assertRaisesRegex(
                    ValueError,
                    r"(?i)surface-impedance.*constraint|constraint.*surface-impedance",
                ):
                    getattr(solver, f"add_{boundary_kind}")(
                        (1, 2),
                        components=("xx",),
                    )
                self.assert_surface_state_equal(solver, surface_before)
                self.assert_constraint_state_equal(solver, constraints_before)

    def test_pml_overlap_and_shared_interface_conflicts_are_atomic_in_either_order(self):
        for surface_range, geometry in (((0, 1), "overlap"), ((1, 2), "adjacent")):
            with self.subTest(geometry=geometry, order="impedance-first"):
                solver = self.make_solver()
                solver.add_impedance_surface(40.0 + 5.0j, x_range=surface_range)
                pml_before = self.pml_state(solver)
                with self.assertRaisesRegex(ValueError, r"(?i)impedance.*pml|pml.*impedance"):
                    solver.add_pml(pml_width=1, direction="x-")
                self.assert_pml_state_equal(solver, pml_before)

            with self.subTest(geometry=geometry, order="pml-first"):
                solver = self.make_solver()
                solver.add_pml(pml_width=1, direction="x-")
                surface_before = self.surface_state(solver)
                pml_before = self.pml_state(solver)
                with self.assertRaisesRegex(ValueError, r"(?i)impedance.*pml|pml.*impedance"):
                    solver.add_impedance_surface(40.0 + 5.0j, x_range=surface_range)
                self.assert_surface_state_equal(solver, surface_before)
                self.assert_pml_state_equal(solver, pml_before)

    def test_terminal_run_has_one_interface_and_internal_run_has_two(self):
        terminal = self.make_solver()
        terminal.add_impedance_surface(40.0 + 5.0j, x_range=(0, 2))
        terminal_boundary = self.compile_boundary(terminal)
        self.assertEqual(len(terminal_boundary.rows), 1)

        internal = self.make_solver()
        internal.add_impedance_surface(40.0 + 5.0j, x_range=(2, 4))
        internal_boundary = self.compile_boundary(internal)
        self.assertEqual(len(internal_boundary.rows), 2)


if __name__ == "__main__":
    unittest.main()
