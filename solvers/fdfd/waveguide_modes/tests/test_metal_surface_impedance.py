import math
import subprocess
import sys
import unittest
from pathlib import Path

from fdfd_waveguide_modes.metal_surface_impedance import (
    METAL_RESISTIVITIES_OHM_M,
    MU_0_H_PER_M,
    canonical_metal_name,
    good_conductor_surface_impedance,
    metal_conductivity,
    metal_resistivity,
)


EXPECTED_RESISTIVITIES_OHM_M = {
    "aluminium": 2.650e-8,
    "copper": 1.676e-8,
    "gold": 2.192e-8,
    "molybdenum": 5.340e-8,
    "palladium": 1.054e-7,
    "silver": 1.586e-8,
    "tungsten": 5.280e-8,
    "zinc": 5.964e-8,
}


class MetalSurfaceImpedanceTests(unittest.TestCase):
    def test_family_implementations_and_impedance_compilers_do_not_collide(self):
        from fdfd_waveguide_modes import ModeSolver1D, ModeSolver2D
        one = ModeSolver1D(10e9, .08, 8, 1)
        two = ModeSolver2D(10e9, .04, .04, 4, 4, 1)
        one.add_impedance_surface(1+1j, x_range=(0, 1))
        two.add_impedance_surface(1+1j, x_range=(0, 1), y_range=(0, 4))
        self.assertNotEqual(type(one), type(two))



    def test_resistivity_table_matches_reference_values_exactly(self):
        self.assertEqual(dict(METAL_RESISTIVITIES_OHM_M), EXPECTED_RESISTIVITIES_OHM_M)
        for metal, expected in EXPECTED_RESISTIVITIES_OHM_M.items():
            self.assertEqual(metal_resistivity(metal), expected)

    def test_symbols_and_full_names_are_case_insensitive(self):
        aliases = {
            "Al": "aluminium",
            "CU": "copper",
            "au": "gold",
            "mO": "molybdenum",
            "PD": "palladium",
            "Ag": "silver",
            "w": "tungsten",
            "Zn": "zinc",
        }
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(canonical_metal_name(alias), canonical)
                self.assertEqual(metal_resistivity(alias), metal_resistivity(canonical.upper()))

    def test_aluminum_spelling_alias(self):
        self.assertEqual(canonical_metal_name("aluminum"), "aluminium")
        self.assertEqual(metal_resistivity("ALUMINUM"), 2.650e-8)

    def test_conductivity_is_exact_reciprocal_of_resistivity(self):
        for metal, resistivity in EXPECTED_RESISTIVITIES_OHM_M.items():
            with self.subTest(metal=metal):
                self.assertEqual(metal_conductivity(metal), 1.0 / resistivity)
                self.assertAlmostEqual(metal_conductivity(metal) * resistivity, 1.0)

    def test_good_conductor_impedance_uses_positive_j_for_exp_plus_jwt(self):
        frequency_hz = 10e9
        resistivity = EXPECTED_RESISTIVITIES_OHM_M["copper"]
        expected_rs = math.sqrt(math.pi * frequency_hz * MU_0_H_PER_M * resistivity)

        impedance = good_conductor_surface_impedance("Cu", frequency_hz)

        self.assertAlmostEqual(impedance.real, expected_rs)
        self.assertAlmostEqual(impedance.imag, expected_rs)
        self.assertGreater(impedance.real, 0.0)
        self.assertGreater(impedance.imag, 0.0)

    def test_relative_permeability_scales_impedance_by_square_root(self):
        base = good_conductor_surface_impedance("W", 1e9)
        magnetic = good_conductor_surface_impedance(
            "W", 1e9, relative_permeability=4.0
        )
        self.assertAlmostEqual(magnetic.real, 2.0 * base.real)
        self.assertAlmostEqual(magnetic.imag, 2.0 * base.imag)

    def test_invalid_lookup_and_physical_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            metal_resistivity("iron")
        with self.assertRaises(TypeError):
            metal_resistivity(29)
        with self.assertRaises(ValueError):
            good_conductor_surface_impedance("Cu", 0.0)
        with self.assertRaises(ValueError):
            good_conductor_surface_impedance("Cu", math.inf)
        with self.assertRaises(ValueError):
            good_conductor_surface_impedance("Cu", 1e9, relative_permeability=0.0)


if __name__ == "__main__":
    unittest.main()
