import math
import subprocess
import sys
import unittest
from pathlib import Path

from Mode_Solver_2D import (
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
    def test_2d_direct_folder_compatibility_import(self):
        solver_directory = Path(__file__).resolve().parents[1] / "Mode_Solver_2D"
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                "-c",
                (
                    "import metal_surface_impedance as presets; "
                    "assert presets.canonical_metal_name('Cu') == 'copper'"
                ),
            ],
            cwd=solver_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_folder_1d_and_2d_compilers_do_not_collide(self):
        repository_root = Path(__file__).resolve().parents[1]
        scripts = (
            (
                "sys.path.insert(0, str(root / 'Mode_Solver_1D')); "
                "from Mode_Solver_1D import ModeSolver1D; "
                "sys.path.insert(0, str(root / 'Mode_Solver_2D')); "
                "from Mode_Solver_2D import ModeSolver2D; "
                "solver = ModeSolver2D(10e9, 0.04, 0.04, 4, 4, 1); "
                "solver.add_impedance_surface(1+1j, x_range=(0, 1), "
                "y_range=(0, 4))"
            ),
            (
                "sys.path.insert(0, str(root / 'Mode_Solver_2D')); "
                "from Mode_Solver_2D import ModeSolver2D; "
                "sys.path.insert(0, str(root / 'Mode_Solver_1D')); "
                "from Mode_Solver_1D import ModeSolver1D; "
                "solver = ModeSolver1D(10e9, 0.08, 8, 1); "
                "solver.add_impedance_surface(1+1j, x_range=(0, 1))"
            ),
        )
        for script in scripts:
            with self.subTest(script=script):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-W",
                        "error",
                        "-c",
                        "from pathlib import Path; import sys; "
                        "root = Path.cwd(); "
                        + script,
                    ],
                    cwd=repository_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

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
