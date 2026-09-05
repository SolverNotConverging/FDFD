"""Sweep ordinary frequency and save complete results to one HDF5 file.

Every point is independently meshed, solved for its lead modes, launched,
and scattered.  The resulting HDF5 file can be opened with
``fem-waveguide-scattering-viewer <output-directory>/results.h5``.
"""

from __future__ import annotations
from cem_common import Material, materials

import numpy as np
import fem_waveguide_scattering as scattering

from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs/examples/fem/waveguide_scattering" / Path(
    __file__,
).stem


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies_hz = np.linspace(191.0e12, 195.0e12, 3)
    insert = Material(name="weak dielectric insert", epsilon=1.002)
    simulation = scattering.WaveguideScatteringSolver2D(frequency=float(frequencies_hz[0]), angle=0.0, x_range=(0.0, 1e-06), z_range=(-3e-06, 3e-06), background_material=materials.vacuum, boundary=materials.PEC)
    simulation.add_rectangle(x_range=(0.0, 1e-06), z_range=(-3e-07, 3e-07), name='weak_insert', material=insert)
    simulation.add_pml(order=3, target_reflection=1e-08, thickness=8e-07, direction='z')

    sweep = simulation.sweep(
        frequencies_hz,
        mesh_options={'wavelength_elements': 8},
        mode_options={'max_refinements': 0, 'num_modes': 1, 'neff_guess': 1.0},
        max_refinements=0,
    )
    output_path = sweep.save(OUTPUT_DIR / 'results.h5')

    print("frequency (Hz) =", sweep.frequencies_hz)
    print("S11 =", sweep.S11)
    print("S21 =", sweep.S21)
    print("R =", sweep.reflection)
    print("T =", sweep.transmission)
    print("power-balance error =", sweep.power_balance_error)
    print("HDF5 sweep =", output_path)

    sweep.show()


if __name__ == "__main__":
    main()
