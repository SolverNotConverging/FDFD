"""Sweep ordinary frequency and save complete results to one HDF5 file.

Every point is independently meshed, solved for its lead modes, launched,
and scattered.  The resulting HDF5 file can be opened with
``fem-waveguide-scattering-viewer frequency_sweep.h5``.
"""

from __future__ import annotations

import numpy as np
import fem_waveguide_scattering as wf


def main() -> None:
    frequencies_hz = np.linspace(191.0e12, 195.0e12, 3)
    simulation = wf.WaveguideScatteringSolver2D(frequency=float(frequencies_hz[0]), angle=0.0, x_range=(0.0, 1e-06), z_range=(-3e-06, 3e-06), background_epsilon=1.0, transverse_boundary='pec')
    simulation.add_rectangle(x_range=(0.0, 1e-06), z_range=(-3e-07, 3e-07), epsilon=1.002, name='weak_insert')
    simulation.add_pml(order=3, target_reflection=1e-08, thickness=8e-07, direction='z')

    sweep = simulation.sweep(frequencies_hz, mesh_options={'wavelength_elements': 8}, mode_options={'max_refinements': 0, 'num_modes': 1, 'neff_guess': 1.0}, max_refinements=0)
    output_path = sweep.save('frequency_sweep.h5')

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
