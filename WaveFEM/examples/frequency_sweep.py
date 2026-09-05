"""Sweep ordinary frequency and save complete results to one HDF5 file.

Every point is independently meshed, solved for its lead modes, launched,
and scattered.  The resulting HDF5 file can be opened with
``wavefem-viewer frequency_sweep.h5``.
"""

from __future__ import annotations

import numpy as np
import wavefem as wf


def main() -> None:
    frequencies_hz = np.linspace(191.0e12, 195.0e12, 3)
    simulation = wf.Scattering2D(
        solver_options=wf.SolverOptions(max_refinements=0),
        frequency=float(frequencies_hz[0]),
        angle=0.0,
        x_span=(0.0, 1.0e-6),
        z_span=(-3.0e-6, 3.0e-6),
        background_eps=1.0,
        transverse_boundary="pec",
    )
    simulation.add_rectangle(
        x=(0.0, 1.0e-6),
        z=(-0.30e-6, 0.30e-6),
        eps=1.002,
        name="weak_insert",
    )
    simulation.add_pml(z=0.8e-6, order=3, target_reflection=1e-8)

    sweep = simulation.sweep_frequencies(
        frequencies_hz,
        h5_path="frequency_sweep.h5",
        mesh_options={"wavelength_elements": 8},
        mode_options={"max_refinements": 0, "num_modes": 1, "neff_guess": 1.0},
    )

    print("frequency (Hz) =", sweep.frequencies_hz)
    print("S11 =", sweep.S11)
    print("S21 =", sweep.S21)
    print("R =", sweep.reflection)
    print("T =", sweep.transmission)
    print("power-balance error =", sweep.power_balance_error)
    print("HDF5 sweep =", sweep.h5_path)

    sweep.visualize_with_gui()


if __name__ == "__main__":
    main()
