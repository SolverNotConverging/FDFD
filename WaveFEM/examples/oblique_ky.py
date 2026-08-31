"""Open transverse slab guide with nonzero prescribed ``ky``."""

from __future__ import annotations

import numpy as np

import wavefem as wf


def main() -> None:
    frequency_hz = 193.414489e12
    k0 = 2.0 * np.pi * frequency_hz / wf.C0
    simulation = wf.Scattering2D(
        frequency=frequency_hz,
        ky=0.10 * k0,
        x_span=(-1.5e-6, 1.5e-6),
        z_span=(-2.5e-6, 2.5e-6),
        background_eps=1.44**2,
        # Sparse direct solves can accumulate a few 1e-10 of normalized
        # residual on this oblique, PML-stretched mesh.  Keep the example's
        # acceptance threshold comfortably below plotting accuracy.
        solver_options=wf.SolverOptions(tolerance=1.0e-8),
    )
    simulation.add_rectangle(
        x=(-0.22e-6, 0.22e-6),
        z="all",
        eps=3.45**2,
        background=True,
        name="core",
    )
    simulation.add_rectangle(
        x=(-0.22e-6, 0.22e-6),
        z=(-0.25e-6, 0.25e-6),
        eps=3.451**2,
        name="perturbation",
    )
    simulation.add_pml(x=0.35e-6, z=0.65e-6)
    simulation.mesh(wavelength_elements=9)

    modes = simulation.solve_modes(num_modes=1, neff_guess=3.2, num_elements=54)
    simulation.set_incident_mode(modes[0])
    result = simulation.run(h5_path="oblique_ky.h5")

    print("neff =", modes[0].neff)
    print("S11 =", result.S11)
    print("S21 =", result.S21)
    print("power-balance error =", result.power_balance_error)
    print("HDF5 result =", result.h5_path)
    print("diagnostics =", result.check())
    result.visualize_with_gui()


if __name__ == "__main__":
    main()
