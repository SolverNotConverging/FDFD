"""Open transverse slab guide at a nonzero oblique propagation angle."""

from __future__ import annotations

import numpy as np

import fem_waveguide_scattering as wf


def main() -> None:
    frequency_hz = 193.414489e12
    simulation = wf.WaveguideScatteringSolver2D(frequency=frequency_hz, angle=np.degrees(np.arcsin(0.1)), x_range=(-1.5e-06, 1.5e-06), z_range=(-2.5e-06, 2.5e-06), background_epsilon=1.44 ** 2)
    simulation.add_rectangle(x_range=(-2.2e-07, 2.2e-07), z_range='all', epsilon=3.45 ** 2, background=True, name='core')
    simulation.add_rectangle(x_range=(-2.2e-07, 2.2e-07), z_range=(-2.5e-07, 2.5e-07), epsilon=3.451 ** 2, name='perturbation')
    simulation.add_pml(thickness=3.5e-07, direction='x')
    simulation.add_pml(thickness=6.5e-07, direction='z')
    simulation.mesh(wavelength_elements=9)

    modes = simulation.solve_modes(max_refinements=0, num_modes=1, neff_guess=3.2, num_elements=54)
    incident = simulation.set_incident_mode(modes[0])
    result = simulation.solve(max_refinements=0, linear_solver_tolerance=1e-08)
    output_path = result.save('oblique_angle.h5')

    print("propagation angle (deg) =", simulation.angle)
    print("resolved ky =", simulation.ky)
    print("z-directed neff =", incident.mode.neff)
    print("S11 =", result.S11)
    print("S21 =", result.S21)
    print("power-balance error =", result.power_balance_error)
    print("HDF5 result =", output_path)
    print("diagnostics =", result.check())
    result.show()


if __name__ == "__main__":
    main()
