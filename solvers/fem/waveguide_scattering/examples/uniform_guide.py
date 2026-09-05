"""Fixed-mesh full-vector lead modes and second-order uniform scattering."""

import fem_waveguide_scattering as wf


def main():
    simulation = wf.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.0, x_range=(0.0, 0.5), z_range=(-2.0, 2.0), transverse_boundary='pec')
    simulation.add_pml(thickness=0.5, direction='z')
    simulation.mesh(max_element_size=0.15, element_order=2)
    modes = simulation.solve_modes(num_modes=1, neff_guess=1., num_elements=16, max_refinements=0)
    simulation.set_incident_mode(0)
    result = simulation.solve(max_refinements=0)
    print("TEM effective index (exact 1):", modes[0].neff)
    print("Reflection (exact 0):", result.reflection)
    print("Transmission (exact 1):", result.transmission)
    result.show()
    return modes, result


if __name__ == "__main__":
    main()
