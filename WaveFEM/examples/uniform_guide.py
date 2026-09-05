"""Fixed-mesh full-vector lead modes and second-order uniform scattering."""

import wavefem as wf


def main():
    cross_section = wf.CrossSection(x_span=(0., .5), background=wf.Material(), boundary="pec")
    lead_solver = wf.ModeSolver(cross_section, wavelength=1., num_elements=16)
    modes = lead_solver.solve(num_modes=1, neff_guess=1., max_refinements=0)
    simulation = wf.Scattering2D(
        wavelength=1., x_span=(0., .5), z_span=(-2., 2.), transverse_boundary="pec",
        solver_options=wf.SolverOptions(element_order=2, max_refinements=0),
    )
    simulation.add_pml(z=.5)
    simulation.mesh(max_element_size=.15)
    simulation.set_modes(modes)
    simulation.set_incident_mode(0)
    result = simulation.solve(max_refinements=0)
    print("TEM effective index (exact 1):", modes[0].neff)
    print("Reflection (exact 0):", result.reflection)
    print("Transmission (exact 1):", result.transmission)
    result.visualize_with_gui()
    return modes, result


if __name__ == "__main__":
    main()
