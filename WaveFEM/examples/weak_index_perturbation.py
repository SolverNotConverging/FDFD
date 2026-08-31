"""End-to-end scattered-field solve for a weak dielectric insert.

This deliberately small PEC guide runs quickly and exercises the same mixed
Nedelec--Lagrange assembly, z-PML, mode projection, and power accounting used
by open transverse examples.
"""

from __future__ import annotations

import wavefem as wf


def main() -> None:
    simulation = wf.Scattering2D(
        frequency=193.414489e12,
        ky=0.0,
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
    mesh = simulation.mesh(wavelength_elements=8)
    print("selected max element size =", mesh.info.requested_maximum_edge)

    modes = simulation.solve_modes(num_modes=1, neff_guess=1.0)
    simulation.set_incident_mode(modes[0])
    result = simulation.run(h5_path="weak_index_perturbation.h5")

    print("beta =", modes[0].beta)
    print("S11 =", result.S11)
    print("S21 =", result.S21)
    print("power-balance error =", result.power_balance_error)
    print("HDF5 result =", result.h5_path)
    print("diagnostics =", result.check())
    result.visualize_with_gui()


if __name__ == "__main__":
    main()
