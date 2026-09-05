"""End-to-end scattered-field solve for a weak dielectric insert.

This deliberately small PEC guide runs quickly and exercises the same mixed
Nedelec--Lagrange assembly, z-PML, mode projection, and power accounting used
by open transverse examples.
"""

from __future__ import annotations
from cem_common import Material, materials

import fem_waveguide_scattering as scattering

from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs/examples/fem/waveguide_scattering" / Path(
    __file__,
).stem


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    insert = Material(name="weak dielectric insert", epsilon=1.002)
    simulation = scattering.WaveguideScatteringSolver2D(frequency=193414489000000.0, angle=0.0, x_range=(0.0, 1e-06), z_range=(-3e-06, 3e-06), background_material=materials.vacuum, boundary=materials.PEC)
    simulation.add_rectangle(x_range=(0.0, 1e-06), z_range=(-3e-07, 3e-07), name='weak_insert', material=insert)
    simulation.add_pml(order=3, target_reflection=1e-08, thickness=8e-07, direction='z')
    mesh = simulation.mesh(wavelength_elements=8)
    print("selected max element size =", mesh.info.requested_maximum_edge)

    modes = simulation.solve_modes(max_refinements=0, num_modes=1, neff_guess=1.0)
    simulation.set_incident_mode(modes[0])
    result = simulation.solve(max_refinements=0)
    output_path = result.save(OUTPUT_DIR / 'results.h5')

    print("beta =", modes[0].beta)
    print("S11 =", result.S11)
    print("S21 =", result.S21)
    print("power-balance error =", result.power_balance_error)
    print("HDF5 result =", output_path)
    print("diagnostics =", result.check())
    result.show()


if __name__ == "__main__":
    main()
