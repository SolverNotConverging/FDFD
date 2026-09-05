"""Solve the grounded-slab leaky-wave cell used by the FDFD example.

The lower face is the PEC ground, a high-permittivity slab fills the full
period, and a short PEC patch perturbs only the top of that slab.  Radiation
leaks into the air/PML region above the periodically repeated patches.
"""

from __future__ import annotations
from cem_common import Material, materials, shapes

from fem_periodic_modes import PeriodicModeSolver2D


MM = 1.0e-3


def main() -> None:
    substrate = Material(name="antenna substrate", epsilon=10.2)
    solver = PeriodicModeSolver2D(frequency=20000000000.0, x_range=(0.0, 10.0 * MM), z_range=(0.0, 8.0 * MM), polarization='TM', boundary=materials.PEC)
    solver.add_rectangle(x_range=(0.0, 1.27 * MM), z_range=(0.0, 8.0 * MM), name='grounded_dielectric_slab', material=substrate)
    solver.add_geometry(name='top_pec_perturbation', shape=shapes.Rectangle(bounds=((1.27 * MM, 1.32 * MM), (1.0 * MM, 2.0 * MM))), material=materials.PEC)
    solver.add_pml(thickness=2.5 * MM, direction='x+')
    solver.mesh(max_element_size=0.35 * MM)

    modes = solver.solve(
        max_refinements=0,
        direction='all',
        eigensolver='auto',
        max_pml_fraction=None,
        num_modes=4,
        neff_guess=0,
    )
    print("neff:", modes.neff)
    print("PML energy fractions:", [mode.pml_fraction for mode in modes])

    solver.show()


if __name__ == "__main__":
    main()
