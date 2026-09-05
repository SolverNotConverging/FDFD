"""surface wave antenna 2d with a material-first periodic unit cell.

The compact grid is a workflow demonstration; refine before interpreting leakage.
"""
from pathlib import Path
from cem_common import Material, materials
from fdfd_periodic_modes import PeriodicModeSolver2D

OUTPUT = Path(__file__).resolve().parents[3] / "outputs/examples/fdfd/periodic_modes/surface_wave_antenna_2d"


def build_solver(*, frequency=25e9):
    dielectric = Material(name="guide dielectric", epsilon=4.)
    solver = PeriodicModeSolver2D(frequency=frequency, x_range=.01, z_range=.006, polarization="TM")
    solver.add_rectangle(x_range=(0., .002), z_range=(0., .006), material=dielectric, name="slab")
    solver.add_rectangle(x_range=(0., .0005), z_range=(0., .006), material=materials.PEC, name="ground")
    solver.add_rectangle(x_range=(.002, .0025), z_range=(.0015, .003), material=materials.PEC, name="loading tooth")
    solver.add_pml(thickness=.0015, direction="x", sigma_max=1.)
    return solver


def main():
    solver = build_solver()
    solver.mesh(resolution=(240, 160))
    result = solver.solve(num_modes=2, neff_guess=1.5, eigensolver="eigs")
    print("Bloch effective indices:", result.neff)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / "modes.h5")
    result.show()
    return result


if __name__ == "__main__":
    main()
