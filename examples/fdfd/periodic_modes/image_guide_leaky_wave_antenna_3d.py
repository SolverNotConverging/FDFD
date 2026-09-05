"""image guide leaky wave antenna 3d with a material-first periodic unit cell.

The compact grid is a workflow demonstration; refine before interpreting leakage.
"""
from pathlib import Path
from cem_common import Material, materials
from fdfd_periodic_modes import PeriodicModeSolver3D

OUTPUT = Path(__file__).resolve().parents[3] / "outputs/examples/fdfd/periodic_modes/image_guide_leaky_wave_antenna_3d"


def build_solver(*, frequency=30e9):
    dielectric = Material(name="guide dielectric", epsilon=4.)
    solver = PeriodicModeSolver3D(frequency=frequency, x_range=.012, y_range=.008, z_range=.006)
    solver.add_box(x_range=(.004, .008), y_range=(.001, .004), z_range=(0., .006), material=dielectric, name="image guide")
    solver.add_box(x_range=(0., .012), y_range=(0., .001), z_range=(0., .006), material=materials.PEC, name="ground")
    solver.add_box(x_range=(.004, .008), y_range=(.004, .005), z_range=(.002, .004), material=materials.PEC, name="loading tooth")
    solver.add_pml(thickness=.0015, direction="x", sigma_max=1.)
    return solver


def main():
    solver = build_solver()
    solver.mesh(resolution=(12, 8, 8))
    result = solver.solve(num_modes=2, neff_guess=1.5, eigensolver="eigs")
    print("Bloch effective indices:", result.neff)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / "modes.h5")
    result.show()
    return result


if __name__ == "__main__":
    main()
