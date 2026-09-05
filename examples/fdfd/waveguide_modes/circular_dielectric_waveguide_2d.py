"""circular dielectric waveguide 2d: define materials, assign shapes, mesh, and solve."""
from pathlib import Path
from cem_common import Material, materials, shapes
from fdfd_waveguide_modes import ModeSolver2D

OUTPUT = Path(__file__).resolve().parents[3] / "outputs/examples/fdfd/waveguide_modes/circular_dielectric_waveguide_2d"


def build_solver():
    solver = ModeSolver2D(frequency=100e9, x_range=.01, y_range=.01, background_material=materials.vacuum)
    core = Material(name="dielectric core", epsilon=6.)
    solver.add_circle(center=(.005, .005), radius=.003, material=core, name="core")
    return solver


def main():
    solver = build_solver()
    solver.mesh(resolution=(50, 50))
    result = solver.solve(num_modes=4, neff_guess=2.)
    print("Effective indices:", result.neff)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / "modes.h5")
    result.show()
    return result


if __name__ == "__main__":
    main()
