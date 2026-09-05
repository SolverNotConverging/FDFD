"""rectangular waveguide 2d: define materials, assign shapes, mesh, and solve."""
from pathlib import Path
from cem_common import Material, materials, shapes
from fdfd_waveguide_modes import ModeSolver2D

OUTPUT = Path(__file__).resolve().parents[3] / "outputs/examples/fdfd/waveguide_modes/rectangular_waveguide_2d"


def build_solver():
    solver = ModeSolver2D(frequency=100e9, x_range=.012, y_range=.010, background_material=materials.vacuum)
    copper = materials.copper
    # One Boolean frame avoids overlapping conductor assignments at the corners.
    wall = shapes.Difference(
        shape=shapes.Rectangle(bounds=((.0019, .0101), (.0019, .0081))),
        tool=shapes.Rectangle(bounds=((.002, .010), (.002, .008))),
    )
    solver.add_geometry(shape=wall, material=copper, name="copper wall")
    return solver


def main():
    solver = build_solver()
    solver.mesh(resolution=(120, 100))
    result = solver.solve(num_modes=4, neff_guess=.99)
    print("Effective indices:", result.neff)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / "modes.h5")
    result.show()
    return result


if __name__ == "__main__":
    main()
