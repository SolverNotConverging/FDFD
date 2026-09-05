"""grounded slab 1d: define materials, assign shapes, mesh, and solve."""
from pathlib import Path
from cem_common import Material, materials, shapes
from fdfd_waveguide_modes import ModeSolver1D

OUTPUT = Path(__file__).resolve().parents[3] / "outputs/examples/fdfd/waveguide_modes/grounded_slab_1d"


def build_solver():
    solver = ModeSolver1D(frequency=30e9, x_range=.010, background_material=materials.vacuum)
    slab = Material(name="slab", epsilon=10.2)
    ground = materials.PEC
    solver.add_layer(x_range=(.003, .00427), material=slab)
    solver.add_layer(x_range=(.0029, .003), material=ground)
    solver.add_pml(thickness=.0008, sigma_max=10.)
    return solver


def main():
    solver = build_solver()
    solver.mesh(resolution=1000)
    result = solver.solve(num_modes=4, neff_guess=2.8)
    print("Effective indices:", result.neff)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / "modes.h5")
    result.show()
    return result


if __name__ == "__main__":
    main()
