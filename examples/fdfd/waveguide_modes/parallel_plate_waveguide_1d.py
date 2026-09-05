"""parallel plate waveguide 1d: define materials, assign shapes, mesh, and solve."""
from pathlib import Path
from cem_common import Material, materials, shapes
from fdfd_waveguide_modes import ModeSolver1D

OUTPUT = Path(__file__).resolve().parents[3] / "outputs/examples/fdfd/waveguide_modes/parallel_plate_waveguide_1d"


def build_solver():
    solver = ModeSolver1D(frequency=100e9, x_range=.008, background_material=materials.vacuum)
    dielectric = Material(name="anisotropic fill", epsilon=(4., 5., 6.))
    wall = materials.PMC
    solver.add_layer(x_range=(.003, .0045), material=dielectric)
    solver.add_layer(x_range=(.0029, .003), material=wall)
    solver.add_layer(x_range=(.0045, .0046), material=wall)
    solver.add_pml(thickness=.0008, sigma_max=10.)
    return solver


def main():
    solver = build_solver()
    solver.mesh(resolution=800)
    result = solver.solve(num_modes=4, neff_guess=2.)
    print("Effective indices:", result.neff)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / "modes.h5")
    result.show()
    return result


if __name__ == "__main__":
    main()
