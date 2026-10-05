"""parallel plate waveguide 1d: define materials, assign shapes, mesh, and solve."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "cem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path
from cem_common import Material, materials, shapes
from fdfd_waveguide_modes import ModeSolver1D

OUTPUT = _ROOT / "outputs/fdfd_waveguide_modes/examples/parallel_plate_waveguide_1d"


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
