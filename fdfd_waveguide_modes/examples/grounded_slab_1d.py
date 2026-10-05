"""grounded slab 1d: define materials, assign shapes, mesh, and solve."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fdfd_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path
from fdfd_common import Material, materials, shapes
from fdfd_waveguide_modes import ModeSolver1D

OUTPUT = _ROOT / "fdfd_waveguide_modes/outputs/examples/grounded_slab_1d"


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
